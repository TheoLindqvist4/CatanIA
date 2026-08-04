"""Head-to-head matches, played across processes.

:func:`catan.agents.play_match` plays one game at a time, which is right for a heuristic —
a game costs 56 ms — and impractical for an agent that searches. At 32 simulations a move a
single game costs **5.2 seconds**, so the 400-game promotion match the design guide asks for
is over half an hour per rung, and the guide's recommended 1,000 games is an hour and a half.
The gate then costs more than the training run it is gating, which is how a gate stops being
run.

This plays the same match on ``workers`` processes. Two things make it worth having rather
than obvious:

**The result does not depend on how many workers ran it.** Game *g* uses seed ``seed + g``
and seat shift ``g % 2``, exactly as ``play_match`` does, and games are handed out by index —
so which worker plays which game changes nothing. ``tests/test_alphazero.py`` asserts a
4-worker match and a 1-worker match give the identical tally, because an evaluator whose
answer moves with the machine it ran on is not an evaluator.

That required one change of substance. Agents are **stateful**: every one of them holds an
RNG seeded once at construction, and ``play_match`` lets that state carry from game to game,
so game 17's result depends on all sixteen before it. That is fine sequentially and
impossible to reproduce in parallel. So every agent is **re-seeded per game**, from the game's
index. The consequence is worth stating plainly rather than burying: a match played here is
*not* the same draw as the same match played by ``play_match``, it is an independent sample
of the same quantity. Measured on a heuristic-versus-greedy match the two agree to within the
interval, as they must, but they are not the same number.

**Agents are specifications, not objects.** Windows spawns rather than forks, so everything
crossing the boundary is pickled — and a loaded network is 780 KB per agent per game. Each
worker is told *what* to build (``{"kind": "mcts", "path": ...}``) and builds it once, at
startup, then reuses it for its whole slice.

Torch threads are pinned to one per worker for the reason recorded in ``CLAUDE.md``: the
OpenMP pool is sized at import, so N workers on a 20-core box would otherwise ask for 14N
threads and spend their time in the scheduler.
"""

import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from catan.agents import play_game
from catan.rulesets import RANKED_1V1
from training.evaluate import confidence_interval

#: Per-process state: the two agents, built once.
_ARENA = {}


def build_agent(spec):
    """An agent from a picklable description.

    ``{"kind": "heuristic", "noise": 0}``
    ``{"kind": "mcts", "path": ..., "simulations": 32, "temperature": 0.0, "seed": 0}``
    ``{"kind": "policy", "path": ...}``            a network with no search
    ``{"kind": "ppo_champion"}``                   models/champion.pt, grafted if it predates
                                                   the current observation
    """
    kind = spec["kind"]
    if kind == "heuristic":
        from catan.agents import HeuristicAgent

        return HeuristicAgent(spec.get("seed", 0), noise=spec.get("noise", 0))
    if kind == "greedy":
        from catan.agents import GreedyAgent

        return GreedyAgent(spec.get("seed", 0))
    if kind == "random":
        from catan.agents import RandomAgent

        return RandomAgent(spec.get("seed", 0))
    if kind == "mcts":
        from training.alphazero.agent import MCTSAgent

        return MCTSAgent.load(spec["path"], simulations=spec.get("simulations", 32),
                              temperature=spec.get("temperature", 0.0),
                              seed=spec.get("seed", 0))
    if kind == "policy":
        from training.agent import PolicyAgent

        return PolicyAgent.load(spec["path"], temperature=spec.get("temperature", 0.0),
                                seed=spec.get("seed", 0))
    if kind == "ppo_champion":
        from training.alphazero.champion import load_previous_technique

        agent = load_previous_technique(temperature=spec.get("temperature", 0.0),
                                        seed=spec.get("seed", 0))
        if agent is None:
            raise ValueError("there is no usable PPO champion to play against")
        return agent
    raise ValueError(f"unknown agent kind {kind!r}")


def _configure(payload):
    """Runs once per child: pin threads, build both agents."""
    import torch

    torch.set_num_threads(1)
    settings = pickle.loads(payload)
    _ARENA.clear()
    _ARENA.update({
        "agents": {1: build_agent(settings["a"]), 2: build_agent(settings["b"])},
        "seed": settings["seed"],
        "ruleset": settings["ruleset"],
        "max_turns": settings["max_turns"],
    })


def reseed(agent, seed):
    """Put ``agent`` into a known random state, whatever kind of agent it is.

    Every agent in this project holds its own generator, and they are not the same type:
    the engine's agents use ``random.Random``, the learned ones use a numpy Generator and a
    ``random.Random`` for determinization. Set whichever exist and ignore the rest — an agent
    with no randomness at all is a legitimate case, not an error.
    """
    if hasattr(agent, "rng"):
        agent.rng = (np.random.default_rng(seed)
                     if isinstance(agent.rng, np.random.Generator)
                     else random.Random(seed))
    if hasattr(agent, "_stream"):
        agent._stream = random.Random(seed + 1)
    return agent


def _play(indices):
    """Play the games with these indices. Returns ``(wins_a, wins_b, truncated)``."""
    state = _ARENA
    agents, players = state["agents"], [1, 2]
    wins = {1: 0, 2: 0}
    truncated = 0
    for game in indices:
        # Same seating as play_match. Re-seeding per game is what makes the answer
        # independent of which worker drew this index — see the module docstring.
        shift = game % 2
        for offset, agent in enumerate(agents.values()):
            reseed(agent, state["seed"] + game * 31 + offset)
        seated = {players[i]: agents[players[(i + shift) % 2]] for i in range(2)}
        info = play_game(seated, seed=state["seed"] + game, ruleset=state["ruleset"],
                         max_turns=state["max_turns"])
        if info["winner"] is None:
            truncated += 1
        else:
            index = players.index(info["winner"])
            wins[players[(index + shift) % 2]] += 1
    return wins[1], wins[2], truncated


def compete(a, b, games=300, seed=10_000, workers=None, ruleset=RANKED_1V1,
            max_turns=800):
    """Play ``a`` against ``b``, seats swapped every other game, across processes.

    Args:
        a, b: agent specifications — see :func:`build_agent`.
        workers: processes. Defaults to leaving four cores free, because this is usually run
            while something else is using the machine.

    Returns:
        The same dict :func:`training.evaluate.evaluate` returns, so callers and log formats
        do not care which one produced it.
    """
    workers = max(1, (os.cpu_count() or 4) - 4) if workers is None else workers
    workers = max(1, min(workers, games))
    payload = pickle.dumps({"a": a, "b": b, "seed": seed, "ruleset": ruleset,
                            "max_turns": max_turns}, protocol=pickle.HIGHEST_PROTOCOL)

    slices = [list(range(start, games, workers)) for start in range(workers)]

    previous = {}
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        previous[name] = os.environ.get(name)
        os.environ[name] = "1"
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=_configure,
                                 initargs=(payload,)) as pool:
            parts = list(pool.map(_play, slices))
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    wins = sum(p[0] for p in parts)
    losses = sum(p[1] for p in parts)
    truncated = sum(p[2] for p in parts)
    decided = wins + losses
    low, high = confidence_interval(wins, decided)
    return {
        "wins": wins, "losses": losses, "truncated": truncated, "games": games,
        "win_rate": wins / decided if decided else 0.0,
        "ci": (low, high), "ci_width": high - low,
    }


# --------------------------------------------------------------------------- #
# Choosing what to submit to the gate                                         #
# --------------------------------------------------------------------------- #

def rank(paths, opponent=None, games=150, simulations=32, seed=30_000, workers=None,
         log=print):
    """Play each checkpoint against a fixed opponent and return them best-first.

    This is the step that D17 in the decision record exists to insist on. A run's ``best.pt``
    is chosen by the in-loop evaluation, which for cost reasons scores the **raw policy** —
    and the champion plays *with search*. Measured on one run those two moved in opposite
    directions, so "best policy" is not "best player" and picking ``best.pt`` unexamined can
    promote the wrong network.

    Every candidate plays the **same games** — same seed, same opponent, same seat rotation —
    so the comparison between them is paired rather than two independent samples.
    """
    opponent = {"kind": "heuristic", "noise": 0} if opponent is None else opponent
    results = []
    for path in paths:
        result = compete({"kind": "mcts", "path": str(path), "simulations": simulations},
                         opponent, games=games, seed=seed, workers=workers)
        result["path"] = str(path)
        results.append(result)
        low, high = result["ci"]
        log(f"  {str(path):<48} {100 * result['win_rate']:5.1f}%  "
            f"[{100 * low:4.1f}, {100 * high:4.1f}]  trunc {result['truncated']:>3}")
    return sorted(results, key=lambda r: r["win_rate"], reverse=True)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Play checkpoints against a fixed opponent, best first",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--games", type=int, default=150)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--seed", type=int, default=30_000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--against", default="heuristic",
                        choices=["heuristic", "ppo_champion", "greedy", "random"])
    arguments = parser.parse_args(argv)

    print(f"{arguments.games} games each at {arguments.simulations} simulations, "
          f"against {arguments.against}:")
    ordered = rank(arguments.checkpoints,
                   opponent={"kind": arguments.against,
                             **({"noise": 0} if arguments.against == "heuristic" else {})},
                   games=arguments.games, simulations=arguments.simulations,
                   seed=arguments.seed, workers=arguments.workers)
    print(f"\nbest: {ordered[0]['path']}  ({100 * ordered[0]['win_rate']:.1f}%)")
    return 0


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    raise SystemExit(main())

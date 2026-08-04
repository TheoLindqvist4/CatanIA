"""How fast this thing actually is.

    python -m benchmark.benchmark                 # every measurement
    python -m benchmark.benchmark engine          # the simulator alone
    python -m benchmark.benchmark selfplay        # search + network, one process
    python -m benchmark.benchmark scaling         # throughput against worker count

The guide's section 6 asks for games/sec, turns/sec, ms/game and the machine's utilisation,
and section 7 gives targets: under 100 ms/game to start, under 20 to be comfortable, under 5
to be excellent. Those targets are about the *simulator*, and the distinction matters more
than it looks:

* A game of **random** play takes 56 ms here — but it also takes 679 turns, because random
  play flails at 15 points. Per *decision* the engine does about 36,000 a second.
* A game of **AlphaZero self-play** takes far longer than 56 ms and always will, because each
  decision costs dozens of network evaluations. Its ms/game figure is a property of the
  simulation count, not of the engine.

So both are reported, separately, and neither is allowed to stand in for the other. The number
that decides how a training run should be configured is ``positions/sec``, which is what the
replay buffer is actually fed with.

**Warm-up is not optional.** Self-play keeps games in flight between calls, so a cold
measurement measures a pipeline that is still filling. ``CLAUDE.md`` records an earlier
benchmark in this repository reporting 4 workers as faster than 8 for exactly that reason.
Every measurement here runs a discarded round first.
"""

import argparse
import os
import random
import time

from catan import action_space, rules
from catan.rulesets import RANKED_1V1
from catan.state import GameState, Phase


def _resources():
    """CPU and memory use, when the platform will say. ``None`` rather than a guess.

    The guide asks for both. ``psutil`` is an optional dependency rather than a required one —
    the engine has no dependencies at all and a speed measurement is a poor reason to give
    that up — so these fields are simply absent when it is not installed. Absent, not
    estimated: a made-up utilisation figure is worse than no figure.
    """
    try:
        import psutil                                   # optional; see requirements.txt
    except ImportError:
        return {"cpu_percent": None, "memory_mb": None}
    process = psutil.Process()
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "memory_mb": round(process.memory_info().rss / 1e6, 1),
    }


# --------------------------------------------------------------------------- #
# The simulator on its own                                                    #
# --------------------------------------------------------------------------- #

def play_random_game(seed, max_actions=20_000, prefer_building=True):
    """One game of uniformly random legal play. Returns ``(actions, turns, finished)``."""
    state = GameState(num_players=2, seed=seed, ruleset=RANKED_1V1)
    rng = random.Random(seed ^ 0x5EED)
    actions = 0
    for _ in range(max_actions):
        if state.phase is Phase.GAME_OVER:
            break
        if state.phase is Phase.ROLL:
            pre_roll = rules.legal_actions(state)
            if pre_roll and rng.random() < 0.3:
                rules.apply(state, rng.choice(pre_roll))
            else:
                rules.roll_dice(state)
        else:
            legal = rules.legal_actions(state)
            if not legal:
                break
            if prefer_building and len(legal) > 1:
                builds = [a for a in legal if a != rules.end_turn()]
                legal = builds or legal
            rules.apply(state, rng.choice(legal))
        actions += 1
    return actions, state.turn_number, state.winner is not None


def engine(games=300, seed=0, log=print):
    """Games per second with no network in the loop. The guide's headline measurement."""
    play_random_game(seed)                                   # warm-up: import, first board
    started = time.perf_counter()
    actions = turns = finished = 0
    for game in range(games):
        a, t, done = play_random_game(seed + game)
        actions += a
        turns += t
        finished += done
    elapsed = time.perf_counter() - started

    result = {
        "games": games,
        "elapsed": round(elapsed, 2),
        "games_per_second": round(games / elapsed, 1),
        "turns_per_second": round(turns / elapsed, 0),
        "actions_per_second": round(actions / elapsed, 0),
        "ms_per_game": round(1000 * elapsed / games, 2),
        "average_turns": round(turns / games, 0),
        "average_actions": round(actions / games, 0),
        "finished": finished,
        **_resources(),
    }
    _report("engine, uniformly random play", result, log)
    log(f"  target band: <100 ms initial, <20 good, <5 excellent  "
        f"(this is {result['ms_per_game']:.1f} ms over {result['average_turns']:.0f} turns; "
        f"random play needs many more turns than a real policy)")
    return result


# --------------------------------------------------------------------------- #
# Self-play, which is what a training run actually runs                       #
# --------------------------------------------------------------------------- #

def selfplay(positions=1_500, simulations=48, width=24, seed=0, checkpoint=None,
             log=print):
    """One worker's throughput: positions and games per second, with the search in the loop."""
    import numpy as np
    import torch

    torch.set_num_threads(1)
    from training.alphazero.network import load_for_alphazero, new_network
    from training.alphazero.self_play import Generator

    if checkpoint:
        net, _ = load_for_alphazero(checkpoint)
    else:
        net = new_network()
    net.eval()

    def evaluate(obs, masks):
        with torch.no_grad():
            logits, value = net(torch.from_numpy(obs))
            logits = net._apply_mask(logits, torch.from_numpy(masks))
            return torch.softmax(logits, dim=-1).numpy(), value.numpy()

    generator = Generator(evaluate, {"simulations": simulations}, seed=seed, width=width)
    generator.run(positions=positions // 4)                  # warm-up: fill the pipeline

    started = time.perf_counter()
    samples, results = generator.run(positions=positions)
    elapsed = time.perf_counter() - started

    turns = sum(r["turns"] for r in results)
    result = {
        "simulations": simulations,
        "width": width,
        "positions": len(samples),
        "games": len(results),
        "elapsed": round(elapsed, 2),
        "positions_per_second": round(len(samples) / elapsed, 1),
        "games_per_second": round(len(results) / elapsed, 2),
        "evaluations_per_second": round(len(samples) * simulations / elapsed, 0),
        "ms_per_game": round(1000 * elapsed / len(results), 0) if results else None,
        "average_turns": round(turns / len(results), 0) if results else None,
        "searched_fraction": (
            round(sum(r["searched"] for r in results)
                  / max(1, sum(r["decisions"] for r in results)), 3) if results else None
        ),
        **_resources(),
    }
    _report(f"self-play, one process, {simulations} simulations/move", result, log)
    return result


def scaling(workers=(1, 2, 4, 8, 14), positions=2_000, simulations=48, width=24, seed=0,
            log=print):
    """Throughput against worker count. Run this before believing any speed-up.

    Each row is a fresh pool, warmed up and then measured, so the cost of spawning twenty
    Windows processes is not charged to the throughput.
    """
    import torch

    from training.alphazero.config import Config
    from training.alphazero.network import new_network
    from training.alphazero.workers import ParallelSelfPlay

    net = new_network()
    results = {}
    for count in workers:
        config = Config({
            "self_play_workers": count,
            "envs_per_worker": width,
            "mcts_simulations": simulations,
        })
        pool = ParallelSelfPlay(net, config)
        try:
            pool.generate(positions // 2)                    # warm-up
            started = time.perf_counter()
            arrays, games = pool.generate(positions)
            elapsed = time.perf_counter() - started
        finally:
            pool.close()
        made = len(arrays[0])
        results[count] = {
            "positions_per_second": round(made / elapsed, 1),
            "games_per_second": round(len(games) / elapsed, 2),
            "positions": made,
            "games": len(games),
            "elapsed": round(elapsed, 2),
        }
        log(f"  {count:>3} workers  {made:>6,} positions / {elapsed:5.1f}s = "
            f"{results[count]['positions_per_second']:>7,.0f}/sec, "
            f"{results[count]['games_per_second']:>5.2f} games/sec")
    return results


# --------------------------------------------------------------------------- #

def required_games(ms_per_game, cores, target=1_000_000):
    """Section 16's arithmetic: how long ``target`` games would take. For planning a run.

    ``gps = cores * 1000 / ms_per_game``. Worth doing *before* a long run rather than after:
    the difference between "this finishes tonight" and "this finishes on Thursday" is one
    multiplication, and it is the number that should decide the simulation count.

    The parallel efficiency this assumes is not 1. Measured here, fourteen workers deliver
    about five times one worker's throughput rather than fourteen, so halve the answer and
    then be pleasantly surprised.
    """
    per_second = cores * 1000 / ms_per_game
    return {
        "games_per_second": round(per_second, 1),
        "seconds": round(target / per_second, 0),
        "minutes": round(target / per_second / 60, 1),
        "hours": round(target / per_second / 3600, 2),
    }


def plan(result, cores=None, targets=(10_000, 100_000, 1_000_000), log=print):
    """Print section 16's table for a measured ``ms_per_game``."""
    cores = os.cpu_count() if cores is None else cores
    ms = result.get("ms_per_game")
    if not ms:
        return {}
    log("")
    log(f"at {ms:.0f} ms/game on {cores} cores, ignoring parallel inefficiency:")
    out = {}
    for target in targets:
        out[target] = required_games(ms, cores, target)
        log(f"  {target:>10,} games  {out[target]['hours']:>8.2f} h  "
            f"({out[target]['games_per_second']:.0f} games/sec)")
    return out


def _report(title, result, log):
    log(f"\n{title}")
    for key, value in result.items():
        if value is None:
            continue
        log(f"  {key:<24} {value}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Speed measurement",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("what", nargs="?", default="all",
                        choices=["all", "engine", "selfplay", "scaling"])
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--positions", type=int, default=1_500)
    parser.add_argument("--simulations", type=int, default=48)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--workers", type=int, nargs="*", default=None)
    parser.add_argument("--checkpoint", default=None)
    arguments = parser.parse_args(argv)

    print(f"cores: {os.cpu_count()}")
    if arguments.what in ("all", "engine"):
        engine(games=arguments.games)
    if arguments.what in ("all", "selfplay"):
        measured = selfplay(positions=arguments.positions, simulations=arguments.simulations,
                            width=arguments.width, checkpoint=arguments.checkpoint)
        plan(measured)
    if arguments.what in ("all", "scaling"):
        counts = arguments.workers or [1, 2, 4, 8, 14]
        print("\nscaling across processes")
        scaling(workers=counts, positions=arguments.positions,
                simulations=arguments.simulations, width=arguments.width)
    return 0


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    raise SystemExit(main())

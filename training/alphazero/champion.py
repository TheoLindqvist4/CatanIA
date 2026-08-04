"""The AlphaZero model the game plays against.

A second champion, beside the first. ``models/champion.pt`` is the PPO lineage and is not
touched by anything in this package; ``models/champion_az.pt`` is this one. Two files rather
than one because they are trained by different methods and the interesting question — which
of them a person should be playing — is answered by a match, not by whichever finished last.

    checkpoints/alphazero/   scratch. A run owns it, rewrites it, may ruin it. Not in git.
    models/champion_az.pt    the champion. Changes only by promotion. In git.

The interfaces read ``models/`` and never ``checkpoints/``, so a training run cannot disturb
a game in progress. Installation is a rename over a fully written file, so a game that loads
the champion mid-promotion gets either the old one or the new one and never half of either.

**The first promotion is gated too, and that is the point of this module existing rather
than reusing :mod:`training.champion`.** ``CLAUDE.md`` records the hole in the older gate: when
no champion loads, ``promote`` takes its ``reigning is None`` branch and installs
*immediately* — no Wilson bound, no regression check — and then writes the ``beat_heuristic``
baseline that every later candidate is measured against. It fires exactly when the encoder has
changed, which is exactly when nobody is watching for it. Here, a first candidate still has to
beat the fixed heuristic with its Wilson lower bound above 50% before it is installed, and the
record says in so many words that it was the first.
"""

import argparse
import datetime
import json
import pathlib

from catan import action_space, encoder

MODELS = pathlib.Path("models")

#: This lineage's champion. Deliberately not ``champion.pt``.
CHAMPION = MODELS / "champion_az.pt"
RECORD = MODELS / "champion_az.json"

#: The other lineage, read-only from here.
PPO_CHAMPION = MODELS / "champion.pt"

#: Games in a promotion match. 400 gives a Wilson interval of about +-4.9 points, so a
#: candidate has to win around 55% for the lower bound to clear 50% — the guide's threshold,
#: arrived at from the statistics rather than chosen.
#:
#: This was 300 while matches were sequential. A searching agent runs the network once per
#: simulation at batch 1, so a game costs 5.2 seconds at 32 simulations against 0.29 without
#: search, and 400 games is half an hour *per rung* with three rungs to play — which is how a
#: gate stops being run. :mod:`training.alphazero.arena` plays the match across processes and
#: brings a rung back to a couple of minutes, so the number is set by what makes a good gate
#: rather than by what fits in an afternoon.
PROMOTION_GAMES = 400

#: Simulations the champion is measured at **and played at**. A win rate is a property of the
#: ``(weights, simulations)`` pair, so measuring at one number and playing at another would
#: publish a figure for a player nobody faces. The interfaces import this rather than keeping
#: their own copy.
CHAMPION_SIMULATIONS = 32

#: How far a candidate may fall against the fixed baseline before promotion is refused, even
#: if it beat the champion.
MAX_BASELINE_REGRESSION = 0.05


def load(path=None, simulations=CHAMPION_SIMULATIONS, temperature=0.0, seed=None):
    """The AlphaZero champion as a playable agent, or ``None`` if there is not a usable one.

    Never raises. A missing file, a missing PyTorch, or a model built for a different
    observation or action space all mean the same thing to a caller: offer something else.
    A checkpoint from before an encoder change loads perfectly well and then fails on the
    first move, which is the failure this exists to prevent.

    ``path`` defaults to :data:`CHAMPION` but is resolved *when called*, not when this
    function was defined. Written ``path=CHAMPION`` it would capture the module constant at
    import, so redirecting the champion — which a test does on every run, and which anyone
    debugging a candidate will try — would silently keep reading the real one.
    """
    path = pathlib.Path(CHAMPION if path is None else path)
    if not path.is_file():
        return None
    from training.alphazero.agent import MCTSAgent

    agent = None
    try:
        agent = MCTSAgent.load(path, simulations=simulations, temperature=temperature,
                               seed=seed)
    except Exception:
        # Falls through to the graft. The network's constructor *raises* when `obs_size`
        # does not match the encoder, so an observation change lands here rather than
        # producing a loaded-but-wrong-shaped model — which is why the graft cannot be
        # attempted only after a successful load.
        agent = None

    if agent is None or agent.net.obs_size != encoder.SIZE:
        # The observation changed under a champion promoted against the old one. That used
        # to mean "offer the heuristic instead", and it is how both interfaces silently lost
        # their learned opponent once already. It does not have to: every change to this
        # observation appends, so `network.graft` gives the new columns zero weight and the
        # result computes *exactly* the function that was measured. Verified rather than
        # asserted — this champion scored 74.7% before the change and 75.6% after, over 400
        # and 200 games. See ``docs/decisions/0024-what-a-placement-can-see.md``.
        try:
            from training.alphazero.network import load_for_alphazero

            net, _ = load_for_alphazero(path)
            agent = MCTSAgent(net, simulations=simulations, temperature=temperature,
                              seed=seed)
        except Exception:
            return None

    if agent.net.num_actions != action_space.NUM_ACTIONS:
        return None                       # the action space moved; nothing can save that
    if agent.net.obs_size != encoder.SIZE:
        return None
    return agent


def load_previous_technique(temperature=0.0, seed=None):
    """The PPO champion, grafted onto the current observation if it predates it.

    ``training.champion.load`` deliberately refuses a checkpoint whose observation does not
    match, and that is right for *its* callers — a stale model plays nonsense. But the
    AlphaZero ladder needs to play the previous technique to know whether it has beaten it,
    and the graft is exact: the new observation columns are given zero weight, so the grafted
    network computes the same function on the same position. See
    :func:`training.alphazero.network.graft`.

    Returns ``None`` when there is no PPO champion or it cannot be reconciled.
    """
    if not PPO_CHAMPION.is_file():
        return None
    try:
        from training.agent import PolicyAgent
        from training.alphazero.network import load_for_alphazero
        import torch

        checkpoint = torch.load(PPO_CHAMPION, map_location="cpu", weights_only=False)
        if checkpoint["config"].get("obs_size") == encoder.SIZE:
            return PolicyAgent.load(PPO_CHAMPION, temperature=temperature, seed=seed)
        # Grafted, and kept as a *policy* agent: the value head was reset by the graft, and
        # a PolicyAgent never reads it. Playing it as an MCTSAgent would search with a value
        # head that has never been trained, which measures the graft rather than the model.
        net, _ = load_for_alphazero(PPO_CHAMPION, value_activation="linear")
        return PolicyAgent(net, temperature=temperature, seed=seed)
    except Exception:
        return None


def record():
    """What is known about the reigning AlphaZero champion, or ``{}``."""
    if not RECORD.is_file():
        return {}
    try:
        return json.loads(RECORD.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def describe():
    """One line about the champion, for a startup message or a CLI."""
    if load() is None:
        return "no AlphaZero champion yet"
    info = record()
    parts = [f"AlphaZero champion from {info.get('promoted_at', 'an unknown date')}"]
    if info.get("beat_heuristic") is not None:
        parts.append(f"{100 * info['beat_heuristic']:.1f}% vs the heuristic")
    if info.get("beat_ppo_champion") is not None:
        parts.append(f"{100 * info['beat_ppo_champion']:.1f}% vs the PPO champion")
    return ", ".join(parts)


# --------------------------------------------------------------------------- #
# Promotion                                                                   #
# --------------------------------------------------------------------------- #

def promote(candidate_path, games=PROMOTION_GAMES, seed=41_000,
            simulations=CHAMPION_SIMULATIONS, force=False, log=print):
    """Install ``candidate_path`` as the AlphaZero champion if it earns the place.

    Returns ``(promoted, reason)``.
    """
    from training.alphazero.arena import compete
    from training.alphazero.evaluator import better, format_result

    if load(candidate_path, simulations=simulations) is None:
        return False, f"{candidate_path} is not a usable model for this engine"

    MODELS.mkdir(parents=True, exist_ok=True)
    reigning_exists = load(simulations=simulations) is not None

    # Matches run across processes. Sequentially, a 300-game match at 32 simulations is 26
    # minutes *per rung*, which is how a gate stops being run — see
    # :mod:`training.alphazero.arena`.
    me = {"kind": "mcts", "path": str(candidate_path), "simulations": simulations}

    log(f"candidate: {candidate_path} at {simulations} simulations/move")
    log(f"{games} games against the heuristic:")
    against_baseline = compete(me, {"kind": "heuristic", "noise": 0}, games=games, seed=seed)
    log("  " + format_result("heuristic", against_baseline))

    against_ppo = None
    if load_previous_technique() is not None:
        log(f"{games} games against the PPO champion:")
        against_ppo = compete(me, {"kind": "ppo_champion"}, games=games, seed=seed + 1)
        log("  " + format_result("ppo", against_ppo))

    results = {
        "beat_heuristic": against_baseline["win_rate"],
        "beat_ppo_champion": None if against_ppo is None else against_ppo["win_rate"],
        "beat_champion": None,
        "games": games,
        "simulations": simulations,
    }

    if force:
        _install(candidate_path, {**results, "forced": True})
        return True, "forced"

    if not reigning_exists:
        # The hole in the older gate, closed. A first candidate is still measured; it just
        # has nothing of its own lineage to be measured against, so the fixed baseline is
        # the whole test rather than a side condition.
        if not better(against_baseline):
            low, high = against_baseline["ci"]
            return False, (
                f"first AlphaZero candidate, so the heuristic is the whole gate: "
                f"{100 * against_baseline['win_rate']:.1f}% with the interval "
                f"[{100 * low:.1f}, {100 * high:.1f}] — not shown better than the baseline"
            )
        _install(candidate_path, {**results, "first_of_lineage": True})
        return True, (f"first AlphaZero champion: "
                      f"{100 * against_baseline['win_rate']:.1f}% against the heuristic "
                      f"(lower bound {100 * against_baseline['ci'][0]:.1f}%)")

    log(f"{games} games against the reigning AlphaZero champion:")
    against_champion = compete(
        me, {"kind": "mcts", "path": str(CHAMPION), "simulations": simulations},
        games=games, seed=seed + 2)
    log("  " + format_result("champion", against_champion))
    results["beat_champion"] = against_champion["win_rate"]

    if not better(against_champion):
        low, high = against_champion["ci"]
        return False, (f"beat the champion {100 * against_champion['win_rate']:.1f}% but the "
                       f"interval [{100 * low:.1f}, {100 * high:.1f}] includes 50% — "
                       f"not shown better")

    # And it must not have got there by learning the champion's habits. Self-play is
    # non-transitive; without this a policy can climb the ladder while getting worse.
    previous = record().get("beat_heuristic")
    if previous is not None:
        drop = previous - against_baseline["win_rate"]
        if drop > MAX_BASELINE_REGRESSION:
            return False, (f"beat the champion but fell {100 * drop:.1f} points against the "
                           f"heuristic ({100 * previous:.1f}% -> "
                           f"{100 * against_baseline['win_rate']:.1f}%) — likely overfitted "
                           f"to the champion rather than better at the game")

    _install(candidate_path, results)
    return True, (f"promoted: {100 * against_champion['win_rate']:.1f}% against the champion "
                  f"(lower bound {100 * against_champion['ci'][0]:.1f}%)")


def _install(candidate_path, results):
    """Replace the champion atomically, so a game in progress never sees half a file."""
    import torch

    MODELS.mkdir(parents=True, exist_ok=True)
    source = torch.load(candidate_path, map_location="cpu", weights_only=False)

    staging = CHAMPION.with_suffix(".incoming")
    torch.save({
        "config": source["config"],
        "weights": source["weights"],
        "iteration": source.get("iteration"),
        "lineage": "alphazero",
    }, staging)
    staging.replace(CHAMPION)

    previous = record()
    entry = {
        "promoted_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "lineage": "alphazero",
        "source": str(candidate_path),
        "observation": encoder.SIZE,
        "actions": action_space.NUM_ACTIONS,
        **results,
    }
    entry["history"] = (previous.get("history", []) + [
        {k: v for k, v in previous.items() if k != "history"}
    ])[-10:] if previous else []
    RECORD.write_text(json.dumps(entry, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="The AlphaZero model the game plays against",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("show", help="what the AlphaZero champion is")
    show.set_defaults(func=lambda a: print(describe()) or
                      print(json.dumps(record(), indent=2) if record() else ""))

    run = sub.add_parser("promote", help="install a candidate if it is measurably better")
    run.add_argument("candidate")
    run.add_argument("--games", type=int, default=PROMOTION_GAMES)
    run.add_argument("--seed", type=int, default=41_000)
    run.add_argument("--simulations", type=int, default=CHAMPION_SIMULATIONS)
    run.add_argument("--force", action="store_true",
                     help="install without the match; for restoring a known-good model")
    run.set_defaults(func=_promote_command)

    arguments = parser.parse_args(argv)
    return arguments.func(arguments)


def _promote_command(arguments):
    import torch

    torch.set_num_threads(4)
    promoted, reason = promote(arguments.candidate, games=arguments.games,
                               seed=arguments.seed, simulations=arguments.simulations,
                               force=arguments.force)
    print(("PROMOTED — " if promoted else "kept the current champion — ") + reason)
    print(describe())
    return 0 if promoted else 1


if __name__ == "__main__":
    raise SystemExit(main())

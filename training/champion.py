"""The model the game actually plays against.

Training is destructive by nature: a run overwrites its own checkpoints, its *best so far*
is only the best within that run, and a fine-tune that goes badly leaves the newest file
worse than what came before. None of that should ever reach a person sitting down to play.

So there are two places, and only one of them is allowed near a player:

    checkpoints/    scratch. A run owns it, rewrites it, and may ruin it. Not in git.
    models/         the champion. Changes only by promotion, and only upward. In git.

The interfaces read ``models/champion.pt`` and nothing else, so a run in progress cannot
disturb a game in progress — the champion is a file that is replaced atomically, never
written into.

**Promotion is earned, not assumed.** ``python -m training.champion promote <candidate>``
plays the candidate against the reigning champion and refuses unless it is *measurably*
better: the Wilson lower bound on its win rate must clear 50%, which at 400 games means
winning about 55% before it counts. A candidate is also checked against the heuristic, so a
policy that beats the champion by exploiting one weakness while collapsing against
everything else does not get in — self-play is non-transitive and this is where that shows
up.

The champion is committed. It is under a megabyte, and a checkout that cannot play the
strongest bot is a checkout where the interesting half of the project is missing.
"""

import argparse
import datetime
import json
import pathlib
import shutil

import torch

from catan import action_space, encoder
from catan.agents import HeuristicAgent

#: Where the playable model lives. Deliberately not under ``checkpoints/``.
MODELS = pathlib.Path("models")
CHAMPION = MODELS / "champion.pt"
RECORD = MODELS / "champion.json"

#: Games in a promotion match. 400 gives a Wilson interval of roughly +-5 points, so a
#: candidate has to win about 55% for the lower bound to clear 50%.
PROMOTION_GAMES = 400

#: How far the candidate may fall against the fixed baseline before promotion is refused,
#: even if it beat the champion. Guards against a policy that has learned the champion's
#: quirks rather than the game.
MAX_BASELINE_REGRESSION = 0.05


def load(path=CHAMPION, temperature=0.0, seed=None):
    """The champion as a playable agent, or ``None`` if there is not a usable one.

    Never raises. A missing file, a missing PyTorch, or a model built for a different
    observation or action space all mean the same thing to a caller: play the heuristic
    instead. A checkpoint from before an encoder change loads perfectly well and then fails
    on the first move, which is the failure this exists to prevent.
    """
    path = pathlib.Path(path)
    if not path.is_file():
        return None
    try:
        from training.agent import PolicyAgent

        agent = PolicyAgent.load(path, temperature=temperature, seed=seed)
    except Exception:
        return None
    if agent.net.obs_size != encoder.SIZE:
        return None
    if agent.net.num_actions != action_space.NUM_ACTIONS:
        return None
    return agent


def record():
    """What is known about the reigning champion, or ``{}``."""
    if not RECORD.is_file():
        return {}
    try:
        return json.loads(RECORD.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def describe():
    """One line about the champion, for a startup message or a CLI."""
    if load() is None:
        return "no champion (the game falls back to the heuristic)"
    info = record()
    beat = info.get("beat_heuristic")
    when = info.get("promoted_at", "unknown date")
    return (f"champion from {when}"
            + (f", {100 * beat:.1f}% against the heuristic" if beat else ""))


# --------------------------------------------------------------------------- #
# Promotion                                                                   #
# --------------------------------------------------------------------------- #

def compare(candidate, defender, games=PROMOTION_GAMES, seed=31_000, label="defender"):
    """Play ``candidate`` against ``defender``, seats swapped. Returns the evaluation."""
    from training.evaluate import evaluate, format_result

    result = evaluate(candidate, defender, games=games, seed=seed, max_turns=800)
    print("  " + format_result(label, result), flush=True)
    return result


def promote(candidate_path, games=PROMOTION_GAMES, seed=31_000, force=False, log=print):
    """Install ``candidate_path`` as the champion if it earns the place.

    Returns ``(promoted, reason)``.
    """
    candidate = load(candidate_path)
    if candidate is None:
        return False, f"{candidate_path} is not a usable model for this engine"

    MODELS.mkdir(parents=True, exist_ok=True)
    reigning = load()
    baseline = HeuristicAgent(0)

    log(f"candidate: {candidate_path}")
    log(f"{games} games against the heuristic:")
    against_baseline = compare(candidate, baseline, games, seed, "heuristic")

    if reigning is None:
        _install(candidate_path, {"beat_heuristic": against_baseline["win_rate"],
                                  "beat_champion": None, "games": games})
        return True, "no reigning champion; installed"

    log(f"{games} games against the reigning champion:")
    against_champion = compare(candidate, reigning, games, seed + 1, "champion")

    if force:
        _install(candidate_path, {"beat_heuristic": against_baseline["win_rate"],
                                  "beat_champion": against_champion["win_rate"],
                                  "games": games, "forced": True})
        return True, "forced"

    # Must be better than the champion, not merely different. The lower bound clearing 50%
    # is the whole test: a candidate that wins 52% of 400 games has not shown anything.
    low, _ = against_champion["ci"]
    if low <= 0.5:
        return False, (f"beat the champion {100 * against_champion['win_rate']:.1f}% "
                       f"but the interval [{100 * low:.1f}, "
                       f"{100 * against_champion['ci'][1]:.1f}] includes 50% — not shown better")

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

    _install(candidate_path, {"beat_heuristic": against_baseline["win_rate"],
                              "beat_champion": against_champion["win_rate"],
                              "games": games})
    return True, (f"promoted: {100 * against_champion['win_rate']:.1f}% against the champion "
                  f"(lower bound {100 * low:.1f}%)")


def _install(candidate_path, results):
    """Replace the champion atomically, so a game in progress never sees half a file."""
    MODELS.mkdir(parents=True, exist_ok=True)
    from training.agent import export

    staging = CHAMPION.with_suffix(".incoming")
    export(candidate_path, staging)
    staging.replace(CHAMPION)

    previous = record()
    entry = {
        "promoted_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
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
        description="The model the game plays against",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("show", help="what the champion is")
    show.set_defaults(func=lambda a: print(describe()) or
                      print(json.dumps(record(), indent=2) if record() else ""))

    run = sub.add_parser("promote", help="install a candidate if it is measurably better")
    run.add_argument("candidate")
    run.add_argument("--games", type=int, default=PROMOTION_GAMES)
    run.add_argument("--seed", type=int, default=31_000)
    run.add_argument("--force", action="store_true",
                     help="install without the match; for restoring a known-good model")
    run.set_defaults(func=_promote_command)

    arguments = parser.parse_args(argv)
    return arguments.func(arguments)


def _promote_command(arguments):
    torch.set_num_threads(4)
    promoted, reason = promote(arguments.candidate, games=arguments.games,
                               seed=arguments.seed, force=arguments.force)
    print(("PROMOTED — " if promoted else "kept the current champion — ") + reason)
    print(describe())
    return 0 if promoted else 1


if __name__ == "__main__":
    raise SystemExit(main())

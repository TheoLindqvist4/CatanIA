"""Measuring an AlphaZero candidate, and deciding whether it is better.

Reuses :mod:`training.evaluate` rather than restating it: the Wilson interval, the seat
swapping and the truncation count are the same problem for both techniques, and two
implementations of "is 54% of 400 games meaningful" would eventually disagree.

What is added here is the **ladder** a candidate has to climb, and it has one more rung than
the PPO gate:

``heuristic``
    The fixed yardstick. It has not changed, so a number against it is comparable with every
    number this project has ever recorded — which is exactly why ``CLAUDE.md`` says not to
    change it.

``the previous technique``
    ``models/champion.pt``, grafted onto the current observation. Without this rung the two
    champions never meet, and "which one should the interface offer" has no answer.

``the reigning AlphaZero champion``
    The usual self-play ladder rung.

The search is what is being evaluated, not just the network, so every evaluation plays
:class:`~training.alphazero.agent.MCTSAgent` with a stated simulation count. A win rate is
therefore a property of ``(weights, simulations)`` and both belong in the record.
"""

from catan.agents import HeuristicAgent
from catan.rulesets import RANKED_1V1
from training.evaluate import confidence_interval, evaluate, format_result

#: Games in a routine check between iterations. +-6.9 points at 50%: enough to notice a
#: collapse, not enough to promote on, which is what ``promotion_games`` is for.
EVALUATION_GAMES = 200


def evaluate_agent(agent, opponent=None, games=EVALUATION_GAMES, seed=10_000,
                   max_turns=800):
    """One matchup. Thin wrapper so every call site uses the same ruleset and turn cap."""
    return evaluate(agent, opponent, games=games, seed=seed, ruleset=RANKED_1V1,
                    max_turns=max_turns)


def ladder(agent, games=EVALUATION_GAMES, seed=10_000, include=("heuristic",), log=None):
    """Play ``agent`` against each named rung. Returns ``{name: result}``.

    Rungs are named rather than passed as objects so a caller does not have to know how to
    build the previous technique's champion, and so a rung that cannot be built — no file, an
    observation it does not fit — is *skipped and reported*, not a crash in the middle of a
    training run.
    """
    results = {}
    for offset, name in enumerate(include):
        opponent = _opponent(name)
        if opponent is None:
            results[name] = None
            continue
        results[name] = evaluate_agent(agent, opponent, games=games, seed=seed + offset)
        if log is not None:
            log("  " + format_result(name, results[name]))
    return results


def _opponent(name):
    """Build a named rung, or ``None`` when this checkout has no such opponent."""
    if name == "heuristic":
        return HeuristicAgent(0)
    if name == "ppo_champion":
        from training.alphazero import champion

        return champion.load_previous_technique()
    if name == "champion":
        from training.alphazero import champion

        return champion.load()
    raise ValueError(f"unknown evaluation rung {name!r}")


def better(result, threshold=0.5):
    """Whether a result shows the candidate is *better*, not merely ahead.

    The Wilson lower bound has to clear ``threshold``. A candidate that won 52% of 400 games
    has shown nothing — the interval covers 47% to 57% — and promoting on the point estimate
    is how a ladder climbs while the player gets worse.
    """
    low, _ = result["ci"]
    return low > threshold


__all__ = ["EVALUATION_GAMES", "better", "confidence_interval", "evaluate_agent",
           "format_result", "ladder"]

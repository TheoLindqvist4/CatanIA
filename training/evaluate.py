"""Measuring a policy against a fixed opponent.

Self-play win rate is not progress — it sits at 50% whether the agent is improving or
cycling. The only honest number is against something that does not move, so every evaluation
here is against :class:`~catan.agents.HeuristicAgent`, whose strength is already recorded
(96.7% against greedy, 98.3% against random).

**How many games.** A win rate estimated from *n* games has a 95% interval of roughly
``±1.96·√(p(1-p)/n)``. At p≈0.5 that is ±9.8 points for n=100 and ±4.4 for n=500. So a
100-game evaluation cannot distinguish 45% from 55%, and reading iteration-to-iteration
wiggle in one is reading noise. The default here is 200 (±6.9 points) for the periodic check,
and the CLI takes a larger number for the ones that get written down.
"""

import math

from catan.agents import GreedyAgent, HeuristicAgent, RandomAgent, play_match
from catan.rulesets import RANKED_1V1


def confidence_interval(wins, games, z=1.96):
    """Wilson interval — correct near 0 and 1, where the normal approximation is not.

    Returns ``(low, high)`` as fractions.
    """
    if games == 0:
        return (0.0, 1.0)
    p = wins / games
    denominator = 1 + z * z / games
    centre = (p + z * z / (2 * games)) / denominator
    spread = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def evaluate(agent, opponent=None, games=200, seed=10_000, ruleset=RANKED_1V1,
             max_turns=1_000):
    """Play ``agent`` against ``opponent``, seats swapped every other game.

    Returns a dict with the win rate, its interval, and the truncation count — which matters
    on its own: an agent that never loses because it never finishes has not learned to win.
    """
    opponent = HeuristicAgent(0) if opponent is None else opponent
    tally = play_match(
        {1: agent, 2: opponent}, games=games, seed=seed,
        ruleset=ruleset, max_turns=max_turns,
    )
    wins, losses, truncated = tally[1], tally[2], tally["truncated"]
    decided = wins + losses
    rate = wins / decided if decided else 0.0
    low, high = confidence_interval(wins, decided)
    return {
        "wins": wins, "losses": losses, "truncated": truncated,
        "games": games, "win_rate": rate, "ci": (low, high),
        "ci_width": high - low,
    }


def evaluate_all(agent, games=200, seed=10_000):
    """Against the whole ladder, for the record."""
    return {
        "heuristic": evaluate(agent, HeuristicAgent(0), games=games, seed=seed),
        "greedy": evaluate(agent, GreedyAgent(0), games=games, seed=seed + 1),
        "random": evaluate(agent, RandomAgent(0), games=games, seed=seed + 2),
    }


def format_result(name, result):
    low, high = result["ci"]
    return (f"{name:<10} {result['wins']:>4}-{result['losses']:<4} "
            f"{100 * result['win_rate']:5.1f}%  "
            f"[{100 * low:4.1f}, {100 * high:4.1f}]  trunc {result['truncated']}")

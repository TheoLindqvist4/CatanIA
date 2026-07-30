"""Baseline agents.

An agent is any callable ``(observation, info) -> action index``. That is deliberately the
whole interface: a network fits it, and so does :func:`random_agent`.

These exist to be beaten. Their job is to give a learned policy something to measure against,
and to make the environment exercisable end to end — an untested environment is where silent
training bugs live.

    from catan.agents import RandomAgent, GreedyAgent, play_match
    print(play_match({1: GreedyAgent(0), 2: RandomAgent(0)}, games=100))
"""

import random

from catan import action_space
from catan.actions import ActionType
from catan.env import CatanEnv

#: What the greedy agent thinks each action type is worth, highest first. Cities before
#: settlements because a city is 2 points for 5 cards on ground you already hold; roads last
#: because they only pay off through Longest Road or reaching a spot.
GREEDY_PRIORITY = (
    ActionType.BUILD_CITY,
    ActionType.BUILD_SETTLEMENT,
    ActionType.BUY_DEV_CARD,
    ActionType.PLAY_KNIGHT,
    ActionType.PLAY_MONOPOLY,
    ActionType.PLAY_YEAR_OF_PLENTY,
    ActionType.PLAY_ROAD_BUILDING,
    ActionType.BUILD_ROAD,
    ActionType.MOVE_ROBBER,
    ActionType.DISCARD,
    ActionType.TRADE_WITH_BANK,
    ActionType.END_TURN,
)


class RandomAgent:
    """Picks uniformly among the legal actions.

    The floor. Anything that cannot beat this is broken.
    """

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def __call__(self, observation, info):
        return self.rng.choice(info["legal"])

    def __repr__(self):
        return "RandomAgent()"


class GreedyAgent:
    """Takes the highest-priority action type available, breaking ties at random.

    Not clever — it has no idea *where* to build, only *what*. But it beats random
    comfortably, because random spends its resources on trades and ends turns it could
    have built on.
    """

    def __init__(self, seed=None, priority=GREEDY_PRIORITY):
        self.rng = random.Random(seed)
        self.priority = priority

    def __call__(self, observation, info):
        by_type = {}
        for index in info["legal"]:
            by_type.setdefault(action_space.decode(index).type, []).append(index)
        for kind in self.priority:
            if kind in by_type:
                return self.rng.choice(by_type[kind])
        return self.rng.choice(info["legal"])   # an action type we never listed

    def __repr__(self):
        return "GreedyAgent()"


def play_game(agents, seed=None, num_players=None, ruleset=None, max_turns=None,
              on_step=None):
    """Play one game with ``{player: agent}``. Returns the final ``info``.

    ``info["winner"]`` is ``None`` if the game was truncated.
    """
    num_players = num_players if num_players is not None else len(agents)
    env = CatanEnv(num_players=num_players, ruleset=ruleset,
                   **({} if max_turns is None else {"max_turns": max_turns}))
    observation, info = env.reset(seed=seed)

    while not info["done"]:
        action = agents[info["player"]](observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        if on_step is not None:
            on_step(env, info)

    return info


def play_match(agents, games=50, seed=0, **kwargs):
    """Play ``games`` and tally the outcome.

    Seats are **swapped every other game** so a result is not just a first-player
    advantage — in Catan that advantage is real and large.

    Returns:
        dict: wins per player number, plus ``"truncated"``.
    """
    players = sorted(agents)
    tally = {player: 0 for player in players}
    tally["truncated"] = 0

    for game in range(games):
        # rotate which agent sits in which seat
        shift = game % len(players)
        seated = {
            players[i]: agents[players[(i + shift) % len(players)]]
            for i in range(len(players))
        }
        info = play_game(seated, seed=seed + game, **kwargs)

        if info["winner"] is None:
            tally["truncated"] += 1
        else:
            # report the win against the *agent*, not the seat it happened to hold
            winning_seat_index = players.index(info["winner"])
            tally[players[(winning_seat_index + shift) % len(players)]] += 1

    return tally

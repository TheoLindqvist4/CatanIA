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

from catan import action_space, heuristics
from catan.actions import ActionType
from catan.dev_cards import LARGEST_ARMY_MINIMUM, LONGEST_ROAD_MINIMUM
from catan.env import CatanEnv
from catan.resources import CITY_COST, NUM_RESOURCES, ROAD_COST, SETTLEMENT_COST, Resource
from catan.state import Phase
from catan.topology import VERTEX_TILES

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
    # Below every card play, so the baseline still plays a development card before rolling
    # when it holds one — the behaviour it had when that was the only legal move, which
    # keeps every win rate already recorded against it comparable.
    ActionType.ROLL,
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


class HeuristicAgent:
    """Plays with positional judgement: it chooses *where*, not only *what*.

    That is the whole difference from :class:`GreedyAgent`, and it is most of the strength.
    Everything it knows comes from ``info["view"]``, a :class:`~catan.view.PublicView`, so
    it cannot read the opponent's cards even by mistake.

    Difficulty is **noise added to its evaluations** rather than rules taken away. A weaker
    setting misjudges which spot is best, which is how a weaker human plays; hobbling the
    rules instead produces an opponent that behaves in ways no player would.

    Args:
        seed: for tie-breaking and noise.
        noise: standard deviation added to each evaluation, as a fraction of its spread.
            0 plays its best. See :data:`DIFFICULTY`.
    """

    def __init__(self, seed=None, noise=0.0):
        self.rng = random.Random(seed)
        self.noise = noise

    def __repr__(self):
        return f"HeuristicAgent(noise={self.noise})"

    # -- helpers -------------------------------------------------------- #

    def _jitter(self, value, scale=1.0):
        if not self.noise:
            return value
        return value + self.rng.gauss(0.0, self.noise * scale)

    def _best(self, options, score):
        """The highest-scoring option, with noise applied and ties broken randomly."""
        best, best_score = [], float("-inf")
        for option in options:
            value = self._jitter(score(option))
            if value > best_score + 1e-12:
                best, best_score = [option], value
            elif value > best_score - 1e-12:
                best.append(option)
        return self.rng.choice(best) if best else None

    # -- the decision --------------------------------------------------- #

    def __call__(self, observation, info):
        view = info.get("view")
        if view is None:                       # driven without a view: fall back
            return GreedyAgent(0)(observation, info)

        by_type = {}
        for index in info["legal"]:
            action = action_space.decode(index)
            by_type.setdefault(action.type, []).append((action, index))

        for handler in (
            self._setup_settlement, self._setup_road, self._discard, self._robber,
            self._city, self._settlement, self._dev_card_play, self._road,
            self._buy_dev, self._trade,
        ):
            choice = handler(view, by_type)
            if choice is not None:
                return choice
        return by_type.get(ActionType.END_TURN, [(None, info["legal"][0])])[0][1]

    # -- setup: the two placements decide most 1v1 games ---------------- #

    def _setup_settlement(self, view, by_type):
        if view.phase is not Phase.SETUP_SETTLEMENT:
            return None
        options = by_type.get(ActionType.BUILD_SETTLEMENT, [])
        have = heuristics.income(view, view.me)
        return self._best(
            options,
            lambda item: heuristics.settlement_value(view, view.me, item[0].position, have),
        )[1]

    def _setup_road(self, view, by_type):
        """Point the starting road at the best spot it opens, not at a random neighbour."""
        if view.phase is not Phase.SETUP_ROAD:
            return None
        options = by_type.get(ActionType.BUILD_ROAD, [])
        have = heuristics.income(view, view.me)
        return self._best(
            options,
            lambda item: heuristics.road_value(view, view.me, item[0].position, have),
        )[1]

    # -- forced choices -------------------------------------------------- #

    def _discard(self, view, by_type):
        """Give up whatever is furthest from a plan: the biggest pile of the cheapest thing."""
        if view.phase is not Phase.DISCARD:
            return None
        hand = view.my_hand
        options = by_type.get(ActionType.DISCARD, [])
        return self._best(
            options,
            lambda item: hand[item[0].position]
            / heuristics.RESOURCE_WEIGHT[Resource(item[0].position)],
        )[1]

    def _robber(self, view, by_type):
        """Hurt the leader most, and rob whoever can spare it least."""
        if view.phase is not Phase.MOVE_ROBBER:
            return None
        options = by_type.get(ActionType.MOVE_ROBBER, [])

        def score(item):
            action = item[0]
            victim = action.extra
            damage = sum(
                heuristics.robber_damage(view, action.position, opponent)
                for opponent in view.opponents
            )
            # a card taken now is worth having, and from the player holding most
            steal = 0.35 * view.hand_size(victim) if victim else 0.0
            return damage + steal

        return self._best(options, score)[1]

    # -- building -------------------------------------------------------- #

    def _city(self, view, by_type):
        """Upgrade the best producer. Two points on ground already held is the efficient
        route to 15, so this outranks everything else that costs cards."""
        options = by_type.get(ActionType.BUILD_CITY, [])
        if not options:
            return None
        return self._best(
            options, lambda item: heuristics.city_value(view, view.me, item[0].position)
        )[1]

    def _settlement(self, view, by_type):
        options = by_type.get(ActionType.BUILD_SETTLEMENT, [])
        if not options:
            return None
        have = heuristics.income(view, view.me)
        return self._best(
            options,
            lambda item: heuristics.settlement_value(view, view.me, item[0].position, have),
        )[1]

    def _road(self, view, by_type):
        """Only build a road that leads somewhere, or that takes Longest Road.

        Without the threshold the agent lays track for its own sake and starves itself of
        the wood and brick a settlement needs.
        """
        options = by_type.get(ActionType.BUILD_ROAD, [])
        if not options:
            return None
        have = heuristics.income(view, view.me)
        chasing_award = self._road_award_is_close(view)

        best = self._best(
            options, lambda item: heuristics.road_value(view, view.me, item[0].position, have)
        )
        value = heuristics.road_value(view, view.me, best[0].position, have)
        if value < ROAD_THRESHOLD and not chasing_award:
            return None
        return best[1]

    def _road_award_is_close(self, view):
        mine = view.longest_road(view.me)
        if view.longest_road_holder == view.me:
            return False
        best_other = max((view.longest_road(p) for p in view.opponents), default=0)
        return mine + 1 >= max(LONGEST_ROAD_MINIMUM, best_other + 1)

    # -- cards ----------------------------------------------------------- #

    def _buy_dev(self, view, by_type):
        """Buy when the cards would otherwise sit idle — not while saving for a city."""
        if ActionType.BUY_DEV_CARD not in by_type:
            return None
        hand = view.my_hand
        saving_for_city = (hand[Resource.ORE] >= 2 and hand[Resource.WHEAT] >= 1
                           and view.cities_left[view.me] > 0)
        if saving_for_city:
            return None
        return by_type[ActionType.BUY_DEV_CARD][0][1]

    def _dev_card_play(self, view, by_type):
        if ActionType.PLAY_MONOPOLY in by_type:
            # The bank shows what everyone else is holding: every card missing from it is
            # in a hand. Take whatever is most held and most useful.
            options = by_type[ActionType.PLAY_MONOPOLY]
            return self._best(
                options,
                lambda item: heuristics.held_by_others(view, item[0].position)
                * heuristics.RESOURCE_WEIGHT[Resource(item[0].position)],
            )[1]

        if ActionType.PLAY_YEAR_OF_PLENTY in by_type:
            options = by_type[ActionType.PLAY_YEAR_OF_PLENTY]
            hand = view.my_hand
            return self._best(options, lambda item: _completes(hand, item[0]))[1]

        if ActionType.PLAY_KNIGHT in by_type:
            # Worth playing if the robber is hurting us, or Largest Army is in reach.
            hurting = any(
                view.robber_tile in VERTEX_TILES[v] for v in view.buildings_of(view.me)
            )
            close = view.knights_played[view.me] + 1 >= LARGEST_ARMY_MINIMUM
            if hurting or close:
                return by_type[ActionType.PLAY_KNIGHT][0][1]

        if ActionType.PLAY_ROAD_BUILDING in by_type and self._road_award_is_close(view):
            return by_type[ActionType.PLAY_ROAD_BUILDING][0][1]
        return None

    # -- trading --------------------------------------------------------- #

    def _trade(self, view, by_type):
        """Trade only to close a gap on a build we could otherwise make.

        Trading for its own sake is how the random agent burns its hand.
        """
        options = by_type.get(ActionType.TRADE_WITH_BANK, [])
        if not options:
            return None
        hand = view.my_hand
        wanted = _shortfall(hand, view)
        if not wanted:
            return None
        useful = [item for item in options if item[0].extra in wanted]
        if not useful:
            return None
        return self._best(useful, lambda item: wanted[item[0].extra])[1]


#: A road has to open something worth having before it is worth the wood and brick.
ROAD_THRESHOLD = 0.30

#: Named difficulty settings: the noise added to every evaluation.
DIFFICULTY = {"easy": 0.9, "medium": 0.35, "hard": 0.0}


def _completes(hand, action):
    """How much a pair of free resources moves us toward a build."""
    wish = [0] * NUM_RESOURCES
    wish[action.position] += 1
    wish[action.extra] += 1
    best = 0.0
    for cost in (CITY_COST, SETTLEMENT_COST, ROAD_COST):
        before = sum(max(0, cost[r] - hand[r]) for r in range(NUM_RESOURCES))
        after = sum(max(0, cost[r] - hand[r] - wish[r]) for r in range(NUM_RESOURCES))
        best = max(best, before - after)
    return best


def _shortfall(hand, view):
    """``{resource: urgency}`` for what a build is missing, most valuable build first."""
    targets = []
    if view.cities_left[view.me] > 0:
        targets.append((CITY_COST, 3.0))
    if view.settlements_left[view.me] > 0:
        targets.append((SETTLEMENT_COST, 2.0))
    targets.append((ROAD_COST, 1.0))

    for cost, weight in targets:
        missing = {r: cost[r] - hand[r] for r in range(NUM_RESOURCES) if hand[r] < cost[r]}
        if missing and sum(missing.values()) <= 2:
            return {resource: weight / amount for resource, amount in missing.items()}
    return {}


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

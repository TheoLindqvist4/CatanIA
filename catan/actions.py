"""What a player can do, as data.

An :class:`Action` is a ``(type, position, extra)`` triple — small, hashable, comparable,
and cheap to put in a list. The two integer slots mean whatever the type says they mean:

===========================  ==========================  ====================
type                         position                    extra
===========================  ==========================  ====================
``END_TURN``                 unused                      unused
``BUILD_ROAD``               road id                     unused
``BUILD_SETTLEMENT``         vertex id                   unused
``BUILD_CITY``               vertex id                   unused
``TRADE_WITH_BANK``          resource given              resource received
``MOVE_ROBBER``              tile id                     player to rob, 0 for nobody
``DISCARD``                  resource discarded          unused
``BUY_DEV_CARD``             unused                      unused
``PLAY_KNIGHT``              unused                      unused
``PLAY_ROAD_BUILDING``       unused                      unused
``PLAY_YEAR_OF_PLENTY``      first resource taken        second resource taken
``PLAY_MONOPOLY``            resource demanded           unused
===========================  ==========================  ====================

``DISCARD`` gives up **one** card at a time. Choosing a whole multiset to discard does not
flatten into a discrete action space; one card per action does, at the cost of several
steps. The phase simply stays until the player is down to their limit.

Two plain ints rather than a variable payload, because Phase 3 has to flatten every
action into one discrete index and a fixed arity makes that a lookup rather than a parse.

Setup placements reuse ``BUILD_SETTLEMENT`` / ``BUILD_ROAD`` rather than getting their
own action types. Whether a placement is free and whether it must touch the settlement
just placed follows from ``state.phase``, so there is one action per (thing, position)
across the whole game. That keeps the flat action space as small as possible.

Phase 2 still has to add the robber, discard and dev-card types.
"""

from enum import IntEnum
from typing import NamedTuple

from catan.resources import Resource


class ActionType(IntEnum):
    END_TURN = 0
    BUILD_ROAD = 1
    BUILD_SETTLEMENT = 2
    BUILD_CITY = 3
    TRADE_WITH_BANK = 4
    MOVE_ROBBER = 5
    DISCARD = 6
    BUY_DEV_CARD = 7
    PLAY_KNIGHT = 8
    PLAY_ROAD_BUILDING = 9
    PLAY_YEAR_OF_PLENTY = 10
    PLAY_MONOPOLY = 11


#: Playing one of these consumes the turn's single development card.
DEV_CARD_PLAYS = frozenset({
    ActionType.PLAY_KNIGHT,
    ActionType.PLAY_ROAD_BUILDING,
    ActionType.PLAY_YEAR_OF_PLENTY,
    ActionType.PLAY_MONOPOLY,
})


#: Action types that carry a road id in ``position``.
ROAD_ACTIONS = frozenset({ActionType.BUILD_ROAD})

#: Action types that carry a vertex id in ``position``.
VERTEX_ACTIONS = frozenset({ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY})

#: Action types that build something and so can win the game.
BUILD_ACTIONS = frozenset(VERTEX_ACTIONS | ROAD_ACTIONS)


class Action(NamedTuple):
    type: ActionType
    position: int = 0
    extra: int = 0

    def __repr__(self):
        """Never raises, whatever the fields hold.

        This repr appears inside the ``IllegalAction`` message raised *because* an
        action is malformed. An exception here would mask the real error — which it
        already did once, for a bad type, and again for a bad resource index.
        """
        kind = _enum_name(ActionType, self.type)
        if self.type == ActionType.END_TURN:
            return kind
        if self.type == ActionType.TRADE_WITH_BANK:
            return (f"TRADE({_enum_name(Resource, self.position)}"
                    f"->{_enum_name(Resource, self.extra)})")
        if self.type == ActionType.DISCARD:
            return f"DISCARD({_enum_name(Resource, self.position)})"
        if self.type == ActionType.MOVE_ROBBER:
            victim = f", rob {self.extra}" if self.extra else ""
            return f"MOVE_ROBBER(tile {self.position}{victim})"
        if self.type in (ActionType.BUY_DEV_CARD, ActionType.PLAY_KNIGHT,
                         ActionType.PLAY_ROAD_BUILDING):
            return kind
        if self.type == ActionType.PLAY_YEAR_OF_PLENTY:
            return (f"PLAY_YEAR_OF_PLENTY({_enum_name(Resource, self.position)}"
                    f", {_enum_name(Resource, self.extra)})")
        if self.type == ActionType.PLAY_MONOPOLY:
            return f"PLAY_MONOPOLY({_enum_name(Resource, self.position)})"
        return f"{kind}({self.position})"


def _enum_name(enum_type, value):
    """``value``'s name in ``enum_type``, or a readable fallback if it has none."""
    try:
        return enum_type(value).name.lower() if enum_type is Resource \
            else enum_type(value).name
    except ValueError:
        return f"<invalid {enum_type.__name__} {value!r}>"


def end_turn():
    return Action(ActionType.END_TURN)


def build_road(road):
    return Action(ActionType.BUILD_ROAD, road)


def build_settlement(vertex):
    return Action(ActionType.BUILD_SETTLEMENT, vertex)


def build_city(vertex):
    return Action(ActionType.BUILD_CITY, vertex)


def trade_with_bank(give, take):
    """Give the bank ``give`` at the player's best rate, receive one ``take``."""
    return Action(ActionType.TRADE_WITH_BANK, int(give), int(take))


def move_robber(tile, victim=0):
    """Put the robber on ``tile`` and steal one card from ``victim`` (0 for nobody)."""
    return Action(ActionType.MOVE_ROBBER, tile, victim)


def discard(resource):
    """Give up one card of ``resource``."""
    return Action(ActionType.DISCARD, int(resource))


def buy_dev_card():
    """Draw the top development card, for one sheep, one wheat and one ore."""
    return Action(ActionType.BUY_DEV_CARD)


def play_knight():
    """Play a Knight: move the robber, and count toward Largest Army."""
    return Action(ActionType.PLAY_KNIGHT)


def play_road_building():
    """Play Road Building: two roads at no cost."""
    return Action(ActionType.PLAY_ROAD_BUILDING)


def play_year_of_plenty(first, second):
    """Play Year of Plenty: take two resources from the bank. May be the same twice.

    The pair is sorted, because taking ore and wheat is the same as taking wheat and
    ore. One action per distinct outcome keeps Phase 3's flat action space from carrying
    two indices for the same move, and keeps ``legal_actions`` and ``apply`` in agreement.
    """
    low, high = sorted((int(first), int(second)))
    return Action(ActionType.PLAY_YEAR_OF_PLENTY, low, high)


def play_monopoly(resource):
    """Play Monopoly: every opponent hands over all their ``resource``."""
    return Action(ActionType.PLAY_MONOPOLY, int(resource))

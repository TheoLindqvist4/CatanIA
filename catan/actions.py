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
===========================  ==========================  ====================

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

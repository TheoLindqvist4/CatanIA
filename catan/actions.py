"""What a player can do, as data.

An :class:`Action` is a ``(type, position)`` pair — small, hashable, comparable, and
cheap to put in a list. ``position`` is a road id for ``BUILD_ROAD`` and a vertex id for
the settlement and city actions; it is 0 where the action does not need one.

Setup placements reuse ``BUILD_SETTLEMENT`` / ``BUILD_ROAD`` rather than getting their
own action types. Whether a placement is free and whether it must touch the settlement
just placed follows from ``state.phase``, so there is one action per (thing, position)
across the whole game. That keeps Phase 3's flat action space as small as possible.

Phase 3 adds the flat integer codec and the boolean legality mask on top of this. Phase 2
adds the dev-card, robber, discard and trade action types.
"""

from enum import IntEnum
from typing import NamedTuple


class ActionType(IntEnum):
    END_TURN = 0
    BUILD_ROAD = 1
    BUILD_SETTLEMENT = 2
    BUILD_CITY = 3


#: Action types that carry a road id in ``position``.
ROAD_ACTIONS = frozenset({ActionType.BUILD_ROAD})

#: Action types that carry a vertex id in ``position``.
VERTEX_ACTIONS = frozenset({ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY})


class Action(NamedTuple):
    type: ActionType
    position: int = 0

    def __repr__(self):
        # Must tolerate a bogus type: this repr appears in the IllegalAction message
        # raised *because* the type was bogus, and crashing there hides the real error.
        try:
            name = ActionType(self.type).name
        except ValueError:
            return f"Action(type={self.type!r}, position={self.position!r})"
        return name if self.type == ActionType.END_TURN else f"{name}({self.position})"


def end_turn():
    return Action(ActionType.END_TURN)


def build_road(road):
    return Action(ActionType.BUILD_ROAD, road)


def build_settlement(vertex):
    return Action(ActionType.BUILD_SETTLEMENT, vertex)


def build_city(vertex):
    return Action(ActionType.BUILD_CITY, vertex)

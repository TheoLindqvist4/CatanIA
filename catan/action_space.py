"""The flat discrete action space, and the legality mask.

A network needs a **fixed-size** output: one index per action the game can express, the same
size in every state and for every player count. This module is that mapping.

    ACTIONS[i]        the Action at index i
    encode(action)    Action -> index
    decode(index)     index -> Action
    legal_mask(state) bool per index: may this be played right now?

324 actions, laid out in contiguous blocks by type so a slice of the mask is a slice of one
action kind — which makes both debugging and per-type policy heads straightforward:

===========================  =======  =====================================================
type                         count    layout
===========================  =======  =====================================================
``END_TURN``                       1
``BUILD_ROAD``                    72  road 1..72
``BUILD_SETTLEMENT``              54  vertex 1..54
``BUILD_CITY``                    54  vertex 1..54
``TRADE_WITH_BANK``               20  every ordered pair of distinct resources
``MOVE_ROBBER``                   95  19 tiles x (nobody, or one of 4 players)
``DISCARD``                        5  one per resource
``BUY_DEV_CARD``                   1
``PLAY_KNIGHT``                    1
``PLAY_ROAD_BUILDING``             1
``PLAY_YEAR_OF_PLENTY``           15  every *sorted* pair, doubles included
``PLAY_MONOPOLY``                  5  one per resource
===========================  =======  =====================================================

The size is deliberately independent of ``num_players``: robber actions naming a player who
is not in the game simply never come back legal. A network trained on 1v1 therefore has the
same output shape as one trained on four players.

**The mask is derived from :func:`catan.rules.legal_actions`, never re-derived.** Duplicating
legality here would recreate exactly the bug the engine was rebuilt to remove — two authorities
that can disagree. :func:`legal_mask` translates; it does not decide.
"""

from catan import rules
from catan.actions import (
    roll,
    Action,
    ActionType,
    build_city,
    build_road,
    build_settlement,
    buy_dev_card,
    discard,
    end_turn,
    move_robber,
    play_knight,
    play_monopoly,
    play_road_building,
    play_year_of_plenty,
    trade_with_bank,
)
from catan.resources import NUM_RESOURCES
from catan.state import MAX_PLAYERS
from catan.topology import NUM_ROADS, NUM_TILES, NUM_VERTICES


def _build():
    """Every expressible action, grouped by type. Order defines the indices."""
    blocks = [
        (ActionType.END_TURN, [end_turn()]),
        (ActionType.BUILD_ROAD,
         [build_road(road) for road in range(1, NUM_ROADS + 1)]),
        (ActionType.BUILD_SETTLEMENT,
         [build_settlement(v) for v in range(1, NUM_VERTICES + 1)]),
        (ActionType.BUILD_CITY,
         [build_city(v) for v in range(1, NUM_VERTICES + 1)]),
        (ActionType.TRADE_WITH_BANK,
         [trade_with_bank(give, take)
          for give in range(NUM_RESOURCES)
          for take in range(NUM_RESOURCES)
          if give != take]),
        (ActionType.MOVE_ROBBER,
         [move_robber(tile, victim)
          for tile in range(1, NUM_TILES + 1)
          for victim in range(0, MAX_PLAYERS + 1)]),
        (ActionType.DISCARD,
         [discard(resource) for resource in range(NUM_RESOURCES)]),
        (ActionType.BUY_DEV_CARD, [buy_dev_card()]),
        (ActionType.PLAY_KNIGHT, [play_knight()]),
        (ActionType.PLAY_ROAD_BUILDING, [play_road_building()]),
        (ActionType.PLAY_YEAR_OF_PLENTY,
         # sorted pairs only: taking ore-then-wheat is the same move as the reverse, so
         # it must not occupy two indices
         [play_year_of_plenty(first, second)
          for first in range(NUM_RESOURCES)
          for second in range(first, NUM_RESOURCES)]),
        (ActionType.PLAY_MONOPOLY,
         [play_monopoly(resource) for resource in range(NUM_RESOURCES)]),
        # Appended deliberately: every index above keeps the value it had before ROLL
        # existed, so anything that recorded one still means the same move.
        (ActionType.ROLL, [roll()]),
    ]

    actions, spans, start = [], {}, 0
    for kind, block in blocks:
        spans[kind] = (start, start + len(block))
        actions.extend(block)
        start += len(block)
    return tuple(actions), spans


ACTIONS, _SPANS = _build()

NUM_ACTIONS = len(ACTIONS)

#: action -> its index
INDEX = {action: i for i, action in enumerate(ACTIONS)}

#: action type -> the contiguous ``slice`` of indices it occupies
SLICES = {kind: slice(lo, hi) for kind, (lo, hi) in _SPANS.items()}

#: action type -> how many indices it occupies
COUNTS = {kind: hi - lo for kind, (lo, hi) in _SPANS.items()}


def encode(action):
    """``Action`` -> index.

    Raises:
        KeyError: if the action is not expressible. That means a new action type was added
            to :mod:`catan.actions` without being added here, and the mask would silently
            have dropped it.
    """
    try:
        return INDEX[action]
    except KeyError:
        raise KeyError(
            f"{action!r} is not in the action space — was a new ActionType added "
            f"without extending catan.action_space?"
        ) from None


def decode(index):
    """Index -> ``Action``."""
    if not 0 <= index < NUM_ACTIONS:
        raise IndexError(f"action index must be in 0..{NUM_ACTIONS - 1}, got {index}")
    return ACTIONS[index]


def legal_mask(state):
    """A ``bytearray`` of :data:`NUM_ACTIONS` flags: 1 where the action is legal now.

    A ``bytearray`` rather than a list of bools because it is compact, cheap to build, and
    converts to a numpy array or a torch tensor without a per-element Python loop.

    All zero during :attr:`~catan.state.Phase.ROLL` when the player holds no playable
    development card, and after the game ends — in the first case the driver should call
    :func:`catan.rules.roll_dice`, which is not an action.
    """
    mask = bytearray(NUM_ACTIONS)
    for action in rules.legal_actions(state):
        mask[encode(action)] = 1
    return mask


def legal_indices(state):
    """The legal action indices, ascending."""
    return sorted(encode(action) for action in rules.legal_actions(state))


def grouped(state):
    """Legal actions arranged for a person to choose from.

    ``{ActionType: {(position, extra): index}}`` — pick a *kind* of action, then a target.
    An interface offers the kinds as buttons and highlights the targets on the board, which
    is what replaces asking someone to pick from a flat list of 54.

    This lives here rather than in the interface because it is the same information
    :func:`legal_mask` carries, only shaped differently — and because in Python it can be
    tested, which browser code cannot.
    """
    out = {}
    for action in rules.legal_actions(state):
        out.setdefault(action.type, {})[(action.position, action.extra)] = encode(action)
    return out


def clickable(state):
    """Which board elements are legal targets right now, per action type.

    ``{ActionType: {element_id: index}}`` for the types that name a board element — roads,
    settlements, cities and the robber's destination. Types whose ``position`` is a resource
    rather than a place (trades, discards, Monopoly) are left out: they are chosen from a
    panel, not by clicking the board.

    ``MOVE_ROBBER`` maps a tile to *one* index, the first by victim. A tile offering several
    victims needs a follow-up choice, which :func:`victims_for_tile` supplies.
    """
    board_types = (
        ActionType.BUILD_ROAD,
        ActionType.BUILD_SETTLEMENT,
        ActionType.BUILD_CITY,
        ActionType.MOVE_ROBBER,
    )
    out = {kind: {} for kind in board_types}
    for action in rules.legal_actions(state):
        if action.type in out:
            out[action.type].setdefault(action.position, encode(action))
    return {kind: targets for kind, targets in out.items() if targets}


def victims_for_tile(state, tile):
    """``{player: index}`` for robbing each possible victim on ``tile``.

    Empty if the tile is not a legal destination. A single entry keyed ``0`` means the tile
    is legal but there is nobody to rob, so no follow-up question is needed.
    """
    out = {}
    for action in rules.legal_actions(state):
        if action.type is ActionType.MOVE_ROBBER and action.position == tile:
            out[action.extra] = encode(action)
    return out


def describe(index):
    """``'073 BUILD_SETTLEMENT(1)'`` — for logs and test failures."""
    return f"{index:03d} {decode(index)!r}"


def _validate():
    """Structural check on the mapping, at import. Cheap.

    ``python -O`` strips these; tests/test_action_space.py is the real guarantee.
    """
    assert NUM_ACTIONS == sum(COUNTS.values())
    assert len(INDEX) == NUM_ACTIONS, "duplicate actions in the space"
    assert set(SLICES) == set(ActionType), "an ActionType has no block"

    # blocks are contiguous, in order, and cover everything exactly once
    covered = 0
    for kind in ActionType:
        span = SLICES[kind]
        assert span.start == covered, f"{kind.name} does not start where the last ended"
        covered = span.stop
        assert all(ACTIONS[i].type == kind for i in range(span.start, span.stop))
    assert covered == NUM_ACTIONS

    # round trip
    assert all(decode(encode(action)) == action for action in ACTIONS)


_validate()

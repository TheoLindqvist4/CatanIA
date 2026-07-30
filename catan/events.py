"""What just happened, so an interface can say so.

The engine used to compute all of this and throw it away: :func:`catan.rules.distribute`
worked out who received what and no caller read it, and ``_steal_one_card`` moved a card in
silence. A player watching an opponent's turn could see the board change but never why.

**The rules only ever append; clearing is the caller's job.** That is deliberate: one
``step`` of :class:`~catan.env.CatanEnv` can apply an action *and* roll the dice for the next
player, and if either call cleared the list the other's events would vanish. An earlier
version cleared in both places and silently lost every action that happened to precede an
automatic roll.

:meth:`catan.env.CatanEnv.step` clears at the start and hands back everything that happened
during the step in ``info["events"]``, which is the contract an interface wants. A caller
using :mod:`catan.rules` directly and ignoring events gets a list that grows for the length
of one game — a few hundred kilobytes — and is discarded with it.

An :class:`Event` is a fixed-arity NamedTuple of small ints, like
:class:`catan.actions.Action`, so recording one costs an append and nothing else. Fields
mean different things per kind; :func:`describe` is the one place that knows which.
"""

from enum import IntEnum
from typing import NamedTuple


class EventKind(IntEnum):
    ROLLED = 0           # amount = the roll
    PRODUCED = 1         # player gained amount of resource
    STOLE = 2            # player took resource from other
    ROBBER_MOVED = 3     # player moved the robber to position (a tile)
    DISCARDED = 4        # player gave up one resource
    BUILT = 5            # player built other (a Piece, or 0 for a road) at position
    TRADED = 6           # player gave resource to the bank for other
    BOUGHT_DEV = 7       # player drew a development card
    PLAYED_DEV = 8       # player played position (a DevCard)
    MONOPOLISED = 9      # player took amount of resource from everyone
    AWARD = 10           # player gained (amount=1) or lost (amount=0) award `position`
    TURN_ENDED = 11      # player ended their turn
    GAME_OVER = 12       # player won


class Award(IntEnum):
    LARGEST_ARMY = 0
    LONGEST_ROAD = 1


class Event(NamedTuple):
    kind: EventKind
    player: int = 0
    resource: int = -1
    amount: int = 0
    position: int = 0
    other: int = 0


def describe(event, names=None):
    """One line of plain English.

    Shared by every interface so they cannot disagree about what happened.

    Args:
        event: the event.
        names: optional ``{player: label}``, so a UI can say "You" instead of "P1".
    """
    from catan.dev_cards import DevCard
    from catan.resources import Resource
    from catan.state import Piece

    def who(player):
        if names and player in names:
            return names[player]
        return f"P{player}"

    def res(value):
        return Resource(value).name.lower() if 0 <= value < len(Resource) else "?"

    kind = event.kind
    if kind is EventKind.ROLLED:
        return f"{who(event.player)} rolled {event.amount}"
    if kind is EventKind.PRODUCED:
        return f"{who(event.player)} got {event.amount} {res(event.resource)}"
    if kind is EventKind.STOLE:
        return f"{who(event.player)} stole {res(event.resource)} from {who(event.other)}"
    if kind is EventKind.ROBBER_MOVED:
        return f"{who(event.player)} moved the robber to tile {event.position}"
    if kind is EventKind.DISCARDED:
        return f"{who(event.player)} discarded {res(event.resource)}"
    if kind is EventKind.BUILT:
        thing = {int(Piece.SETTLEMENT): "a settlement",
                 int(Piece.CITY): "a city"}.get(event.other, "a road")
        where = "road" if event.other == 0 else "vertex"
        return f"{who(event.player)} built {thing} at {where} {event.position}"
    if kind is EventKind.TRADED:
        return (f"{who(event.player)} traded {event.amount} {res(event.resource)}"
                f" for 1 {res(event.other)}")
    if kind is EventKind.BOUGHT_DEV:
        return f"{who(event.player)} bought a development card"
    if kind is EventKind.PLAYED_DEV:
        card = DevCard(event.position).name.replace("_", " ").lower()
        return f"{who(event.player)} played {card}"
    if kind is EventKind.MONOPOLISED:
        return f"{who(event.player)} monopolised {event.amount} {res(event.resource)}"
    if kind is EventKind.AWARD:
        award = Award(event.position).name.replace("_", " ").lower()
        verb = "took" if event.amount else "lost"
        return f"{who(event.player)} {verb} {award}"
    if kind is EventKind.TURN_ENDED:
        return f"{who(event.player)} ended the turn"
    if kind is EventKind.GAME_OVER:
        return f"{who(event.player)} wins"
    return kind.name.lower()

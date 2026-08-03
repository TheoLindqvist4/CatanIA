"""Development cards.

Twenty-five cards in one deck, shuffled at the start of the game:

    14  Knight          move the robber and steal, and count toward Largest Army
     5  Victory Point   never played — worth a point while held
     2  Road Building   two free roads
     2  Year of Plenty  two resources of your choice from the bank
     2  Monopoly        every opponent hands over all of one resource

A player's holding is a ``list[int]`` of length 5 indexed by :class:`DevCard`, matching
how hands and costs are laid out in :mod:`catan.resources` — the encoder wants one shape.

Three timing rules, the first two of which need their own bookkeeping:

* **one card per turn** — ``state.dev_card_played_this_turn``
* **not the turn you bought it** — ``state.dev_cards_new`` records this turn's purchases,
  and they become playable when the turn ends.
* **only a Knight before the dice** — ``catan.actions.PRE_ROLL_PLAYS``. The Knight decides
  which tile pays out this turn; the other three do the same thing either side of the roll.

Victory Point cards are the exception to everything: they are never *played*, they simply
count while held, and they stay hidden until they win the game.
"""

from enum import IntEnum


class DevCard(IntEnum):
    KNIGHT = 0
    VICTORY_POINT = 1
    ROAD_BUILDING = 2
    YEAR_OF_PLENTY = 3
    MONOPOLY = 4


NUM_DEV_CARDS = len(DevCard)

#: The standard deck: 25 cards.
DECK_COUNTS = {
    DevCard.KNIGHT: 14,
    DevCard.VICTORY_POINT: 5,
    DevCard.ROAD_BUILDING: 2,
    DevCard.YEAR_OF_PLENTY: 2,
    DevCard.MONOPOLY: 2,
}

DECK_SIZE = sum(DECK_COUNTS.values())

#: Cards that are played as an action. A Victory Point card is not one of them.
PLAYABLE = tuple(card for card in DevCard if card is not DevCard.VICTORY_POINT)

#: Free roads granted by Road Building.
ROAD_BUILDING_ROADS = 2

#: Resources granted by Year of Plenty.
YEAR_OF_PLENTY_RESOURCES = 2

#: Knights needed before Largest Army can be awarded.
LARGEST_ARMY_MINIMUM = 3

#: Road segments needed before Longest Road can be awarded.
LONGEST_ROAD_MINIMUM = 5

#: Victory points each special card is worth.
AWARD_VICTORY_POINTS = 2


def empty_holding():
    return [0] * NUM_DEV_CARDS


def build_deck(rng):
    """A shuffled 25-card deck, drawn from the end with ``pop()``.

    The order is fixed once shuffled rather than drawn at random per purchase, so a
    clone replays the same sequence — the deck is hidden information, not a fresh die
    roll. Phase 3's encoder must mask it: a search that can read ``state.dev_deck``
    would be cheating.
    """
    deck = [card for card, count in DECK_COUNTS.items() for _ in range(count)]
    rng.shuffle(deck)
    return deck

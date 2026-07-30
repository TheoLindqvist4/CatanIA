"""The five resources, and what things cost.

A player's hand is a plain ``list[int]`` of length 5 indexed by :class:`Resource`. Not a
dict: the encoder (Phase 3) wants a fixed-width vector, ``clone()`` wants a cheap copy,
and affordability checks want integer indexing. ``hand[Resource.WHEAT]`` still reads
fine because :class:`Resource` is an ``IntEnum``.

Costs are 5-tuples in the same layout, built from keyword arguments so the definitions
stay readable.

The desert is **not** a resource. A desert tile carries ``None`` in
``Board.tile_resources``, which is why iterating production can skip it with
``is None`` rather than comparing against a magic string.
"""

from enum import IntEnum


class Resource(IntEnum):
    WOOD = 0
    BRICK = 1
    SHEEP = 2
    WHEAT = 3
    ORE = 4


NUM_RESOURCES = len(Resource)

#: A tile with no resource.
DESERT = None


def _cost(**amounts):
    """``_cost(wood=1, brick=1)`` -> ``(1, 1, 0, 0, 0)``."""
    cost = [0] * NUM_RESOURCES
    for name, amount in amounts.items():
        cost[Resource[name.upper()]] = amount
    return tuple(cost)


ROAD_COST = _cost(wood=1, brick=1)
SETTLEMENT_COST = _cost(wood=1, brick=1, sheep=1, wheat=1)
CITY_COST = _cost(wheat=2, ore=3)
DEV_CARD_COST = _cost(sheep=1, wheat=1, ore=1)

#: Cards of each resource in the bank at the start of the game.
BANK_PER_RESOURCE = 19

#: Cards you must give the bank for one card back, without a harbour.
BANK_RATE = 4
#: With a generic (3:1) harbour.
GENERIC_HARBOUR_RATE = 3
#: With the matching specific (2:1) harbour.
SPECIFIC_HARBOUR_RATE = 2


def empty_hand():
    return [0] * NUM_RESOURCES


def can_afford(hand, cost):
    """Whether ``hand`` covers every resource in ``cost``."""
    return all(held >= needed for held, needed in zip(hand, cost))


def pay(hand, cost):
    """Deduct ``cost`` from ``hand`` in place.

    Raises:
        ValueError: if the hand cannot cover it. Callers should have checked, so this
            firing means a legality check and a mutation disagreed.
    """
    if not can_afford(hand, cost):
        raise ValueError(f"cannot pay {describe(cost)} from {describe(hand)}")
    for resource, needed in enumerate(cost):
        hand[resource] -= needed


def total(hand):
    """Number of cards in a hand. The discard rule (Phase 2) triggers above 7."""
    return sum(hand)


def describe(amounts):
    """Human-readable ``'2 wheat, 3 ore'``. For messages and test failures only."""
    parts = [
        f"{amount} {Resource(resource).name.lower()}"
        for resource, amount in enumerate(amounts)
        if amount
    ]
    return ", ".join(parts) if parts else "nothing"

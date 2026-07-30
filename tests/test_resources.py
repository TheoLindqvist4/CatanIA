"""The resource vocabulary and the cost table."""

import pytest

from catan import resources
from catan.resources import (
    CITY_COST,
    DESERT,
    DEV_CARD_COST,
    NUM_RESOURCES,
    ROAD_COST,
    SETTLEMENT_COST,
    Resource,
)


def test_there_are_five_tradeable_resources_and_the_desert_is_not_one():
    assert NUM_RESOURCES == 5
    assert [r.name for r in Resource] == ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
    assert list(Resource) == [0, 1, 2, 3, 4]
    assert DESERT is None
    assert DESERT not in list(Resource)


def test_costs_match_the_rulebook():
    def as_dict(cost):
        return {Resource(i).name.lower(): n for i, n in enumerate(cost) if n}

    assert as_dict(ROAD_COST) == {"wood": 1, "brick": 1}
    assert as_dict(SETTLEMENT_COST) == {"wood": 1, "brick": 1, "sheep": 1, "wheat": 1}
    assert as_dict(CITY_COST) == {"wheat": 2, "ore": 3}
    assert as_dict(DEV_CARD_COST) == {"sheep": 1, "wheat": 1, "ore": 1}


def test_every_cost_is_a_fixed_width_vector():
    """The encoder (Phase 3) needs one layout for hands and costs alike."""
    for cost in (ROAD_COST, SETTLEMENT_COST, CITY_COST, DEV_CARD_COST):
        assert isinstance(cost, tuple)
        assert len(cost) == NUM_RESOURCES


def test_a_hand_is_indexable_by_resource():
    hand = resources.empty_hand()
    assert hand == [0, 0, 0, 0, 0]
    hand[Resource.WHEAT] = 3
    assert hand[3] == 3


def test_can_afford_needs_every_resource_not_just_the_total():
    assert resources.can_afford([1, 1, 0, 0, 0], ROAD_COST)
    assert resources.can_afford([9, 9, 9, 9, 9], ROAD_COST)
    # five cards, but none of them brick
    assert not resources.can_afford([5, 0, 0, 0, 0], ROAD_COST)


def test_paying_deducts_exactly_the_cost():
    hand = [2, 2, 2, 5, 5]
    resources.pay(hand, CITY_COST)
    assert hand == [2, 2, 2, 3, 2]


def test_paying_what_you_cannot_afford_raises():
    """A legality check and a mutation disagreeing must be loud, not silent."""
    hand = [0, 0, 0, 1, 1]
    with pytest.raises(ValueError):
        resources.pay(hand, CITY_COST)
    assert hand == [0, 0, 0, 1, 1], "a failed payment must not partially deduct"


def test_total_counts_cards_for_the_discard_rule():
    assert resources.total([1, 2, 0, 3, 2]) == 8


def test_describe_is_readable():
    assert resources.describe(CITY_COST) == "2 wheat, 3 ore"
    assert resources.describe(resources.empty_hand()) == "nothing"

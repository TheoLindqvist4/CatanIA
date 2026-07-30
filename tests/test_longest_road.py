"""Longest road, under the rules settled in docs/decisions/0006.

Two rulings are in force:

1. **Strict simple path** — a route may not pass through the same intersection twice.
2. **An opponent's building breaks a road** — a chain may end at one, not continue past.

Both are pinned here, including the cases where the strict reading differs visibly from
the roads-only reading it replaced.
"""

import random

import pytest

import catan.topology as T
from catan import rules
from helpers import fresh, put_building, put_roads


def longest(roads, opponents=(), mine=(), player=1, seed=1):
    """Longest road for ``player`` given a road set and some buildings."""
    state = fresh(seed=seed)
    put_roads(state, player, roads)
    for vertex in opponents:
        put_building(state, 2 if player != 2 else 1, vertex)
    for vertex in mine:
        put_building(state, player, vertex)
    return rules.longest_road_length(state, player)


# --------------------------------------------------------------------------- #
# Basics                                                                      #
# --------------------------------------------------------------------------- #

def test_no_roads_is_zero():
    assert longest([]) == 0


def test_a_single_road_is_one():
    assert longest([30]) == 1


def test_disconnected_roads_do_not_add_up():
    assert longest([3, 16]) == 1


def test_a_straight_chain_counts_every_segment():
    # 1=(1,4) 2=(1,5) 3=(2,5) 4=(2,6) -> the route 4-1-5-2-6
    assert longest([1, 2]) == 2
    assert longest([1, 2, 3]) == 3
    assert longest([1, 2, 3, 4]) == 4


def test_a_branch_counts_only_one_arm():
    # 22=(15,20) 30=(20,25) 31=(20,26): a Y centred on vertex 20
    assert longest([22, 30, 31]) == 2


def test_chains_crossing_vertex_5_and_35_are_measured_correctly():
    """Regression: the hand-written road map omitted a neighbour at each of these."""
    assert longest([2, 3]) == 2
    assert longest([2, 3, 4]) == 3
    assert longest([43, 51]) == 2
    assert longest([42, 51]) == 2


def test_only_your_own_roads_count():
    state = fresh(seed=1)
    put_roads(state, 1, [1, 2])
    put_roads(state, 2, [3, 4])
    assert rules.longest_road_length(state, 1) == 2
    assert rules.longest_road_length(state, 2) == 2


# --------------------------------------------------------------------------- #
# Ruling 1: strict simple path                                                #
# --------------------------------------------------------------------------- #

def test_a_route_may_not_reuse_an_intersection():
    """The decided case. 6, not the 7 the roads-only version gave.

    Vertex 8 carries all three of the player's roads (7, 11, 12). Reaching 7 required
    passing through it twice.
    """
    assert longest([7, 11, 12, 19, 20, 25, 26, 27]) == 6


def test_a_closed_loop_of_six_roads_counts_as_five():
    """A visible consequence of ruling 1, flagged in decision 0006.

    The six roads around tile 1 form a cycle over six vertices. A simple path can visit
    all six vertices but only five of the roads — closing the loop would revisit the
    start. The roads-only reading would answer 6.

    If this turns out to be undesirable, decision 0006 is the place to change it, and
    this test is the thing to change with it.
    """
    ring = T.TILE_ROADS[1]
    assert len(ring) == 6
    assert longest(ring) == 5


def test_a_loop_with_a_tail_uses_the_tail_instead_of_closing():
    ring = T.TILE_ROADS[1]
    assert longest(list(ring) + [11]) == 6


def test_a_figure_of_eight_still_never_reuses_a_vertex():
    two_rings = sorted(set(T.TILE_ROADS[1]) | set(T.TILE_ROADS[2]))
    result = longest(two_rings)
    assert result <= len(two_rings)
    # 10 roads over 10 vertices sharing one junction: a simple path cannot use them all
    assert result < len(two_rings)


# --------------------------------------------------------------------------- #
# Ruling 2: an opponent's building breaks a road                               #
# --------------------------------------------------------------------------- #

def test_an_opponent_building_in_the_middle_splits_the_chain():
    # route 4-1-5-2-6; an opponent sits on vertex 5, the middle
    assert longest([1, 2, 3, 4]) == 4
    assert longest([1, 2, 3, 4], opponents=[5]) == 2


def test_an_opponent_building_at_an_endpoint_does_not_shorten_anything():
    """A chain may *end* at an opponent's building. Vertex 4 is a route endpoint."""
    assert longest([1, 2, 3, 4], opponents=[4]) == 4


def test_an_opponent_just_inside_an_endpoint_shortens_by_one():
    # vertex 1 is interior (between roads 1 and 2), so it splits 1 | 3
    assert longest([1, 2, 3, 4], opponents=[1]) == 3


def test_your_own_buildings_never_break_your_road():
    assert longest([1, 2, 3, 4], mine=[5]) == 4
    assert longest([1, 2, 3, 4], mine=[1, 5, 2]) == 4


def test_a_city_blocks_exactly_like_a_settlement():
    from catan.state import Piece
    state = fresh(seed=1)
    put_roads(state, 1, [1, 2, 3, 4])
    put_building(state, 2, 5, Piece.CITY)
    assert rules.longest_road_length(state, 1) == 2


def test_blocking_is_symmetric_between_players():
    state = fresh(seed=1)
    put_roads(state, 1, [1, 2, 3, 4])
    put_roads(state, 2, [7, 11, 12])
    put_building(state, 1, 8)  # player 1 blocks player 2's junction
    assert rules.longest_road_length(state, 2) == 1
    assert rules.longest_road_length(state, 1) == 4


# --------------------------------------------------------------------------- #
# Invariants                                                                  #
# --------------------------------------------------------------------------- #

def test_longest_road_never_exceeds_the_number_of_roads_owned():
    rng = random.Random(7)
    for _ in range(40):
        roads = rng.sample(range(1, T.NUM_ROADS + 1), rng.randint(1, 15))
        assert 1 <= longest(roads) <= len(roads)


def test_adding_a_road_never_shortens_the_longest_road():
    rng = random.Random(3)
    for _ in range(30):
        roads = rng.sample(range(1, T.NUM_ROADS + 1), 8)
        before = longest(roads)
        extra = next(r for r in range(1, T.NUM_ROADS + 1) if r not in roads)
        assert longest(roads + [extra]) >= before


def test_adding_an_opponent_building_never_lengthens_a_road():
    rng = random.Random(5)
    for _ in range(30):
        roads = rng.sample(range(1, T.NUM_ROADS + 1), 8)
        before = longest(roads)
        vertex = rng.randint(1, T.NUM_VERTICES)
        assert longest(roads, opponents=[vertex]) <= before


# --------------------------------------------------------------------------- #
# Reporting the holder                                                        #
# --------------------------------------------------------------------------- #

def test_the_holder_is_the_single_longest():
    state = fresh(seed=1)
    put_roads(state, 1, [1, 2, 3, 4])
    put_roads(state, 2, [7, 11])
    assert rules.longest_road_holder(state) == (1, 4)


def test_a_tie_has_no_holder():
    state = fresh(seed=1)
    put_roads(state, 1, [1, 2])
    put_roads(state, 2, [19, 24])
    assert rules.longest_road_holder(state) == (None, 2)


def test_no_roads_at_all_has_no_holder():
    assert rules.longest_road_holder(fresh(seed=1)) == (None, 0)


def test_the_award_itself_is_not_implemented_yet():
    """Phase 2 owns the 2 victory points, the 5-segment minimum and keep-until-beaten.
    Until then a long road must not silently affect the score."""
    state = fresh(seed=1)
    put_roads(state, 1, [1, 2, 3, 4, 7, 11])
    assert rules.longest_road_length(state, 1) >= 5
    assert rules.victory_points(state, 1) == 0

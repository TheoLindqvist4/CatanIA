"""The bank, harbours, and trading with the bank.

Trading is what makes the game finishable: before it, only 4 of 40 random games reached
10 points, because a settlement needs four different resources and most players' buildings
reach only three.
"""

import collections
import random

import pytest

import catan.topology as T
from catan import rules
from catan.actions import ActionType, build_road, build_settlement, trade_with_bank
from catan.board import GENERIC_HARBOUR, HARBOUR_SPACING, HARBOUR_TYPES, Board
from catan.resources import (
    BANK_PER_RESOURCE,
    BANK_RATE,
    GENERIC_HARBOUR_RATE,
    NUM_RESOURCES,
    ROAD_COST,
    SPECIFIC_HARBOUR_RATE,
    can_afford,
    Resource,
)
from catan.rules import IllegalAction
from catan.state import Piece
from helpers import (
    complete_setup,
    enough_for_everything,
    fresh,
    give,
    in_build_phase,
    put_building,
    put_road,
)


def make_board(seed=0):
    return Board(rng=random.Random(seed))


# =========================================================================== #
# HARBOURS                                                                    #
# =========================================================================== #

def test_there_are_nine_harbours_four_generic_and_one_per_resource():
    board = make_board()
    assert len(board.harbours) == 9
    kinds = collections.Counter(board.harbours.values())
    assert kinds[GENERIC_HARBOUR] == 4
    for resource in Resource:
        assert kinds[resource] == 1
    assert sorted(HARBOUR_TYPES, key=lambda h: (h is not GENERIC_HARBOUR, h)) == \
        sorted(board.harbours.values(), key=lambda h: (h is not GENERIC_HARBOUR, h))


def test_harbours_sit_on_coastal_roads():
    board = make_board()
    assert set(board.harbours) <= set(T.COASTAL_ROADS)


def test_harbour_spacing_covers_the_coastline_exactly():
    assert sum(HARBOUR_SPACING) == len(T.COASTAL_ROADS) == 30
    assert len(HARBOUR_SPACING) == 9


def test_harbours_are_evenly_spread_and_never_share_a_vertex():
    board = make_board()
    positions = sorted(T.COASTAL_CYCLE.index(road) for road in board.harbours)
    gaps = [b - a for a, b in zip(positions, positions[1:])]
    gaps.append(len(T.COASTAL_CYCLE) - positions[-1] + positions[0])
    assert set(gaps) <= {3, 4}, f"uneven harbour spacing: {gaps}"

    vertices = [v for road in board.harbours for v in T.ROAD_VERTICES[road]]
    assert len(vertices) == len(set(vertices)) == 18


def test_both_endpoints_of_a_harbour_road_grant_it():
    """A harbour is on an edge; either of its two settlement spots gets the port."""
    board = make_board()
    for road, harbour in board.harbours.items():
        for vertex in T.ROAD_VERTICES[road]:
            assert harbour in board.harbours_at(vertex)


def test_vertices_away_from_a_harbour_grant_nothing():
    board = make_board()
    harbour_vertices = {v for r in board.harbours for v in T.ROAD_VERTICES[r]}
    for vertex in range(1, T.NUM_VERTICES + 1):
        if vertex not in harbour_vertices:
            assert board.harbours_at(vertex) == frozenset()


def test_harbour_placement_is_reproducible_and_varies_by_seed():
    assert make_board(7).harbours == make_board(7).harbours
    layouts = {tuple(sorted(make_board(s).harbours.items(),
                            key=lambda kv: kv[0])) for s in range(10)}
    assert len(layouts) > 1


def test_harbour_positions_are_fixed_only_the_types_move():
    """Position comes from the coastline walk; the seed only shuffles which harbour
    lands where — docs/decisions/0010-harbour-placement.md."""
    positions = {frozenset(make_board(seed).harbours) for seed in range(15)}
    assert len(positions) == 1, "harbour positions should not depend on the seed"


def test_harbours_are_part_of_the_board_layout():
    """Equality and hashing must account for harbours, not just tiles."""
    board = make_board(3)
    assert board.layout[2] == tuple(sorted(board.harbours.items(),
                                           key=lambda kv: kv[0]))

    twin = make_board(3)
    assert board == twin and hash(board) == hash(twin)

    # same tiles, different harbours -> a different board
    twin.harbours = {road: GENERIC_HARBOUR for road in twin.harbours}
    assert board != twin


# =========================================================================== #
# TRADE RATES                                                                 #
# =========================================================================== #

def test_without_a_harbour_everything_is_four_to_one():
    state = fresh(seed=1)
    put_building(state, 1, _vertex_without_harbour(state))
    assert rules.trade_rates(state, 1) == [BANK_RATE] * NUM_RESOURCES


def test_a_generic_harbour_gives_three_to_one_on_everything():
    state = fresh(seed=1)
    vertex = _harbour_vertex(state, GENERIC_HARBOUR)
    put_building(state, 1, vertex)
    assert rules.trade_rates(state, 1) == [GENERIC_HARBOUR_RATE] * NUM_RESOURCES


def test_a_specific_harbour_gives_two_to_one_on_its_resource_only():
    state = fresh(seed=1)
    for resource in Resource:
        probe = fresh(seed=1)
        put_building(probe, 1, _harbour_vertex(probe, resource))
        rates = rules.trade_rates(probe, 1)
        assert rates[resource] == SPECIFIC_HARBOUR_RATE
        for other in Resource:
            if other != resource:
                assert rates[other] == BANK_RATE


def test_harbours_combine_to_the_best_rate_per_resource():
    state = fresh(seed=1)
    put_building(state, 1, _harbour_vertex(state, GENERIC_HARBOUR))
    put_building(state, 1, _harbour_vertex(state, Resource.ORE))
    rates = rules.trade_rates(state, 1)
    assert rates[Resource.ORE] == SPECIFIC_HARBOUR_RATE
    for other in Resource:
        if other != Resource.ORE:
            assert rates[other] == GENERIC_HARBOUR_RATE


def test_a_city_grants_a_harbour_just_like_a_settlement():
    state = fresh(seed=1)
    put_building(state, 1, _harbour_vertex(state, GENERIC_HARBOUR), Piece.CITY)
    assert rules.trade_rates(state, 1) == [GENERIC_HARBOUR_RATE] * NUM_RESOURCES


def test_an_opponents_harbour_does_not_help_you():
    state = fresh(seed=1)
    put_building(state, 2, _harbour_vertex(state, GENERIC_HARBOUR))
    assert rules.trade_rates(state, 1) == [BANK_RATE] * NUM_RESOURCES


# =========================================================================== #
# TRADING                                                                     #
# =========================================================================== #

def test_a_four_to_one_trade_moves_exactly_four_for_one():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    put_building(state, 1, _vertex_without_harbour(state))
    give(state, 1, sheep=4)
    before = list(state.bank)

    rules.apply(state, trade_with_bank(Resource.SHEEP, Resource.ORE))

    assert state.hands[1][Resource.SHEEP] == 0
    assert state.hands[1][Resource.ORE] == 1
    assert state.bank[Resource.SHEEP] == before[Resource.SHEEP] + 4
    assert state.bank[Resource.ORE] == before[Resource.ORE] - 1


def test_a_harbour_reduces_what_the_trade_costs():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    put_building(state, 1, _harbour_vertex(state, GENERIC_HARBOUR))
    give(state, 1, sheep=3)

    rules.apply(state, trade_with_bank(Resource.SHEEP, Resource.ORE))
    assert state.hands[1][Resource.SHEEP] == 0
    assert state.hands[1][Resource.ORE] == 1


def test_trading_needs_enough_cards():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    put_building(state, 1, _vertex_without_harbour(state))
    give(state, 1, sheep=3)  # one short of 4:1

    trade = trade_with_bank(Resource.SHEEP, Resource.ORE)
    assert trade not in rules.legal_actions(state)
    with pytest.raises(IllegalAction):
        rules.apply(state, trade)
    assert state.hands[1][Resource.SHEEP] == 3


def test_you_cannot_trade_a_resource_for_itself():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    give(state, 1, sheep=9)
    trade = trade_with_bank(Resource.SHEEP, Resource.SHEEP)
    assert trade not in rules.legal_actions(state)
    with pytest.raises(IllegalAction):
        rules.apply(state, trade)


def test_you_cannot_take_a_resource_the_bank_has_run_out_of():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    give(state, 1, sheep=9)
    state.bank[Resource.ORE] = 0

    trade = trade_with_bank(Resource.SHEEP, Resource.ORE)
    assert trade not in rules.legal_actions(state)
    with pytest.raises(IllegalAction):
        rules.apply(state, trade)


@pytest.mark.parametrize("give_res,take_res", [(-1, 0), (0, 5), (5, 0), (0, -3)])
def test_trading_a_non_resource_raises(give_res, take_res):
    state = fresh(seed=1)
    in_build_phase(state, 1)
    enough_for_everything(state, 1, times=20)
    with pytest.raises(IllegalAction):
        rules.apply(state, trade_with_bank(give_res, take_res))


def test_trades_are_offered_alongside_builds():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    put_building(state, 1, _vertex_without_harbour(state))
    give(state, 1, sheep=8)

    offered = rules.legal_actions(state)
    trades = [a for a in offered if a.type is ActionType.TRADE_WITH_BANK]
    assert len(trades) == 4, "sheep for each of the other four resources"
    assert all(a.position == Resource.SHEEP for a in trades)
    assert {a.extra for a in trades} == {r for r in Resource if r != Resource.SHEEP}


def test_trading_cannot_be_done_before_rolling():
    state = fresh(seed=1)
    complete_setup(state)
    give(state, state.current_player, sheep=9)
    assert rules.legal_actions(state) == []
    with pytest.raises(IllegalAction):
        rules.apply(state, trade_with_bank(Resource.SHEEP, Resource.ORE))


def test_trading_lets_a_stuck_player_build():
    """The exact situation that made 90% of Phase 1 games unwinnable: a big pile of
    the wrong resources and no wood."""
    state = fresh(seed=1)
    in_build_phase(state, 1)
    put_building(state, 1, _vertex_without_harbour(state))
    give(state, 1, brick=8, sheep=8)

    assert not can_afford(state.hands[1], ROAD_COST), "no wood yet"
    assert not [a for a in rules.legal_actions(state)
                if a.type is ActionType.BUILD_ROAD], "cannot afford a road"

    rules.apply(state, trade_with_bank(Resource.SHEEP, Resource.WOOD))

    assert can_afford(state.hands[1], ROAD_COST)
    roads = [a for a in rules.legal_actions(state) if a.type is ActionType.BUILD_ROAD]
    assert roads, "the trade unblocked road building"
    rules.apply(state, roads[0])


# =========================================================================== #
# THE BANK                                                                    #
# =========================================================================== #

def test_the_bank_starts_with_nineteen_of_each():
    state = fresh()
    assert state.bank == [BANK_PER_RESOURCE] * NUM_RESOURCES


def test_production_comes_out_of_the_bank():
    state = fresh(seed=3)
    vertex, roll, resource = _producing_vertex(state)
    put_building(state, 1, vertex)

    rules.distribute(state, roll)
    assert state.hands[1][resource] == 1
    assert state.bank[resource] == BANK_PER_RESOURCE - 1


def test_paying_for_a_build_returns_the_cards_to_the_bank():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    put_building(state, 1, 20)
    give(state, 1, wood=1, brick=1)
    state.bank[Resource.WOOD] = 5
    state.bank[Resource.BRICK] = 5

    rules.apply(state, build_road(T.VERTEX_ROADS[20][0]))
    assert state.bank[Resource.WOOD] == 6
    assert state.bank[Resource.BRICK] == 6


def test_a_single_claimant_takes_whatever_is_left():
    """Official shortage rule, the one-player case."""
    state = fresh(seed=3)
    vertex, roll, resource = _producing_vertex(state)
    put_building(state, 1, vertex, Piece.CITY)  # owed 2
    state.bank[resource] = 1

    paid = rules.distribute(state, roll)
    assert paid[1][resource] == 1
    assert state.hands[1][resource] == 1
    assert state.bank[resource] == 0


def test_when_the_bank_is_short_and_several_are_owed_nobody_gets_any():
    """Official shortage rule, the multi-player case — the surprising one."""
    state = fresh(seed=3)
    roll, resource, vertices = _two_claimant_setup(state)
    put_building(state, 1, vertices[0])
    put_building(state, 2, vertices[1])
    state.bank[resource] = 1  # two players owed 1 each, only 1 available

    paid = rules.distribute(state, roll)
    assert paid[1][resource] == 0 and paid[2][resource] == 0
    assert state.hands[1][resource] == 0 and state.hands[2][resource] == 0
    assert state.bank[resource] == 1, "the card stays in the bank"


def test_a_shortage_in_one_resource_does_not_block_another():
    state = fresh(seed=3)
    vertex, roll, resource = _producing_vertex(state)
    put_building(state, 1, vertex)
    state.bank[resource] = 0

    paid = rules.distribute(state, roll)
    assert paid[1][resource] == 0
    assert all(n >= 0 for n in state.bank)


def test_the_setup_payout_also_comes_from_the_bank():
    state = fresh(seed=5)
    rng = random.Random(0)
    while state.setup_round == 1:
        rules.apply(state, rng.choice(rules.legal_actions(state)))

    before = list(state.bank)
    player = state.current_player
    vertex = rules.legal_actions(state)[0].position
    expected = collections.Counter(state.board.resources_at(vertex))
    rules.apply(state, build_settlement(vertex))

    for resource, count in expected.items():
        assert state.bank[resource] == before[resource] - count
        assert state.hands[player][resource] >= count


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #

def _harbour_vertex(state, harbour):
    """A buildable vertex granting ``harbour``. ``None`` matches a generic 3:1."""
    for road, kind in state.board.harbours.items():
        if kind != harbour:
            continue
        for vertex in T.ROAD_VERTICES[road]:
            if rules.respects_distance_rule(state, vertex):
                return vertex
    raise AssertionError(f"no free vertex for harbour {harbour}")


def _vertex_without_harbour(state):
    for vertex in range(1, T.NUM_VERTICES + 1):
        if not state.board.harbours_at(vertex) and rules.respects_distance_rule(state, vertex):
            return vertex
    raise AssertionError("no harbour-free vertex")


def _producing_vertex(state):
    for roll in range(2, 13):
        for vertex, productions in state.board.producers_for(roll).items():
            return vertex, roll, productions[0].resource
    raise AssertionError("board produces nothing")


def _two_claimant_setup(state):
    """A roll and two non-adjacent vertices that both collect the same resource on it."""
    for roll in range(2, 13):
        producers = state.board.producers_for(roll)
        by_resource = collections.defaultdict(list)
        for vertex, productions in producers.items():
            by_resource[productions[0].resource].append(vertex)
        for resource, vertices in by_resource.items():
            for a in vertices:
                for b in vertices:
                    if a < b and b not in T.VERTEX_NEIGHBOURS[a]:
                        return roll, resource, (a, b)
    raise AssertionError("no two-claimant configuration on this board")

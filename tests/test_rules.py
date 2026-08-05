"""catan.rules: legality, application, setup, production, scoring."""

import collections
import random

import pytest

import catan.topology as T
from catan import rules
from catan.actions import Action, ActionType, build_city, build_road, build_settlement, end_turn
from catan.board import ROBBER_ROLL
from catan.resources import CITY_COST, ROAD_COST, SETTLEMENT_COST, Resource, can_afford
from catan.rules import IllegalAction
from catan.rulesets import BASE_GAME
from catan.state import (
    MAX_ROADS,
    NO_OWNER,
    Phase,
    Piece,
)
from helpers import (
    complete_setup,
    enough_for_everything,
    extend_to_free_vertex,
    fresh,
    give,
    in_build_phase,
    put_building,
    put_road,
    roll_sequence,
    roll_to_build,
)


# =========================================================================== #
# THE CENTRAL GUARANTEE                                                       #
# =========================================================================== #

def every_possible_action():
    """Every action the game can express, legal or not.

    Enumerated exhaustively so `apply` can be checked against `legal_actions` over the
    whole space rather than a sample — which is what catches an action that one accepts
    and the other never offers.
    """
    from catan.actions import (
        buy_dev_card,
        discard,
        move_robber,
        play_knight,
        play_monopoly,
        play_road_building,
        play_year_of_plenty,
    )
    from catan.resources import NUM_RESOURCES

    actions = [end_turn(), buy_dev_card(), play_knight(), play_road_building()]
    actions += [build_road(r) for r in range(1, T.NUM_ROADS + 1)]
    actions += [build_settlement(v) for v in range(1, T.NUM_VERTICES + 1)]
    actions += [build_city(v) for v in range(1, T.NUM_VERTICES + 1)]
    actions += [
        rules.trade_with_bank(give, take)
        for give in range(NUM_RESOURCES)
        for take in range(NUM_RESOURCES)
    ]
    actions += [discard(r) for r in range(NUM_RESOURCES)]
    actions += [
        move_robber(tile, victim)
        for tile in range(1, T.NUM_TILES + 1)
        for victim in range(0, 5)
    ]
    actions += [play_monopoly(r) for r in range(NUM_RESOURCES)]
    actions += [
        play_year_of_plenty(a, b)
        for a in range(NUM_RESOURCES)
        for b in range(NUM_RESOURCES)
    ]
    return actions


ALL_ACTIONS = every_possible_action()


@pytest.mark.parametrize("num_players", [2, 3])
def test_apply_accepts_exactly_what_legal_actions_offers(num_players):
    """The old bug: `check_valid_*` computed legal moves and `place_*` ignored them, so
    `place_road(p, 70)` succeeded on an empty board. One authority now.

    Checked over the *whole* action space at every step, in both directions.
    """
    rng = random.Random(0)
    state = fresh(num_players=num_players, seed=1)
    complete_setup(state, rng)

    for _ in range(150):
        if state.phase is Phase.GAME_OVER:
            break
        if state.phase is Phase.ROLL and not rules.legal_actions(state):
            rules.roll_dice(state)
            continue

        offered = set(rules.legal_actions(state))
        assert len(offered) == len(rules.legal_actions(state)), "duplicate actions offered"

        for action in ALL_ACTIONS:
            if action in offered:
                rules.apply(state.clone(), action)  # must be accepted
            else:
                with pytest.raises(IllegalAction):
                    rules.apply(state.clone(), action)

        if state.phase is Phase.ROLL:
            rules.roll_dice(state)
        else:
            rules.apply(state, rng.choice(sorted(offered)))


# =========================================================================== #
# SETUP                                                                       #
# =========================================================================== #

def test_setup_starts_by_offering_every_vertex():
    state = fresh()
    actions = rules.legal_actions(state)
    assert len(actions) == T.NUM_VERTICES
    assert {a.position for a in actions} == set(range(1, T.NUM_VERTICES + 1))
    assert all(a.type is ActionType.BUILD_SETTLEMENT for a in actions)


def test_a_placed_settlement_removes_itself_and_its_neighbours():
    state = fresh()
    rules.apply(state, build_settlement(20))
    state.phase = Phase.SETUP_SETTLEMENT  # skip the road for this check

    offered = {a.position for a in rules.legal_actions(state)}
    assert 20 not in offered
    assert not (set(T.VERTEX_NEIGHBOURS[20]) & offered)
    assert 31 in offered  # two steps away is fine


def test_the_setup_road_must_touch_the_settlement_just_placed():
    state = fresh()
    rules.apply(state, build_settlement(20))
    assert state.phase is Phase.SETUP_ROAD

    offered = {a.position for a in rules.legal_actions(state)}
    assert offered == set(T.VERTEX_ROADS[20])
    assert all(a.type is ActionType.BUILD_ROAD for a in offered and rules.legal_actions(state))


def test_setup_placements_are_free():
    state = fresh()
    rules.apply(state, build_settlement(20))
    rules.apply(state, build_road(T.VERTEX_ROADS[20][0]))
    assert state.hands[1] == [0, 0, 0, 0, 0]


def test_setup_consumes_pieces():
    state = fresh()
    rules.apply(state, build_settlement(20))
    rules.apply(state, build_road(T.VERTEX_ROADS[20][0]))
    assert state.settlements_left[1] == 4
    assert state.roads_left[1] == MAX_ROADS - 1


def test_the_second_settlement_pays_out_its_tiles_and_the_first_does_not():
    state = fresh(seed=5)
    order = state.player_order

    # round 1: no payout
    rules.apply(state, build_settlement(20))
    assert sum(state.hands[order[0]]) == 0
    rules.apply(state, build_road(T.VERTEX_ROADS[20][0]))

    # play out the rest of round 1
    rng = random.Random(0)
    while state.setup_round == 1:
        rules.apply(state, rng.choice(rules.legal_actions(state)))

    # round 2: the settlement pays its adjacent tiles
    player = state.current_player
    before = sum(state.hands[player])
    vertex = rules.legal_actions(state)[0].position
    expected = len(state.board.resources_at(vertex))
    rules.apply(state, build_settlement(vertex))
    assert sum(state.hands[player]) == before + expected
    assert expected >= 1


def test_setup_follows_the_snake_order():
    state = fresh(num_players=3, player_order=[2, 3, 1], seed=2)
    rng = random.Random(0)
    seen = []
    while state.in_setup:
        if state.phase is Phase.SETUP_SETTLEMENT:
            seen.append(state.current_player)
        rules.apply(state, rng.choice(rules.legal_actions(state)))
    assert seen == [2, 3, 1, 1, 3, 2]


def test_setup_ends_in_the_roll_phase_with_the_first_player():
    state = fresh(num_players=3, player_order=[2, 3, 1], seed=2)
    complete_setup(state)
    assert state.phase is Phase.ROLL
    assert state.current_player == 2
    assert state.turn_number == 0


def test_every_player_ends_setup_with_two_settlements_and_two_roads():
    state = fresh(num_players=4, seed=6)
    complete_setup(state)
    for player in state.players:
        assert len(state.buildings_of(player)) == 2
        assert len(state.roads_of(player)) == 2
        assert state.settlements_left[player] == 3
        assert state.roads_left[player] == MAX_ROADS - 2
        assert rules.victory_points(state, player) == 2


def test_setup_rejects_the_wrong_kind_of_action():
    state = fresh()
    with pytest.raises(IllegalAction):
        rules.apply(state, build_road(1))
    rules.apply(state, build_settlement(20))
    with pytest.raises(IllegalAction):
        rules.apply(state, build_settlement(31))


# =========================================================================== #
# DICE AND PRODUCTION                                                         #
# =========================================================================== #

def test_rolling_is_only_allowed_in_the_roll_phase():
    state = fresh(seed=1)
    with pytest.raises(IllegalAction):
        rules.roll_dice(state)  # still in setup
    complete_setup(state)
    rules.roll_dice(state)
    assert state.phase is not Phase.ROLL, "a roll must hand over to some decision"
    with pytest.raises(IllegalAction):
        rules.roll_dice(state)


def test_no_actions_are_offered_before_rolling():
    state = fresh(seed=1)
    complete_setup(state)
    assert state.phase is Phase.ROLL
    assert rules.legal_actions(state) == []


def test_rolls_are_two_dice_not_one_uniform_draw():
    """Catan's whole probability structure is the triangular distribution."""
    state = fresh(seed=1)
    complete_setup(state)
    counts = collections.Counter(roll_sequence(state, 24_000))

    assert set(counts) == set(range(2, 13))
    # expected frequencies are 1:2:3:4:5:6:5:4:3:2:1 out of 36
    assert counts[7] > 4 * counts[2], "7 is six times as likely as 2"
    assert counts[6] > counts[4] > counts[2]
    assert counts[8] > counts[10] > counts[12]
    for roll in range(2, 13):
        expected = (6 - abs(7 - roll)) / 36
        assert abs(counts[roll] / 24_000 - expected) < 0.01


def test_production_pays_the_owner_of_a_settlement():
    state = fresh(seed=3)
    vertex, roll, resource = _find_producing_vertex(state)
    put_building(state, 1, vertex, Piece.SETTLEMENT)

    rules.distribute(state, roll)
    assert state.hands[1][resource] == 1
    assert sum(state.hands[2]) == 0


def test_a_city_produces_double():
    state = fresh(seed=3)
    vertex, roll, resource = _find_producing_vertex(state)
    put_building(state, 1, vertex, Piece.CITY)

    rules.distribute(state, roll)
    assert state.hands[1][resource] == 2


def test_production_skips_unowned_vertices():
    state = fresh(seed=3)
    _, roll, _ = _find_producing_vertex(state)
    rules.distribute(state, roll)
    assert all(sum(state.hands[p]) == 0 for p in state.players)


def test_a_seven_pays_nobody():
    state = fresh(seed=3)
    for vertex in range(1, T.NUM_VERTICES + 1):
        state.vertex_owner[vertex] = 1
        state.vertex_piece[vertex] = Piece.CITY
    rules.distribute(state, ROBBER_ROLL)
    assert sum(state.hands[1]) == 0


def _find_producing_vertex(state):
    for roll in range(2, 13):
        for vertex, productions in state.board.producers_for(roll).items():
            return vertex, roll, productions[0].resource
    raise AssertionError("board produces nothing")


# =========================================================================== #
# BUILDING: COST                                                              #
# =========================================================================== #

def test_building_a_road_costs_wood_and_brick():
    state = fresh(seed=1)
    complete_setup(state)
    in_build_phase(state, 1)
    road = _connected_free_road(state, 1)

    give(state, 1, wood=1, brick=1, sheep=5)
    rules.apply(state, build_road(road))
    assert state.hands[1] == [0, 0, 5, 0, 0]
    assert state.edge_owner[road] == 1


def test_building_without_the_resources_is_not_offered_and_not_allowed():
    """The old engine charged nothing, so one brick and one wood built every road."""
    state = fresh(seed=1)
    complete_setup(state)
    in_build_phase(state, 1)
    road = _connected_free_road(state, 1)

    give(state, 1)  # empty hand
    assert build_road(road) not in rules.legal_actions(state)
    with pytest.raises(IllegalAction):
        rules.apply(state, build_road(road))
    assert state.edge_owner[road] == NO_OWNER


def test_one_brick_and_one_wood_builds_exactly_one_road():
    state = fresh(seed=1)
    complete_setup(state)
    in_build_phase(state, 1)

    give(state, 1, wood=1, brick=1)
    built = 0
    while True:
        roads = [a for a in rules.legal_actions(state)
                 if a.type is ActionType.BUILD_ROAD]
        if not roads:
            break
        rules.apply(state, roads[0])
        built += 1
    assert built == 1


def test_a_settlement_costs_four_resources():
    state = fresh(seed=1)
    complete_setup(state)
    in_build_phase(state, 1)
    vertex = extend_to_free_vertex(state, 1)

    give(state, 1, wood=1, brick=1, sheep=1, wheat=1, ore=4)
    rules.apply(state, build_settlement(vertex))
    assert state.hands[1] == [0, 0, 0, 0, 4]
    assert state.vertex_owner[vertex] == 1


def test_straight_after_setup_there_is_no_legal_settlement_spot():
    """Both starting roads end next to your own settlement, which the distance rule
    blocks — expanding genuinely requires building a road first."""
    state = fresh(seed=1)
    complete_setup(state)
    in_build_phase(state, 1)
    enough_for_everything(state, 1)
    assert not [a for a in rules.legal_actions(state)
                if a.type is ActionType.BUILD_SETTLEMENT]
    assert [a for a in rules.legal_actions(state)
            if a.type is ActionType.BUILD_ROAD]


def test_a_city_costs_two_wheat_and_three_ore():
    state = fresh(seed=1)
    complete_setup(state)
    in_build_phase(state, 1)
    vertex = state.buildings_of(1)[0]

    give(state, 1, wheat=2, ore=3, wood=1)
    rules.apply(state, build_city(vertex))
    assert state.hands[1] == [1, 0, 0, 0, 0]


# =========================================================================== #
# BUILDING: CONNECTIVITY AND SPACING                                          #
# =========================================================================== #

def test_a_road_must_connect_to_the_players_network():
    """`place_road(p, 70)` used to succeed on a completely empty board."""
    state = fresh(seed=1)
    in_build_phase(state, 1)
    enough_for_everything(state, 1)

    offered = rules.legal_actions(state)
    assert not [a for a in offered if a.type is ActionType.BUILD_ROAD], \
        "no road can be built with nothing to connect to"
    with pytest.raises(IllegalAction):
        rules.apply(state, build_road(70))


def test_a_road_may_extend_from_a_building_or_from_another_road():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    enough_for_everything(state, 1)

    put_building(state, 1, 20)
    from_building = set(T.VERTEX_ROADS[20])
    offered = {a.position for a in rules.legal_actions(state)
               if a.type is ActionType.BUILD_ROAD}
    assert offered == from_building

    first = sorted(from_building)[0]
    put_road(state, 1, first)
    offered = {a.position for a in rules.legal_actions(state)
               if a.type is ActionType.BUILD_ROAD}
    assert offered > (from_building - {first}), "should now reach further"


def test_a_settlement_needs_one_of_your_own_roads():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    enough_for_everything(state, 1)

    vertex = 20
    assert build_settlement(vertex) not in rules.legal_actions(state)
    put_road(state, 1, T.VERTEX_ROADS[vertex][0])
    assert build_settlement(vertex) in rules.legal_actions(state)


def test_an_opponents_road_does_not_let_you_settle():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    enough_for_everything(state, 1)
    put_road(state, 2, T.VERTEX_ROADS[20][0])
    assert build_settlement(20) not in rules.legal_actions(state)


def test_the_distance_rule_applies_across_players():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    enough_for_everything(state, 1)
    put_building(state, 2, 20)
    put_road(state, 1, T.VERTEX_ROADS[20][0])

    for neighbour in T.VERTEX_NEIGHBOURS[20]:
        assert build_settlement(neighbour) not in rules.legal_actions(state)
    assert build_settlement(20) not in rules.legal_actions(state)


def test_a_road_cannot_be_built_through_an_opponents_building():
    """Official rule: an opponent's settlement blocks the junction."""
    state = fresh(seed=1)
    in_build_phase(state, 1)
    enough_for_everything(state, 1)

    # player 1 owns a road into vertex 20; player 2 sits on vertex 20
    road_in = T.VERTEX_ROADS[20][0]
    put_road(state, 1, road_in)
    put_building(state, 2, 20)

    beyond = [r for r in T.VERTEX_ROADS[20] if r != road_in]
    offered = {a.position for a in rules.legal_actions(state)
               if a.type is ActionType.BUILD_ROAD}
    assert not (set(beyond) & offered), "must not extend past an opponent's building"


def test_your_own_building_does_not_block_you():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    enough_for_everything(state, 1)

    road_in = T.VERTEX_ROADS[20][0]
    put_road(state, 1, road_in)
    put_building(state, 1, 20)

    beyond = [r for r in T.VERTEX_ROADS[20] if r != road_in]
    offered = {a.position for a in rules.legal_actions(state)
               if a.type is ActionType.BUILD_ROAD}
    assert set(beyond) <= offered


def test_a_road_cannot_be_built_twice():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    enough_for_everything(state, 1)
    put_building(state, 1, 20)
    road = T.VERTEX_ROADS[20][0]
    put_road(state, 2, road)
    assert build_road(road) not in rules.legal_actions(state)


# =========================================================================== #
# CITIES                                                                      #
# =========================================================================== #

def test_a_city_must_upgrade_your_own_settlement():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    enough_for_everything(state, 1)

    put_building(state, 2, 20, Piece.SETTLEMENT)
    put_building(state, 1, 31, Piece.SETTLEMENT)

    assert build_city(20) not in rules.legal_actions(state)   # opponent's
    assert build_city(40) not in rules.legal_actions(state)   # empty
    assert build_city(31) in rules.legal_actions(state)


def test_a_city_cannot_be_upgraded_again():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    enough_for_everything(state, 1)
    put_building(state, 1, 31, Piece.SETTLEMENT)
    rules.apply(state, build_city(31))
    assert state.vertex_piece[31] is Piece.CITY
    assert build_city(31) not in rules.legal_actions(state)


def test_upgrading_returns_the_settlement_to_the_supply():
    """Official rule: the settlement piece goes back and can be rebuilt."""
    state = fresh(seed=1)
    in_build_phase(state, 1)
    enough_for_everything(state, 1)
    put_building(state, 1, 31, Piece.SETTLEMENT)
    state.settlements_left[1] = 4
    state.cities_left[1] = 4

    rules.apply(state, build_city(31))
    assert state.settlements_left[1] == 5
    assert state.cities_left[1] == 3


def test_a_city_is_worth_two_points_not_three():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    enough_for_everything(state, 1)
    put_building(state, 1, 31, Piece.SETTLEMENT)
    assert rules.victory_points(state, 1) == 1
    rules.apply(state, build_city(31))
    assert rules.victory_points(state, 1) == 2


# =========================================================================== #
# PIECE LIMITS                                                                #
# =========================================================================== #

def test_running_out_of_roads_stops_road_building():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    enough_for_everything(state, 1, times=99)
    put_building(state, 1, 20)
    state.roads_left[1] = 0
    assert not [a for a in rules.legal_actions(state)
                if a.type is ActionType.BUILD_ROAD]


def test_running_out_of_cities_stops_upgrading():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    enough_for_everything(state, 1)
    put_building(state, 1, 31, Piece.SETTLEMENT)
    state.cities_left[1] = 0
    assert build_city(31) not in rules.legal_actions(state)


# =========================================================================== #
# TURNS AND WINNING                                                           #
# =========================================================================== #

def test_ending_a_turn_advances_to_the_next_player_and_the_roll_phase():
    state = fresh(num_players=3, player_order=[3, 1, 2], seed=1)
    complete_setup(state)
    roll_to_build(state)
    assert state.current_player == 3

    rules.apply(state, end_turn())
    assert state.phase is Phase.ROLL
    assert state.current_player == 1


def test_end_turn_is_always_available_so_a_player_can_never_be_stuck():
    state = fresh(seed=1)
    complete_setup(state)
    for _ in range(50):
        if state.phase is Phase.ROLL:
            roll_to_build(state)
        assert end_turn() in rules.legal_actions(state)
        rules.apply(state, end_turn())


def test_reaching_the_win_condition_ends_the_game():
    # base game, so ten points; ranked 1v1 needs fifteen
    state = fresh(seed=1, ruleset=BASE_GAME)
    in_build_phase(state, 1)
    enough_for_everything(state, 1)

    # 4 cities (8) + 1 settlement (1) = 9, then upgrade one more to cross 10
    spots = _spaced_vertices(state, 5)
    for vertex in spots:
        put_building(state, 1, vertex, Piece.SETTLEMENT)
    assert rules.victory_points(state, 1) == 5

    for vertex in spots[:4]:
        rules.apply(state, build_city(vertex))
        enough_for_everything(state, 1)
    assert rules.victory_points(state, 1) == 9
    assert state.phase is Phase.BUILD
    assert state.winner is None

    # one more settlement crosses the line
    state.settlements_left[1] = 5
    extra = _spaced_vertices(state, 1, avoid=spots)[0]
    put_building(state, 1, extra, Piece.SETTLEMENT)
    rules._check_for_winner(state, 1)

    assert rules.victory_points(state, 1) >= state.ruleset.victory_points_to_win
    assert state.winner == 1
    assert state.phase is Phase.GAME_OVER
    assert rules.legal_actions(state) == []


def test_nothing_can_be_applied_after_the_game_ends():
    state = fresh(seed=1)
    state.phase = Phase.GAME_OVER
    state.winner = 1
    with pytest.raises(IllegalAction):
        rules.apply(state, end_turn())


def test_scores_reports_every_player():
    state = fresh(num_players=4, seed=1)
    complete_setup(state)
    assert rules.scores(state) == {1: 2, 2: 2, 3: 2, 4: 2}


# =========================================================================== #
# DEFENSIVE                                                                   #
# =========================================================================== #

@pytest.mark.parametrize("position", [0, -1, 55, 999])
def test_out_of_range_vertices_raise(position):
    state = fresh(seed=1)
    in_build_phase(state, 1)
    enough_for_everything(state, 1)
    with pytest.raises(IllegalAction):
        rules.apply(state, build_settlement(position))


@pytest.mark.parametrize("position", [0, -1, 73, 999])
def test_out_of_range_roads_raise(position):
    state = fresh(seed=1)
    in_build_phase(state, 1)
    enough_for_everything(state, 1)
    with pytest.raises(IllegalAction):
        rules.apply(state, build_road(position))


def test_applying_a_non_action_raises():
    state = fresh(seed=1)
    for junk in ("build_road", 3, None, (1, 2)):
        with pytest.raises(IllegalAction):
            rules.apply(state, junk)


def test_an_unknown_action_type_raises():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    with pytest.raises(IllegalAction):
        rules.apply(state, Action(99, 1))


@pytest.mark.parametrize("action", [
    Action(99, 1),
    Action(-1, 0),
    Action(ActionType.TRADE_WITH_BANK, -1, 0),
    Action(ActionType.TRADE_WITH_BANK, 0, 99),
    Action(ActionType.BUILD_ROAD, 999),
])
def test_the_repr_of_a_malformed_action_never_raises(action):
    """This has bitten twice: once on a bad action type, once on a bad resource index.

    The repr appears inside the IllegalAction message raised *because* the action is
    malformed, so an exception here replaces a clear error with a confusing one.
    """
    text = repr(action)
    assert isinstance(text, str) and text


@pytest.mark.parametrize("action", [
    Action(99, 1),
    Action(ActionType.TRADE_WITH_BANK, -1, 0),
    Action(ActionType.TRADE_WITH_BANK, 0, 99),
])
def test_a_malformed_action_reports_illegal_not_a_crash(action):
    state = fresh(seed=1)
    in_build_phase(state, 1)
    enough_for_everything(state, 1, times=20)
    with pytest.raises(IllegalAction):
        rules.apply(state, action)


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #

def _connected_free_road(state, player):
    for road in range(1, T.NUM_ROADS + 1):
        if state.is_road_free(road) and any(
            state.vertex_owner[v] == player for v in T.ROAD_VERTICES[road]
        ):
            return road
    raise AssertionError("no connected free road")


def _spaced_vertices(state, count, avoid=()):
    """`count` mutually non-adjacent free vertices."""
    chosen = list(avoid)
    out = []
    for vertex in range(1, T.NUM_VERTICES + 1):
        if state.vertex_owner[vertex] != NO_OWNER:
            continue
        if any(v == vertex or v in T.VERTEX_NEIGHBOURS[vertex] for v in chosen):
            continue
        chosen.append(vertex)
        out.append(vertex)
        if len(out) == count:
            return out
    raise AssertionError(f"could not find {count} spaced vertices")


# --------------------------------------------------------------------------- #
# legal_actions translates the predicates; it does not decide                  #
# --------------------------------------------------------------------------- #

def test_the_road_loop_agrees_with_can_build_road():
    """``legal_actions`` hoists the pieces-and-payment half of ``can_build_road`` out of the
    72-road loop, because the gate above it has just established both and neither can change
    while the loop runs. That is an optimisation with a second authority hiding in it, so it
    is pinned: ``can_build_road`` stays the definition, and the loop has to match it exactly,
    including the order the actions come out in.
    """
    from catan.rulesets import ALL
    from catan.state import GameState
    from helpers import drive

    def check(state):
        if state.phase is not Phase.BUILD:
            return
        player = state.current_player
        offered = [a for a in rules.legal_actions(state)
                   if a.type is ActionType.BUILD_ROAD]
        authoritative = [build_road(road) for road in range(1, T.NUM_ROADS + 1)
                         if rules.can_build_road(state, player, road)]
        assert offered == authoritative

    for ruleset in ALL:
        for num_players in (2, 3, 4):
            state = GameState(num_players=num_players, seed=num_players, ruleset=ruleset)
            drive(state, random.Random(num_players ^ 0x5A), max_actions=600,
                  on_step=check)

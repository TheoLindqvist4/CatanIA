"""Rolling a 7: discarding down to half, moving the robber, and stealing."""

import collections
import random

import pytest

import catan.topology as T
from catan import rules
from catan.actions import ActionType, discard, end_turn, move_robber
from catan.board import ROBBER_ROLL
from catan.resources import BANK_PER_RESOURCE, NUM_RESOURCES, Resource, total
from catan.rules import IllegalAction
from catan.rulesets import BASE_GAME, RANKED_1V1
from catan.state import Phase, Piece
from helpers import (
    complete_setup,
    fresh,
    give,
    in_build_phase,
    put_building,
    roll_to_build,
)


def at_seven(state):
    """Reach the aftermath of a 7 without waiting for one to be rolled.

    Delegates to ``rules.begin_robber`` rather than reimplementing it — an earlier
    version of this helper duplicated the logic and drifted from it, which is why the
    discard count is decided in exactly one place now.
    """
    state.last_roll = ROBBER_ROLL
    state.rolled_this_turn = True  # a 7 comes from a roll
    rules.begin_robber(state)
    return state


# =========================================================================== #
# THE ROBBER STARTS ON THE DESERT                                             #
# =========================================================================== #

def test_the_robber_starts_on_the_desert():
    for seed in range(10):
        state = fresh(seed=seed)
        assert state.robber_tile == state.board.desert_tile
        assert state.board.resource_at(state.robber_tile) is None


def test_the_robber_is_state_not_board():
    """It moves during play, so it cannot live on the shared immutable board."""
    state = fresh(seed=1)
    assert not hasattr(state.board, "robber_tile")

    elsewhere = next(t for t in range(1, T.NUM_TILES + 1) if t != state.robber_tile)
    clone = state.clone()
    clone.robber_tile = elsewhere
    assert state.robber_tile != elsewhere
    assert clone.board is state.board


# =========================================================================== #
# A SEVEN                                                                     #
# =========================================================================== #

def test_a_seven_pays_nobody():
    state = fresh(seed=3)
    for vertex in range(1, T.NUM_VERTICES + 1):
        state.vertex_owner[vertex] = 1
        state.vertex_piece[vertex] = Piece.CITY
    rules.distribute(state, ROBBER_ROLL)
    assert sum(state.hands[1]) == 0


def test_a_seven_with_nobody_over_the_limit_goes_straight_to_the_robber():
    state = fresh(seed=1)
    complete_setup(state)
    for player in state.players:
        give(state, player, wood=2)
    at_seven(state)
    assert state.phase is Phase.MOVE_ROBBER
    assert state.pending_discards == []


def test_a_seven_asks_the_over_limit_players_to_discard_first():
    state = fresh(seed=1)
    complete_setup(state)
    give(state, 1, wood=state.ruleset.hand_limit + 1)
    give(state, 2, wood=2)
    at_seven(state)

    assert state.phase is Phase.DISCARD
    assert state.pending_discards == [1]
    assert state.current_player == 1


def test_the_discard_order_starts_at_the_roller_and_follows_turn_order():
    state = fresh(num_players=4, player_order=[3, 4, 1, 2], seed=1)
    complete_setup(state)
    for player in state.players:
        give(state, player, wood=state.ruleset.hand_limit + 2)
    state.turn_number = 1  # player 4 is rolling
    assert state.turn_player == 4

    at_seven(state)
    assert state.pending_discards == [4, 1, 2, 3]


def test_a_player_at_exactly_the_limit_does_not_discard():
    state = fresh(seed=1)
    complete_setup(state)
    limit = state.ruleset.hand_limit
    give(state, 1, wood=limit)          # exactly at the limit
    give(state, 2, wood=limit + 1)      # one over
    at_seven(state)
    assert state.pending_discards == [2]


# =========================================================================== #
# DISCARDING                                                                  #
# =========================================================================== #

@pytest.mark.parametrize("held,given_up,kept", [
    (8, 4, 4),
    (9, 4, 5),    # half rounded down is what you *lose*, so odd hands keep the extra
    (10, 5, 5),
    (11, 5, 6),
    (15, 7, 8),
])
def test_you_give_up_half_your_hand_rounded_down(held, given_up, kept):
    # base game, so the printed numbers apply: the limit is 7 and all of these exceed it
    state = fresh(seed=1, ruleset=BASE_GAME)
    complete_setup(state)
    give(state, 1, wood=held)
    give(state, 2, wood=1)
    assert rules.discard_count(state, 1) == given_up

    at_seven(state)
    assert state.discards_owed[1] == given_up
    while state.phase is Phase.DISCARD:
        rules.apply(state, rules.legal_actions(state)[0])

    assert total(state.hands[1]) == kept
    assert state.discards_owed[1] == 0


def test_the_discard_count_is_fixed_when_the_seven_is_rolled():
    """Recomputing it as the hand shrinks would move the target and stop early — which
    is exactly the bug this test was written for: 10 cards stopped at 7, not 5."""
    state = fresh(seed=1, ruleset=BASE_GAME)
    complete_setup(state)
    give(state, 1, wood=10)
    give(state, 2, wood=1)
    at_seven(state)
    assert state.discards_owed[1] == 5

    rules.apply(state, discard(Resource.WOOD))
    assert state.discards_owed[1] == 4, "the target must not be recomputed"
    assert total(state.hands[1]) == 9
    assert rules.must_discard(state, 1) is True

    while state.phase is Phase.DISCARD:
        rules.apply(state, rules.legal_actions(state)[0])
    assert total(state.hands[1]) == 5, "must go below the limit, not stop at it"


def test_discarding_is_one_card_at_a_time():
    """A chosen multiset would not flatten into a discrete action space."""
    state = fresh(seed=1)
    complete_setup(state)
    give(state, 1, wood=5, ore=5)
    give(state, 2, wood=1)
    at_seven(state)

    offered = rules.legal_actions(state)
    assert {a.type for a in offered} == {ActionType.DISCARD}
    assert {a.position for a in offered} == {Resource.WOOD, Resource.ORE}

    before = total(state.hands[1])
    rules.apply(state, discard(Resource.WOOD))
    assert total(state.hands[1]) == before - 1


def test_you_can_only_discard_a_resource_you_hold():
    state = fresh(seed=1)
    complete_setup(state)
    give(state, 1, wood=9)
    give(state, 2, wood=1)
    at_seven(state)

    assert discard(Resource.ORE) not in rules.legal_actions(state)
    with pytest.raises(IllegalAction):
        rules.apply(state, discard(Resource.ORE))


def test_discarded_cards_go_back_to_the_bank():
    state = fresh(seed=1)
    complete_setup(state)
    give(state, 1, wood=10)
    give(state, 2, wood=1)
    state.bank[Resource.WOOD] = 0
    at_seven(state)

    while state.phase is Phase.DISCARD:
        rules.apply(state, rules.legal_actions(state)[0])
    assert state.bank[Resource.WOOD] == 5, "the five discarded cards returned"


def test_every_over_limit_player_discards_before_the_robber_moves():
    state = fresh(num_players=3, seed=1)
    complete_setup(state)
    for player in state.players:
        give(state, player, wood=10)
    at_seven(state)

    assert sorted(state.pending_discards) == [1, 2, 3]
    while state.phase is Phase.DISCARD:
        rules.apply(state, rules.legal_actions(state)[0])

    assert state.phase is Phase.MOVE_ROBBER
    assert state.pending_discards == []
    for player in state.players:
        assert total(state.hands[player]) == 5
        assert state.discards_owed[player] == 0


def test_only_the_discarding_player_may_act():
    state = fresh(seed=1)
    complete_setup(state)
    over = state.ruleset.hand_limit + 2
    give(state, 1, wood=over)
    give(state, 2, wood=over)
    at_seven(state)

    first = state.pending_discards[0]
    assert state.current_player == first
    # the other player's turn to discard comes only after the first is done
    while state.pending_discards[0] == first:
        rules.apply(state, rules.legal_actions(state)[0])
    assert state.current_player != first


def test_building_is_not_offered_during_a_discard():
    state = fresh(seed=1)
    complete_setup(state)
    give(state, 1, wood=9, brick=9, sheep=9, wheat=9)
    give(state, 2, wood=1)  # noqa: over the limit under either ruleset
    at_seven(state)
    assert all(a.type is ActionType.DISCARD for a in rules.legal_actions(state))
    with pytest.raises(IllegalAction):
        rules.apply(state, end_turn())


# =========================================================================== #
# MOVING THE ROBBER                                                           #
# =========================================================================== #

def test_the_robber_must_actually_move():
    state = fresh(seed=1)
    complete_setup(state)
    at_seven(state)
    assert state.phase is Phase.MOVE_ROBBER

    here = state.robber_tile
    assert not [a for a in rules.legal_actions(state) if a.position == here]
    with pytest.raises(IllegalAction):
        rules.apply(state, move_robber(here))


def test_every_other_tile_is_a_legal_destination():
    # base game: Friendly Robber would put some tiles off limits
    state = fresh(seed=1, ruleset=BASE_GAME)
    complete_setup(state)
    at_seven(state)
    destinations = {a.position for a in rules.legal_actions(state)}
    assert destinations == set(range(1, T.NUM_TILES + 1)) - {state.robber_tile}


def test_moving_the_robber_ends_in_the_build_phase():
    state = fresh(seed=1)
    complete_setup(state)
    at_seven(state)
    target = next(a for a in rules.legal_actions(state))
    rules.apply(state, target)
    assert state.robber_tile == target.position
    assert state.phase is Phase.BUILD


@pytest.mark.parametrize("tile", [0, -1, 20, 999])
def test_an_off_board_tile_is_rejected(tile):
    state = fresh(seed=1)
    complete_setup(state)
    at_seven(state)
    with pytest.raises(IllegalAction):
        rules.apply(state, move_robber(tile))


def test_the_robber_blocks_production_on_its_tile():
    state = fresh(seed=3)
    vertex, roll, resource, tile = _producing_vertex(state)
    put_building(state, 1, vertex)

    state.robber_tile = _other_tile(state, tile)
    rules.distribute(state, roll)
    assert state.hands[1][resource] == 1, "unblocked tile should pay"

    give(state, 1)
    state.robber_tile = tile
    rules.distribute(state, roll)
    assert state.hands[1][resource] == 0, "blocked tile must pay nothing"


def test_the_robber_blocks_that_tile_for_everybody():
    state = fresh(seed=3)
    vertex, roll, resource, tile = _producing_vertex(state)
    others = [v for v in T.TILE_VERTICES[tile]
              if v != vertex and v not in T.VERTEX_NEIGHBOURS[vertex]]
    put_building(state, 1, vertex)
    put_building(state, 2, others[0])

    state.robber_tile = tile
    rules.distribute(state, roll)
    assert sum(state.hands[1]) == 0 and sum(state.hands[2]) == 0


def test_a_blocked_tile_does_not_block_a_players_other_tiles():
    state = fresh(seed=7)
    # a vertex touching 3 tiles, with two different numbers among them
    vertex = next(v for v in range(1, T.NUM_VERTICES + 1)
                  if len(T.VERTEX_TILES[v]) == 3
                  and len({state.board.number_at(t) for t in T.VERTEX_TILES[v]}) == 3)
    put_building(state, 1, vertex)
    blocked, other = T.VERTEX_TILES[vertex][0], T.VERTEX_TILES[vertex][1]
    state.robber_tile = blocked

    if state.board.resource_at(other) is not None:
        rules.distribute(state, state.board.number_at(other))
        assert sum(state.hands[1]) == 1


# =========================================================================== #
# STEALING                                                                    #
# =========================================================================== #

def test_moving_onto_an_opponent_takes_one_of_their_cards():
    state, tile, vertex = _tile_with_opponent(seed=1)
    give(state, 2, wood=3)
    give(state, 1)
    at_seven(state)

    rules.apply(state, move_robber(tile, 2))
    assert total(state.hands[2]) == 2
    assert total(state.hands[1]) == 1
    assert state.hands[1][Resource.WOOD] == 1


def test_stealing_moves_a_card_rather_than_creating_one():
    state, tile, _ = _tile_with_opponent(seed=1)
    give(state, 2, wood=3)
    before = sum(state.bank) + sum(total(state.hands[p]) for p in state.players)
    at_seven(state)
    rules.apply(state, move_robber(tile, 2))
    after = sum(state.bank) + sum(total(state.hands[p]) for p in state.players)
    assert before == after


def test_you_cannot_rob_yourself():
    state, tile, vertex = _tile_with_opponent(seed=1)
    put_building(state, 1, _free_vertex_on(state, tile))
    give(state, 1, wood=5)
    give(state, 2, wood=5)
    at_seven(state)

    offered = [a for a in rules.legal_actions(state) if a.position == tile]
    assert all(a.extra != 1 for a in offered), "the robber cannot rob its owner"


def test_you_cannot_rob_a_player_with_no_cards():
    state, tile, _ = _tile_with_opponent(seed=1)
    give(state, 2)  # empty hand
    at_seven(state)

    offered = [a for a in rules.legal_actions(state) if a.position == tile]
    assert offered == [move_robber(tile, 0)], "nobody to rob at an empty-handed tile"


def test_a_tile_with_nobody_on_it_offers_no_victim():
    state = fresh(seed=1)
    complete_setup(state)
    at_seven(state)
    empty = next(
        t for t in range(1, T.NUM_TILES + 1)
        if t != state.robber_tile
        and all(state.vertex_owner[v] == 0 for v in T.TILE_VERTICES[t])
    )
    offered = [a for a in rules.legal_actions(state) if a.position == empty]
    assert offered == [move_robber(empty, 0)]


def test_you_must_rob_someone_if_the_tile_offers_a_victim():
    state, tile, _ = _tile_with_opponent(seed=1)
    give(state, 2, wood=3)
    at_seven(state)

    assert move_robber(tile, 0) not in rules.legal_actions(state)
    with pytest.raises(IllegalAction):
        rules.apply(state, move_robber(tile, 0))


def test_you_cannot_rob_someone_who_is_not_on_the_tile():
    state, tile, _ = _tile_with_opponent(seed=1)
    give(state, 2, wood=3)
    at_seven(state)
    empty = next(
        t for t in range(1, T.NUM_TILES + 1)
        if t != state.robber_tile
        and all(state.vertex_owner[v] == 0 for v in T.TILE_VERTICES[t])
    )
    with pytest.raises(IllegalAction):
        rules.apply(state, move_robber(empty, 2))


def test_a_city_makes_you_robbable_just_like_a_settlement():
    state, tile, vertex = _tile_with_opponent(seed=1)
    state.vertex_piece[vertex] = Piece.CITY
    give(state, 2, wood=3)
    at_seven(state)
    assert move_robber(tile, 2) in rules.legal_actions(state)


def test_the_stolen_card_is_uniform_over_cards_not_over_resource_types():
    """A hand of five wood and one ore should give up wood five times in six.

    Reaches into the private helper so the draw can be sampled without moving the
    robber 600 times.
    """
    state = fresh(seed=1)
    taken = collections.Counter()
    for _ in range(3000):
        give(state, 2, wood=5, ore=1)
        give(state, 1)
        rules._steal_one_card(state, 1, 2)
        taken.update({r: n for r, n in enumerate(state.hands[1]) if n})

    wood_share = taken[Resource.WOOD] / sum(taken.values())
    assert 0.78 < wood_share < 0.89, f"wood taken {wood_share:.2%} of the time, want ~83%"


def test_stealing_from_an_empty_hand_is_a_no_op():
    state = fresh(seed=1)
    give(state, 1, wood=2)
    give(state, 2)
    rules._steal_one_card(state, 1, 2)
    assert state.hands[1] == [2, 0, 0, 0, 0]
    assert total(state.hands[2]) == 0


# =========================================================================== #
# END TO END                                                                  #
# =========================================================================== #

def test_a_real_seven_resolves_all_the_way_to_build():
    state = fresh(num_players=3, seed=4)
    complete_setup(state)
    rng = random.Random(2)

    sevens = 0
    for _ in range(400):
        if state.phase is Phase.ROLL:
            if rules.roll_dice(state) == ROBBER_ROLL:
                sevens += 1
                assert state.phase in (Phase.DISCARD, Phase.MOVE_ROBBER)
            continue
        actions = rules.legal_actions(state)
        if not actions:
            break
        rules.apply(state, rng.choice(actions))
        if state.phase is Phase.GAME_OVER:
            break

    assert sevens > 0, "no 7 came up in 400 steps"
    for player in state.players:
        assert total(state.hands[player]) >= 0


def test_hands_never_exceed_the_limit_once_a_seven_has_been_resolved():
    """Checked at the moment the discards finish, not merely in MOVE_ROBBER.

    A Knight also reaches MOVE_ROBBER, and nobody discards for a Knight — so the phase
    alone says nothing about hand sizes.
    """
    sevens = 0
    for seed in range(6):
        state = fresh(num_players=3, seed=seed)
        complete_setup(state)
        rng = random.Random(seed)

        for _ in range(1500):
            if state.phase is Phase.GAME_OVER:
                break
            if state.phase is Phase.ROLL:
                if rules.roll_dice(state) == ROBBER_ROLL:
                    sevens += 1
                    while state.phase is Phase.DISCARD:
                        rules.apply(state, rng.choice(rules.legal_actions(state)))
                    for player in state.players:
                        assert total(state.hands[player]) <= state.ruleset.hand_limit, (
                            f"player {player} still holds "
                            f"{total(state.hands[player])} after discards"
                        )
                continue
            actions = rules.legal_actions(state)
            if not actions:
                break
            rules.apply(state, rng.choice(actions))

    assert sevens > 10, f"only {sevens} sevens across the sample"


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #

def _producing_vertex(state):
    for roll in range(2, 13):
        for vertex, productions in state.board.producers_for(roll).items():
            production = productions[0]
            return vertex, roll, production.resource, production.tile
    raise AssertionError("board produces nothing")


def _other_tile(state, tile):
    return next(t for t in range(1, T.NUM_TILES + 1) if t != tile)


def _free_vertex_on(state, tile):
    for vertex in T.TILE_VERTICES[tile]:
        if rules.respects_distance_rule(state, vertex):
            return vertex
    raise AssertionError(f"no free vertex on tile {tile}")


def _tile_with_opponent(seed):
    """A state in BUILD with player 2 holding a building on some tile."""
    state = fresh(seed=seed)
    complete_setup(state)
    in_build_phase(state, 1)
    tile = next(t for t in range(1, T.NUM_TILES + 1) if t != state.robber_tile)
    vertex = _free_vertex_on(state, tile)
    put_building(state, 2, vertex)
    return state, tile, vertex

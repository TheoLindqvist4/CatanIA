"""catan.state: the ownership model, turn order, and cloning."""

import random

import pytest

import catan.topology as T
from catan import rules
from catan.state import (
    MAX_CITIES,
    MAX_ROADS,
    MAX_SETTLEMENTS,
    NO_OWNER,
    GameState,
    Phase,
    Piece,
)
from helpers import (
    complete_setup,
    fresh,
    give,
    put_building,
    put_road,
    roll_sequence,
)


# --------------------------------------------------------------------------- #
# Ownership replaces the "available set" model                                #
# --------------------------------------------------------------------------- #

def test_a_fresh_state_has_no_owners():
    state = fresh()
    assert all(state.vertex_owner[v] == NO_OWNER for v in range(1, T.NUM_VERTICES + 1))
    assert all(state.vertex_piece[v] is Piece.NONE for v in range(1, T.NUM_VERTICES + 1))
    assert all(state.edge_owner[r] == NO_OWNER for r in range(1, T.NUM_ROADS + 1))


def test_empty_blocked_and_occupied_are_now_distinguishable():
    """The old model collapsed these three into one bit, which is why ownership,
    road-blocking and city upgrades were all impossible."""
    state = fresh()
    put_building(state, 1, 20)
    neighbour = T.VERTEX_NEIGHBOURS[20][0]
    far = 31

    # occupied: has an owner
    assert state.vertex_owner[20] == 1
    # blocked by the distance rule: no owner, but not buildable
    assert state.vertex_owner[neighbour] == NO_OWNER
    assert not rules.respects_distance_rule(state, neighbour)
    # empty: no owner and buildable
    assert state.vertex_owner[far] == NO_OWNER
    assert rules.respects_distance_rule(state, far)


def test_a_settlement_and_a_city_are_told_apart():
    state = fresh()
    put_building(state, 1, 20, Piece.SETTLEMENT)
    put_building(state, 1, 31, Piece.CITY)
    assert state.vertex_piece[20] is Piece.SETTLEMENT
    assert state.vertex_piece[31] is Piece.CITY
    assert rules.victory_points(state, 1) == 3


def test_buildings_of_and_roads_of_report_ownership():
    state = fresh()
    put_building(state, 1, 20)
    put_building(state, 2, 31)
    put_road(state, 1, 30)
    put_road(state, 2, 40)
    assert state.buildings_of(1) == (20,)
    assert state.buildings_of(2) == (31,)
    assert state.roads_of(1) == (30,)
    assert state.roads_of(2) == (40,)


# --------------------------------------------------------------------------- #
# Piece supplies                                                             #
# --------------------------------------------------------------------------- #

def test_starting_supplies_match_the_rulebook():
    state = fresh(num_players=4)
    for player in state.players:
        assert state.settlements_left[player] == MAX_SETTLEMENTS == 5
        assert state.cities_left[player] == MAX_CITIES == 4
        assert state.roads_left[player] == MAX_ROADS == 15


def test_supplies_are_per_player_not_shared():
    state = fresh(num_players=3)
    state.roads_left[1] -= 5
    assert state.roads_left[2] == MAX_ROADS
    state.hands[1][0] = 7
    assert state.hands[2][0] == 0


# --------------------------------------------------------------------------- #
# Players and turn order                                                     #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("count", [2, 3, 4])
def test_supported_player_counts(count):
    state = fresh(num_players=count)
    assert list(state.players) == list(range(1, count + 1))
    assert sorted(state.player_order) == list(range(1, count + 1))


@pytest.mark.parametrize("count", [0, 1, 5, 10])
def test_unsupported_player_counts_raise(count):
    with pytest.raises(ValueError):
        fresh(num_players=count)


def test_a_bad_player_order_raises():
    with pytest.raises(ValueError):
        GameState(num_players=3, seed=0, player_order=[1, 1, 2])


def test_setup_order_is_a_snake():
    state = fresh(num_players=3, player_order=[2, 3, 1])
    assert state.setup_sequence == [2, 3, 1, 1, 3, 2]


def test_setup_round_reports_one_then_two():
    state = fresh(num_players=2)
    assert state.setup_round == 1
    state.setup_step = 1
    assert state.setup_round == 1
    state.setup_step = 2
    assert state.setup_round == 2
    state.phase = Phase.BUILD
    assert state.setup_round is None


def test_turn_order_cycles_and_generalises_beyond_two_players():
    state = fresh(num_players=3, player_order=[3, 1, 2])
    state.phase = Phase.BUILD
    seen = []
    for turn in range(6):
        state.turn_number = turn
        seen.append(state.current_player)
    assert seen == [3, 1, 2, 3, 1, 2]


def test_randomize_order_is_a_permutation_and_uses_the_injected_rng():
    state = fresh(num_players=4, seed=7)
    order = state.randomize_order()
    assert sorted(order) == [1, 2, 3, 4]

    twin = fresh(num_players=4, seed=7)
    assert twin.randomize_order() == order


def test_state_does_not_consume_the_global_random_stream():
    random.seed(21)
    expected = [random.random() for _ in range(3)]
    random.seed(21)
    state = fresh(seed=5)
    state.randomize_order()
    complete_setup(state)
    assert [random.random() for _ in range(3)] == expected


# --------------------------------------------------------------------------- #
# Cloning                                                                    #
# --------------------------------------------------------------------------- #

def test_a_clone_equals_its_original():
    state = fresh(seed=4)
    complete_setup(state)
    assert state.clone() == state


def test_a_clone_shares_the_immutable_board_but_nothing_mutable():
    state = fresh(seed=4)
    clone = state.clone()

    assert clone.board is state.board, "the board is immutable and should be shared"

    for name in ("vertex_owner", "vertex_piece", "edge_owner", "player_order",
                 "settlements_left", "cities_left", "roads_left"):
        assert getattr(clone, name) is not getattr(state, name), f"{name} is shared"
    assert clone.hands is not state.hands
    for player in state.players:
        assert clone.hands[player] is not state.hands[player]


def test_mutating_a_clone_does_not_touch_the_original():
    state = fresh(seed=4)
    complete_setup(state)
    clone = state.clone()

    put_building(clone, 1, next(
        v for v in range(1, T.NUM_VERTICES + 1)
        if rules.respects_distance_rule(clone, v)
    ))
    give(clone, 1, wood=9)
    clone.turn_number += 3

    assert clone != state
    assert state.buildings_of(1) != clone.buildings_of(1)
    assert state.hands[1] != clone.hands[1]


def test_a_clone_replays_identically_by_default():
    """The default clone snapshots the RNG, so it is a true point-in-time copy."""
    state = fresh(seed=4)
    complete_setup(state)
    assert roll_sequence(state.clone(), 20) == roll_sequence(state.clone(), 20)


def test_a_clone_can_share_a_stream_so_rollouts_diverge():
    """Search wants divergent rollouts, not repeated ones."""
    state = fresh(seed=4)
    complete_setup(state)

    a, b = state.clone(rng=state.rng), state.clone(rng=state.rng)
    assert a.rng is b.rng is state.rng
    assert roll_sequence(a, 20) != roll_sequence(b, 20)


def test_equality_ignores_the_random_stream():
    """Two identical positions reached by different draws are the same position."""
    a = fresh(seed=1)
    b = a.clone()
    b.rng = random.Random(999)
    assert a == b


def test_equality_requires_the_same_board_layout():
    a, b = fresh(seed=1), fresh(seed=2)
    assert a.board.layout != b.board.layout, "seeds should differ; pick others"
    assert a != b


def test_boards_compare_by_layout_not_identity():
    """Replaying a seed builds an equal board in a new object, and that is the same
    game — so equality must not demand the same board object."""
    a, b = fresh(seed=11), fresh(seed=11)
    assert a.board is not b.board
    assert a.board == b.board
    assert hash(a.board) == hash(b.board)
    assert a == b


def test_cloning_is_cheap_enough_for_search():
    """MCTS clones constantly; this must stay far below a millisecond."""
    import timeit
    state = fresh(seed=4)
    complete_setup(state)
    elapsed = timeit.timeit(state.clone, number=20_000) / 20_000
    assert elapsed < 100e-6, f"{elapsed * 1e6:.1f} us/clone is too slow for search"


def test_sharing_the_stream_makes_cloning_much_cheaper():
    """Snapshotting a Mersenne Twister copies 625 words and dominates clone cost, so
    search should share the stream. Documented on GameState.clone."""
    import timeit
    state = fresh(seed=4)
    complete_setup(state)

    snapshot = timeit.timeit(state.clone, number=20_000) / 20_000
    shared = timeit.timeit(lambda: state.clone(rng=state.rng), number=20_000) / 20_000
    assert shared < snapshot / 3, (
        f"shared-stream clone {shared * 1e6:.1f} us vs snapshot {snapshot * 1e6:.1f} us "
        "— expected a large saving"
    )
    assert shared < 10e-6

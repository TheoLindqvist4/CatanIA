"""catan.board: layout generation, the production index, and immutability."""

import collections
import random

import pytest

import catan.topology as T
from catan.board import ROBBER_ROLL, Board, Production
from catan.resources import DESERT, Resource


def make_board(seed=0):
    return Board(rng=random.Random(seed))


# --------------------------------------------------------------------------- #
# I/O-freedom and determinism                                                 #
# --------------------------------------------------------------------------- #

def test_constructing_a_board_prints_nothing(capsys):
    make_board()
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_render_returns_a_string_instead_of_printing(capsys):
    board = make_board()
    text = board.render()
    assert isinstance(text, str)
    assert capsys.readouterr().out == ""
    assert len(text.splitlines()) == len(T.ROW_LENGTHS)


def test_same_seed_gives_an_identical_board():
    a, b = make_board(1234), make_board(1234)
    assert a.tile_numbers == b.tile_numbers
    assert a.tile_resources == b.tile_resources


def test_board_does_not_consume_the_global_random_stream():
    random.seed(99)
    expected = [random.random() for _ in range(3)]
    random.seed(99)
    make_board(7)
    make_board(8)
    assert [random.random() for _ in range(3)] == expected


# --------------------------------------------------------------------------- #
# Layout invariants                                                           #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def many_boards():
    return [make_board(seed) for seed in range(60)]


def test_number_tokens_are_the_standard_multiset(many_boards):
    for board in many_boards:
        assert sorted(board.tile_numbers[1:]) == sorted(Board.NUMBERS)


def test_resource_counts_are_standard(many_boards):
    for board in many_boards:
        assert collections.Counter(board.tile_resources[1:]) == \
            collections.Counter(Board.TILE_COUNTS)


def test_resources_are_the_enum_not_strings(many_boards):
    """Names were normalised in Phase 1; 'Weat' is gone."""
    for board in many_boards:
        for tile in range(1, T.NUM_TILES + 1):
            resource = board.resource_at(tile)
            assert resource is DESERT or resource in list(Resource)


def test_the_desert_is_always_the_seven_tile(many_boards):
    for board in many_boards:
        for tile in range(1, T.NUM_TILES + 1):
            assert (board.number_at(tile) == ROBBER_ROLL) == \
                   (board.resource_at(tile) is DESERT)
        assert board.number_at(board.desert_tile) == ROBBER_ROLL


def test_balanced_generation_rule_holds(many_boards):
    for board in many_boards:
        for tile in range(1, T.NUM_TILES + 1):
            a = board.number_at(tile)
            for other in T.TILE_ADJACENCY[tile]:
                b = board.number_at(other)
                assert a != b
                assert frozenset((a, b)) not in Board.UNBALANCED_PAIRS


def test_generation_failure_is_bounded_not_a_hang():
    with pytest.raises(RuntimeError):
        Board(rng=random.Random(0), max_generation_attempts=0)


def test_the_row_view_reassembles_the_flat_arrays():
    board = make_board(8)
    numbers = [n for numbers, _ in board.rows for n in numbers]
    resources = [r for _, resources in board.rows for r in resources]
    assert numbers == board.tile_numbers[1:]
    assert resources == board.tile_resources[1:]


# --------------------------------------------------------------------------- #
# Production index                                                            #
# --------------------------------------------------------------------------- #

def test_production_at_covers_every_tile_touching_a_vertex():
    board = make_board(3)
    for vertex in range(1, T.NUM_VERTICES + 1):
        assert board.production_at(vertex) == tuple(
            Production(tile, board.number_at(tile), board.resource_at(tile))
            for tile in T.VERTEX_TILES[vertex]
        )


def test_producers_for_returns_exactly_what_should_pay_out():
    board = make_board(5)
    for roll in range(2, 13):
        expected = {
            vertex: tuple(
                Production(t, board.number_at(t), board.resource_at(t))
                for t in T.VERTEX_TILES[vertex]
                if board.number_at(t) == roll and board.resource_at(t) is not DESERT
            )
            for vertex in range(1, T.NUM_VERTICES + 1)
        }
        assert board.producers_for(roll) == {v: p for v, p in expected.items() if p}


def test_rolling_a_seven_produces_nothing():
    """Structural: the desert is excluded from the index, not filtered by callers."""
    for seed in range(20):
        assert make_board(seed).producers_for(ROBBER_ROLL) == {}


@pytest.mark.parametrize("bad", [1, 13, 0, -1])
def test_producers_for_rejects_impossible_rolls(bad):
    with pytest.raises(ValueError):
        make_board().producers_for(bad)


def test_resources_at_excludes_the_desert():
    """Used by the setup-round-2 payout, which must not hand out desert."""
    board = make_board(4)
    for vertex in range(1, T.NUM_VERTICES + 1):
        collected = board.resources_at(vertex)
        assert all(r is not DESERT for r in collected)
        assert len(collected) == sum(
            1 for p in board.production_at(vertex) if p.resource is not DESERT
        )


def test_the_balanced_rule_makes_double_production_impossible():
    """No vertex can collect twice from one roll: the tiles meeting at a vertex are
    pairwise adjacent, and equal adjacent numbers are forbidden. The "double 6" spot
    of official Catan does not exist here —
    docs/decisions/0005-balanced-board-generation.md."""
    for seed in range(60):
        board = make_board(seed)
        for roll in range(2, 13):
            for productions in board.producers_for(roll).values():
                assert len(productions) == 1


# --------------------------------------------------------------------------- #
# Immutability contract                                                       #
# --------------------------------------------------------------------------- #

def test_the_board_carries_no_ownership_or_robber_state():
    """Occupancy and the robber belong to GameState, so clones can share the board —
    docs/decisions/0009-immutable-board-mutable-state.md."""
    board = make_board()
    forbidden = ("settlement_positions", "road_positions", "vertex_owner",
                 "edge_owner", "robber_tile")
    for name in forbidden:
        assert not hasattr(board, name), f"Board should not own {name}"


def test_a_board_is_unchanged_by_a_full_game():
    """If a game mutated the board, sharing it across clones would corrupt search."""
    from helpers import drive, snapshot_board
    from catan.state import GameState

    state = GameState(num_players=3, seed=3)
    before = snapshot_board(state.board)
    drive(state, random.Random(11), max_actions=3000)
    assert snapshot_board(state.board) == before

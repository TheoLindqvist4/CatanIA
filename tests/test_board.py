"""Board layout invariants, determinism, and I/O-freedom."""

import collections
import random

import pytest

import catan.topology as T
from Board import DESERT, Board, Production


def make_board(seed=0):
    return Board(rng=random.Random(seed))


# --------------------------------------------------------------------------- #
# The core must not touch stdout                                              #
# --------------------------------------------------------------------------- #

def test_constructing_a_board_prints_nothing(capsys):
    """A training loop builds millions of boards; none may write to stdout."""
    make_board()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_display_board_returns_a_string_instead_of_printing(capsys):
    board = make_board()
    rendering = board.display_board()
    assert isinstance(rendering, str)
    assert capsys.readouterr().out == ""
    assert len(rendering.splitlines()) == len(T.ROW_LENGTHS)


def test_importing_the_game_module_does_not_play_a_game(capsys):
    import importlib

    import Game_2_players
    importlib.reload(Game_2_players)
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------- #
# Determinism                                                                 #
# --------------------------------------------------------------------------- #

def test_same_seed_gives_an_identical_board():
    a, b = make_board(1234), make_board(1234)
    assert a.tile_numbers == b.tile_numbers
    assert a.tile_resources == b.tile_resources
    assert a.vertex_production == b.vertex_production


def test_different_seeds_give_different_boards():
    layouts = {
        (tuple(make_board(seed).tile_numbers), tuple(make_board(seed).tile_resources))
        for seed in range(12)
    }
    assert len(layouts) > 1


def test_board_does_not_consume_the_global_random_stream():
    """Using the global module would couple unrelated games together."""
    random.seed(99)
    expected = [random.random() for _ in range(3)]

    random.seed(99)
    make_board(7)
    make_board(8)
    assert [random.random() for _ in range(3)] == expected


def test_pinned_layout_for_a_fixed_seed():
    """Canary against unintended changes to the generation sequence.

    Not a rule — if generation is deliberately changed (e.g. adding the official
    spiral option) update these values in the same commit.
    """
    board = make_board(42)
    assert board.tile_numbers[1:] == [10, 6, 4, 8, 11, 3, 8, 9, 5, 12,
                                      4, 3, 2, 7, 10, 6, 9, 5, 11]
    assert board.tile_resources[1:] == [
        'Wood', 'Sheep', 'Wood', 'Weat', 'Ore', 'Brick', 'Wood', 'Weat', 'Ore',
        'Sheep', 'Sheep', 'Ore', 'Brick', DESERT, 'Weat', 'Sheep', 'Wood',
        'Brick', 'Weat',
    ]


# --------------------------------------------------------------------------- #
# Layout invariants, over many generated boards                               #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def many_boards():
    return [make_board(seed) for seed in range(60)]


def test_every_tile_gets_a_number_and_a_resource(many_boards):
    for board in many_boards:
        assert board.tile_numbers[0] is None and board.tile_resources[0] is None
        assert all(board.tile_numbers[t] is not None for t in range(1, T.NUM_TILES + 1))
        assert all(board.tile_resources[t] is not None for t in range(1, T.NUM_TILES + 1))


def test_number_tokens_are_the_standard_multiset(many_boards):
    for board in many_boards:
        assert sorted(board.tile_numbers[1:]) == sorted(Board.NUMBERS)


def test_resource_counts_are_standard(many_boards):
    for board in many_boards:
        assert collections.Counter(board.tile_resources[1:]) == \
            collections.Counter(Board.TILE_COUNTS)


def test_the_desert_is_always_the_seven_tile(many_boards):
    for board in many_boards:
        for tile in range(1, T.NUM_TILES + 1):
            assert (board.number_at(tile) == 7) == (board.resource_at(tile) == DESERT)
        assert board.number_at(board.desert_tile) == 7


def test_balanced_generation_rule_holds(many_boards):
    """No equal adjacent numbers, and no 6/8 or 2/12 pairs. A house rule."""
    for board in many_boards:
        for tile in range(1, T.NUM_TILES + 1):
            a = board.number_at(tile)
            for other in T.TILE_ADJACENCY[tile]:
                b = board.number_at(other)
                assert a != b
                assert frozenset((a, b)) not in Board.UNBALANCED_PAIRS


def test_numbers_around_matches_tile_adjacency():
    board = make_board(3)
    for tile in range(1, T.NUM_TILES + 1):
        assert sorted(board.numbers_around(tile)) == sorted(
            board.number_at(other) for other in T.TILE_ADJACENCY[tile]
        )


def test_is_number_valid_rejects_a_duplicate_neighbour():
    board = make_board(3)
    tile = 5
    neighbour_number = board.number_at(T.TILE_ADJACENCY[tile][0])
    assert not board.is_number_valid(tile, neighbour_number)


def test_generation_failure_is_bounded_not_a_hang():
    with pytest.raises(RuntimeError):
        Board(rng=random.Random(0), max_generation_attempts=0)


# --------------------------------------------------------------------------- #
# Flat arrays vs the ragged-row view                                          #
# --------------------------------------------------------------------------- #

def test_the_row_view_reassembles_the_flat_arrays():
    board = make_board(8)
    assert [len(row) for row in board.grid] == list(T.ROW_LENGTHS)
    assert [len(row) for row in board.tile_grid] == list(T.ROW_LENGTHS)
    assert [n for row in board.grid for n in row] == board.tile_numbers[1:]
    assert [r for row in board.tile_grid for r in row] == board.tile_resources[1:]


def test_the_row_view_agrees_with_tile_ids():
    board = make_board(8)
    for tile in range(1, T.NUM_TILES + 1):
        row, col = T.tile_rowcol(tile)
        assert board.grid[row][col] == board.number_at(tile)
        assert board.tile_grid[row][col] == board.resource_at(tile)


def test_the_row_view_is_a_copy_not_internal_state():
    board = make_board()
    rows = board.grid
    rows[0][0] = 999
    assert board.grid[0][0] != 999


# --------------------------------------------------------------------------- #
# Production                                                                  #
# --------------------------------------------------------------------------- #

def test_vertex_production_covers_every_vertex():
    board = make_board()
    assert set(board.vertex_production) == set(range(1, T.NUM_VERTICES + 1))
    for vertex, productions in board.vertex_production.items():
        assert len(productions) == len(T.VERTEX_TILES[vertex])
        assert all(isinstance(p, Production) for p in productions)


def test_vertex_production_matches_the_tiles_touching_each_vertex():
    board = make_board(3)
    for vertex in range(1, T.NUM_VERTICES + 1):
        assert board.vertex_production[vertex] == tuple(
            Production(tile, board.number_at(tile), board.resource_at(tile))
            for tile in T.VERTEX_TILES[vertex]
        )


def test_producers_for_returns_exactly_what_should_pay_out():
    board = make_board(5)
    for roll in range(2, 13):
        producers = board.producers_for(roll)
        expected = {
            vertex: tuple(
                Production(t, board.number_at(t), board.resource_at(t))
                for t in T.VERTEX_TILES[vertex]
                if board.number_at(t) == roll and board.resource_at(t) != DESERT
            )
            for vertex in range(1, T.NUM_VERTICES + 1)
        }
        expected = {v: p for v, p in expected.items() if p}
        assert producers == expected


def test_all_tiles_meeting_at_a_vertex_are_pairwise_adjacent():
    """A geometric fact with a large strategic consequence — see the test below."""
    import itertools
    for vertex in range(1, T.NUM_VERTICES + 1):
        for a, b in itertools.combinations(T.VERTEX_TILES[vertex], 2):
            assert b in T.TILE_ADJACENCY[a]


def test_the_balanced_rule_makes_double_production_impossible():
    """No vertex can ever collect twice from one roll.

    The tiles meeting at a vertex are pairwise adjacent, and the balanced-board rule
    forbids equal numbers on adjacent tiles. So the "double 6" spot that exists in
    official Catan cannot occur here at all — a real strategic difference the agent
    will learn, not an implementation detail. Documented in
    docs/decisions/0005-balanced-board-generation.md.

    If the official spiral layout is added as an option, this must become
    conditional on the generation mode.
    """
    for seed in range(120):
        board = make_board(seed)
        for roll in range(2, 13):
            for productions in board.producers_for(roll).values():
                assert len(productions) == 1


def test_rolling_a_seven_produces_nothing():
    """Structural, not a caller-side filter: the desert is excluded from the index."""
    for seed in range(20):
        assert make_board(seed).producers_for(7) == {}


@pytest.mark.parametrize("bad", [1, 13, 0, -1])
def test_producers_for_rejects_impossible_rolls(bad):
    with pytest.raises(ValueError):
        make_board().producers_for(bad)


def test_desert_never_appears_in_any_payout():
    board = make_board(4)
    for roll in range(2, 13):
        for productions in board.producers_for(roll).values():
            assert all(p.resource != DESERT for p in productions)


# --------------------------------------------------------------------------- #
# Availability and consumption                                                #
# --------------------------------------------------------------------------- #

def test_a_fresh_board_has_everything_available():
    board = make_board()
    assert board.get_available_settlements() == list(range(1, T.NUM_VERTICES + 1))
    assert board.get_available_road() == list(range(1, T.NUM_ROADS + 1))


def test_placing_a_settlement_blocks_it_and_its_neighbours():
    board = make_board()
    board.delete_settlement_position(20)
    assert not board.is_settlement_position_available(20)
    for neighbour in T.VERTEX_NEIGHBOURS[20]:
        assert not board.is_settlement_position_available(neighbour)
    assert board.is_settlement_position_available(31)  # two steps away


def test_deleting_is_idempotent():
    board = make_board()
    board.delete_settlement_position(20)
    remaining = board.get_available_settlements()
    board.delete_settlement_position(20)
    assert board.get_available_settlements() == remaining

    board.delete_road_position(30)
    roads = board.get_available_road()
    board.delete_road_position(30)
    assert board.get_available_road() == roads


def test_getters_return_copies_not_internal_state():
    board = make_board()
    settlements = board.get_available_settlements()
    settlements.clear()
    assert board.get_available_settlements() == list(range(1, T.NUM_VERTICES + 1))


def test_available_roads_from_settlement_excludes_taken_roads():
    """Unfiltered, this let the setup phase offer roads that were already built."""
    board = make_board()
    roads = board.get_adjacent_roads_from_settlement(20)
    assert set(board.get_available_road_from_settlement(20)) == set(roads)

    board.delete_road_position(roads[0])
    assert set(board.get_available_road_from_settlement(20)) == set(roads[1:])


# --------------------------------------------------------------------------- #
# Adjacency wrappers raise instead of returning a sentinel string             #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("method,bad", [
    ("get_adjacents_for_positions", 55),
    ("get_adjacents_for_positions", 0),
    ("get_adjacent_roads_from_settlement", 55),
    ("get_adjacent_roads_from_road", 73),
    ("get_adjacent_settlement_from_road", 0),
    ("get_tiles_for_position", 55),
    ("get_adjacent_tiles", 20),
    ("get_roads_of_tile", 20),
    ("number_at", 20),
    ("resource_at", 0),
])
def test_out_of_range_ids_raise(method, bad):
    board = make_board()
    with pytest.raises(ValueError):
        getattr(board, method)(bad)


def test_adjacency_never_returns_a_string():
    """The old implementation returned an error message, which callers iterated."""
    board = make_board()
    for vertex in range(1, T.NUM_VERTICES + 1):
        assert isinstance(board.get_adjacents_for_positions(vertex), tuple)
    for road in range(1, T.NUM_ROADS + 1):
        assert isinstance(board.get_adjacent_roads_from_road(road), tuple)


# --------------------------------------------------------------------------- #
# Performance                                                                 #
# --------------------------------------------------------------------------- #

def test_adjacency_lookup_is_allocation_free():
    """Guards the ~145x speedup that makes RL-scale rollouts feasible.

    The hand-written version rebuilt a 72-entry dict on every call (~5.5 us). A
    generous ceiling of 1 us/call still fails loudly if that regresses.
    """
    import timeit
    board = make_board()
    elapsed = timeit.timeit(lambda: board.get_adjacent_roads_from_road(12),
                            number=200_000)
    per_call_us = elapsed / 200_000 * 1e6
    assert per_call_us < 1.0, f"{per_call_us:.2f} us/call — adjacency lookup regressed"


def test_dice_payout_is_an_index_lookup_not_a_board_scan():
    """producers_for must not walk all 54 vertices on every roll."""
    import timeit
    board = make_board()
    elapsed = timeit.timeit(lambda: board.producers_for(8), number=200_000)
    per_call_us = elapsed / 200_000 * 1e6
    assert per_call_us < 1.0, f"{per_call_us:.2f} us/call — payout lookup regressed"

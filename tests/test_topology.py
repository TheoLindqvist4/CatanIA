"""Invariants on the generated board geometry.

The geometry is computed from :data:`topology.ROW_LENGTHS` alone. These tests check it
against independently-stated ground truth — most importantly the ids drawn in
``Images/``, which are what make the board comprehensible to a human and must not
drift when the generator changes.
"""

import collections
import itertools

import pytest

import catan.topology as T


# --------------------------------------------------------------------------- #
# The numbering drawn in Images/ is the contract                              #
# --------------------------------------------------------------------------- #

# Transcribed from Images/Catan_settlement_positions.png: each tile's six corners.
DRAWN_TILE_VERTICES = {
    1: (1, 4, 5, 8, 9, 13), 2: (2, 5, 6, 9, 10, 14), 3: (3, 6, 7, 10, 11, 15),
    4: (8, 12, 13, 17, 18, 23), 5: (9, 13, 14, 18, 19, 24), 6: (10, 14, 15, 19, 20, 25),
    7: (11, 15, 16, 20, 21, 26), 8: (17, 22, 23, 28, 29, 34), 9: (18, 23, 24, 29, 30, 35),
    10: (19, 24, 25, 30, 31, 36), 11: (20, 25, 26, 31, 32, 37),
    12: (21, 26, 27, 32, 33, 38), 13: (29, 34, 35, 39, 40, 44),
    14: (30, 35, 36, 40, 41, 45), 15: (31, 36, 37, 41, 42, 46),
    16: (32, 37, 38, 42, 43, 47), 17: (40, 44, 45, 48, 49, 52),
    18: (41, 45, 46, 49, 50, 53), 19: (42, 46, 47, 50, 51, 54),
}

# Transcribed from Images/Catan_road_positions.png: each road's two endpoints.
DRAWN_ROAD_VERTICES = {
    1: (1, 4), 2: (1, 5), 3: (2, 5), 4: (2, 6), 5: (3, 6), 6: (3, 7),
    7: (4, 8), 8: (5, 9), 9: (6, 10), 10: (7, 11),
    11: (8, 12), 12: (8, 13), 13: (9, 13), 14: (9, 14), 15: (10, 14),
    16: (10, 15), 17: (11, 15), 18: (11, 16),
    19: (12, 17), 20: (13, 18), 21: (14, 19), 22: (15, 20), 23: (16, 21),
    24: (17, 22), 25: (17, 23), 26: (18, 23), 27: (18, 24), 28: (19, 24),
    29: (19, 25), 30: (20, 25), 31: (20, 26), 32: (21, 26), 33: (21, 27),
    34: (22, 28), 35: (23, 29), 36: (24, 30), 37: (25, 31), 38: (26, 32), 39: (27, 33),
    40: (28, 34), 41: (29, 34), 42: (29, 35), 43: (30, 35), 44: (30, 36),
    45: (31, 36), 46: (31, 37), 47: (32, 37), 48: (32, 38), 49: (33, 38),
    50: (34, 39), 51: (35, 40), 52: (36, 41), 53: (37, 42), 54: (38, 43),
    55: (39, 44), 56: (40, 44), 57: (40, 45), 58: (41, 45),
    59: (41, 46), 60: (42, 46), 61: (42, 47), 62: (43, 47),
    63: (44, 48), 64: (45, 49), 65: (46, 50), 66: (47, 51),
    67: (48, 52), 68: (49, 52), 69: (49, 53), 70: (50, 53), 71: (50, 54), 72: (51, 54),
}

# Tile neighbours, independently transcribed from Images/Catan_board.png.
DRAWN_TILE_ADJACENCY = {
    1: (2, 4, 5), 2: (1, 3, 5, 6), 3: (2, 6, 7), 4: (1, 5, 8, 9),
    5: (1, 2, 4, 6, 9, 10), 6: (2, 3, 5, 7, 10, 11), 7: (3, 6, 11, 12),
    8: (4, 9, 13), 9: (4, 5, 8, 10, 13, 14), 10: (5, 6, 9, 11, 14, 15),
    11: (6, 7, 10, 12, 15, 16), 12: (7, 11, 16), 13: (8, 9, 14, 17),
    14: (9, 10, 13, 15, 17, 18), 15: (10, 11, 14, 16, 18, 19),
    16: (11, 12, 15, 19), 17: (13, 14, 18), 18: (14, 15, 17, 19), 19: (15, 16, 18),
}


def test_generated_vertex_ids_match_the_drawings():
    """If this fails, Images/Catan_settlement_positions.png no longer describes the code."""
    assert {t: T.TILE_VERTICES[t] for t in DRAWN_TILE_VERTICES} == DRAWN_TILE_VERTICES


def test_generated_road_ids_match_the_drawings():
    """If this fails, Images/Catan_road_positions.png no longer describes the code."""
    assert {r: T.ROAD_VERTICES[r] for r in DRAWN_ROAD_VERTICES} == DRAWN_ROAD_VERTICES


def test_generated_tile_adjacency_matches_the_drawings():
    assert {t: T.TILE_ADJACENCY[t] for t in DRAWN_TILE_ADJACENCY} == DRAWN_TILE_ADJACENCY


def test_regression_roads_2_and_51_have_their_missing_neighbour():
    """The hand-written road->roads map omitted one neighbour from each of these.

    Road 2 = (1, 5) and road 3 = (2, 5) share vertex 5; road 51 = (35, 40) and
    road 43 = (30, 35) share vertex 35. Both omissions silently corrupted
    longest-road calculation and legal-move enumeration. Generating the relation
    makes the class of error impossible; this pins the two known cases.
    """
    assert T.ROAD_NEIGHBOURS[2] == (1, 3, 8)
    assert T.ROAD_NEIGHBOURS[51] == (42, 43, 56, 57)


# --------------------------------------------------------------------------- #
# Sizes and shapes                                                            #
# --------------------------------------------------------------------------- #

def test_counts():
    assert (T.NUM_TILES, T.NUM_VERTICES, T.NUM_ROADS) == (19, 54, 72)
    assert T.NUM_TILES == sum(T.ROW_LENGTHS)


def test_every_table_is_indexed_by_id_with_slot_zero_unused():
    for table, size in (
        (T.TILE_VERTICES, T.NUM_TILES),
        (T.TILE_ROADS, T.NUM_TILES),
        (T.TILE_ADJACENCY, T.NUM_TILES),
        (T.ROAD_VERTICES, T.NUM_ROADS),
        (T.ROAD_NEIGHBOURS, T.NUM_ROADS),
        (T.ROAD_TILES, T.NUM_ROADS),
        (T.VERTEX_TILES, T.NUM_VERTICES),
        (T.VERTEX_ROADS, T.NUM_VERTICES),
        (T.VERTEX_NEIGHBOURS, T.NUM_VERTICES),
    ):
        assert len(table) == size + 1
        assert table[0] == ()
        assert all(isinstance(entry, tuple) for entry in table)


def test_tables_are_immutable():
    with pytest.raises(TypeError):
        T.VERTEX_NEIGHBOURS[1] = (2, 3)


def test_entries_are_sorted():
    """Sorted entries keep legal-move enumeration deterministic."""
    for table in (T.TILE_VERTICES, T.TILE_ADJACENCY, T.ROAD_VERTICES, T.ROAD_TILES,
                  T.ROAD_NEIGHBOURS, T.VERTEX_TILES, T.VERTEX_ROADS,
                  T.VERTEX_NEIGHBOURS):
        assert all(list(entry) == sorted(entry) for entry in table)


# --------------------------------------------------------------------------- #
# The lattice                                                                 #
# --------------------------------------------------------------------------- #

def test_all_coordinates_are_integers():
    """Integer coordinates are what let corners be deduplicated by equality."""
    for table, size in ((T.TILE_XY, T.NUM_TILES), (T.VERTEX_XY, T.NUM_VERTICES)):
        for i in range(1, size + 1):
            assert all(isinstance(c, int) for c in table[i])


def test_vertex_coordinates_are_unique():
    coords = [T.VERTEX_XY[v] for v in range(1, T.NUM_VERTICES + 1)]
    assert len(set(coords)) == T.NUM_VERTICES


def test_vertex_ids_ascend_top_to_bottom_then_left_to_right():
    keys = [(T.VERTEX_XY[v][1], T.VERTEX_XY[v][0]) for v in range(1, T.NUM_VERTICES + 1)]
    assert keys == sorted(keys)


def test_vertex_rows_have_the_expected_hexagonal_profile():
    assert [len(row) for row in T.VERTEX_ROWS] == [3, 4, 4, 5, 5, 6, 6, 5, 5, 4, 4, 3]
    assert sum(len(row) for row in T.VERTEX_ROWS) == T.NUM_VERTICES
    # rows are contiguous id ranges
    assert list(itertools.chain.from_iterable(T.VERTEX_ROWS)) == \
        list(range(1, T.NUM_VERTICES + 1))


def test_road_bands_alternate_slanted_and_vertical():
    bands = [
        len(list(group))
        for _, group in itertools.groupby(
            range(1, T.NUM_ROADS + 1),
            key=lambda r: min(T.VERTEX_XY[v][1] for v in T.ROAD_VERTICES[r]),
        )
    ]
    assert bands == [6, 4, 8, 5, 10, 6, 10, 5, 8, 4, 6]
    assert sum(bands) == T.NUM_ROADS


def test_each_tile_centre_is_two_units_from_its_row_neighbour():
    for row, length in enumerate(T.ROW_LENGTHS):
        centres = [T.TILE_XY[T.tile_index(row, col)] for col in range(length)]
        assert all(y == T.ROW_PITCH * row for _, y in centres)
        xs = [x for x, _ in centres]
        assert all(b - a == 2 for a, b in zip(xs, xs[1:]))


def test_rows_are_centred_on_each_other():
    """Interlocking rows are what make the hexagon a hexagon."""
    widest = max(T.ROW_LENGTHS)
    for row, length in enumerate(T.ROW_LENGTHS):
        first = T.TILE_XY[T.tile_index(row, 0)][0]
        assert first == widest - length


def test_road_midpoints_are_twice_the_average_of_their_endpoints():
    for road in range(1, T.NUM_ROADS + 1):
        u, v = T.ROAD_VERTICES[road]
        assert T.ROAD_MIDPOINT_2X[road] == (
            T.VERTEX_XY[u][0] + T.VERTEX_XY[v][0],
            T.VERTEX_XY[u][1] + T.VERTEX_XY[v][1],
        )


# --------------------------------------------------------------------------- #
# Tiles and vertices                                                          #
# --------------------------------------------------------------------------- #

def test_each_tile_has_six_distinct_corners_and_six_distinct_roads():
    for tile in range(1, T.NUM_TILES + 1):
        assert len(set(T.TILE_VERTICES[tile])) == 6
        assert len(set(T.TILE_ROADS[tile])) == 6


def test_every_vertex_belongs_to_between_one_and_three_tiles():
    counts = collections.Counter(
        v for tile in range(1, T.NUM_TILES + 1) for v in T.TILE_VERTICES[tile]
    )
    assert set(counts) == set(range(1, T.NUM_VERTICES + 1))
    # 18 vertices touch 1 tile, 12 touch 2, 24 touch 3 -> 19 * 6 = 114 incidences
    assert dict(collections.Counter(counts.values())) == {1: 18, 2: 12, 3: 24}
    assert sum(counts.values()) == T.NUM_TILES * 6


def test_vertex_tiles_inverts_tile_vertices():
    for vertex in range(1, T.NUM_VERTICES + 1):
        assert T.VERTEX_TILES[vertex] == tuple(sorted(
            tile for tile in range(1, T.NUM_TILES + 1)
            if vertex in T.TILE_VERTICES[tile]
        ))


def test_road_tiles_inverts_tile_roads():
    for road in range(1, T.NUM_ROADS + 1):
        assert T.ROAD_TILES[road] == tuple(sorted(
            tile for tile in range(1, T.NUM_TILES + 1) if road in T.TILE_ROADS[tile]
        ))


def test_adjacent_tiles_share_exactly_two_corners_and_one_road():
    for tile in range(1, T.NUM_TILES + 1):
        for other in range(1, T.NUM_TILES + 1):
            if tile == other:
                continue
            shared_corners = set(T.TILE_VERTICES[tile]) & set(T.TILE_VERTICES[other])
            shared_roads = set(T.TILE_ROADS[tile]) & set(T.TILE_ROADS[other])
            if other in T.TILE_ADJACENCY[tile]:
                assert len(shared_corners) == 2
                assert len(shared_roads) == 1
            else:
                assert not shared_corners
                assert not shared_roads


def test_tile_adjacency_is_symmetric_and_bounded():
    for tile in range(1, T.NUM_TILES + 1):
        assert tile not in T.TILE_ADJACENCY[tile]
        assert len(T.TILE_ADJACENCY[tile]) <= 6
        for other in T.TILE_ADJACENCY[tile]:
            assert tile in T.TILE_ADJACENCY[other]


# --------------------------------------------------------------------------- #
# Roads                                                                       #
# --------------------------------------------------------------------------- #

def test_each_road_joins_two_distinct_adjacent_vertices():
    for road in range(1, T.NUM_ROADS + 1):
        u, v = T.ROAD_VERTICES[road]
        assert u < v
        assert v in T.VERTEX_NEIGHBOURS[u]
        assert u in T.VERTEX_NEIGHBOURS[v]


def test_roads_are_exactly_the_set_of_adjacent_vertex_pairs():
    edges = {frozenset(T.ROAD_VERTICES[r]) for r in range(1, T.NUM_ROADS + 1)}
    assert len(edges) == T.NUM_ROADS, "duplicate road definitions"
    assert edges == {
        frozenset((v, n))
        for v in range(1, T.NUM_VERTICES + 1)
        for n in T.VERTEX_NEIGHBOURS[v]
    }


def test_road_between_is_the_inverse_of_road_vertices():
    for road in range(1, T.NUM_ROADS + 1):
        u, v = T.ROAD_VERTICES[road]
        assert T.road_between(u, v) == road
        assert T.road_between(v, u) == road


def test_road_between_returns_none_for_non_adjacent_vertices():
    assert T.road_between(1, 54) is None
    assert T.road_between(1, 2) is None  # same vertex row, not joined by a road


def test_vertex_degrees_are_two_or_three():
    """No vertex above degree 3 is what makes "no repeated road" imply
    "no repeated vertex" during path search."""
    histogram = collections.Counter(
        len(T.VERTEX_NEIGHBOURS[v]) for v in range(1, T.NUM_VERTICES + 1)
    )
    assert dict(histogram) == {2: 18, 3: 36}
    assert sum(k * n for k, n in histogram.items()) // 2 == T.NUM_ROADS


def test_vertex_neighbours_is_symmetric():
    for vertex in range(1, T.NUM_VERTICES + 1):
        assert vertex not in T.VERTEX_NEIGHBOURS[vertex]
        for neighbour in T.VERTEX_NEIGHBOURS[vertex]:
            assert vertex in T.VERTEX_NEIGHBOURS[neighbour]


def test_vertex_roads_agrees_with_road_vertices():
    for vertex in range(1, T.NUM_VERTICES + 1):
        assert T.VERTEX_ROADS[vertex] == tuple(sorted(
            r for r in range(1, T.NUM_ROADS + 1) if vertex in T.ROAD_VERTICES[r]
        ))
        assert len(T.VERTEX_ROADS[vertex]) == len(T.VERTEX_NEIGHBOURS[vertex])


def test_road_neighbours_is_exactly_roads_sharing_an_endpoint():
    for road in range(1, T.NUM_ROADS + 1):
        assert T.ROAD_NEIGHBOURS[road] == tuple(sorted(
            other
            for other in range(1, T.NUM_ROADS + 1)
            if other != road
            and set(T.ROAD_VERTICES[other]) & set(T.ROAD_VERTICES[road])
        ))


def test_road_neighbours_is_symmetric():
    for road in range(1, T.NUM_ROADS + 1):
        assert road not in T.ROAD_NEIGHBOURS[road]
        for other in T.ROAD_NEIGHBOURS[road]:
            assert road in T.ROAD_NEIGHBOURS[other]


# --------------------------------------------------------------------------- #
# Coastline — harbours (Phase 2) depend on getting these three apart          #
# --------------------------------------------------------------------------- #

def test_coastal_roads_border_exactly_one_tile():
    assert all(len(T.ROAD_TILES[r]) == 1 for r in T.COASTAL_ROADS)
    assert all(len(T.ROAD_TILES[r]) == 2
               for r in range(1, T.NUM_ROADS + 1) if r not in T.COASTAL_ROADS)


def test_coastline_length_follows_from_edge_incidences():
    """6*19 incidences = 2*interior + coastal, and interior + coastal = 72."""
    coastal = len(T.COASTAL_ROADS)
    interior = T.NUM_ROADS - coastal
    assert 2 * interior + coastal == 6 * T.NUM_TILES
    assert coastal == 30


def test_the_coastline_is_one_closed_loop():
    """Every perimeter vertex has exactly two coastal roads, so it is a single cycle."""
    degree = collections.Counter(
        v for r in T.COASTAL_ROADS for v in T.ROAD_VERTICES[r]
    )
    assert set(degree.values()) == {2}
    assert len(T.PERIMETER_VERTICES) == len(T.COASTAL_ROADS) == 30


def test_corner_vertices_are_a_strict_subset_of_the_perimeter():
    """The 12 notch vertices are on the coast but touch two tiles — harbours go on
    the perimeter, not just the 18 outermost points."""
    assert len(T.CORNER_VERTICES) == 18
    assert set(T.CORNER_VERTICES) < set(T.PERIMETER_VERTICES)
    assert all(len(T.VERTEX_TILES[v]) == 1 for v in T.CORNER_VERTICES)
    notches = set(T.PERIMETER_VERTICES) - set(T.CORNER_VERTICES)
    assert len(notches) == 12
    assert all(len(T.VERTEX_TILES[v]) == 2 for v in notches)


# --------------------------------------------------------------------------- #
# The ragged-row view                                                         #
# --------------------------------------------------------------------------- #

def test_row_start_indices_are_derived_from_row_lengths():
    assert T.ROW_START_INDICES == (0, 3, 7, 12, 16)


def test_tile_index_and_tile_rowcol_round_trip():
    for tile in range(1, T.NUM_TILES + 1):
        row, col = T.tile_rowcol(tile)
        assert 0 <= row < len(T.ROW_LENGTHS)
        assert 0 <= col < T.ROW_LENGTHS[row]
        assert T.tile_index(row, col) == tile


def test_tile_ids_are_row_major():
    assert [T.tile_index(r, c)
            for r, length in enumerate(T.ROW_LENGTHS)
            for c in range(length)] == list(range(1, T.NUM_TILES + 1))


@pytest.mark.parametrize("row,col", [(0, 3), (5, 0), (-1, 0), (2, 5), (0, -1)])
def test_tile_index_rejects_off_board_coordinates(row, col):
    with pytest.raises(ValueError):
        T.tile_index(row, col)


# --------------------------------------------------------------------------- #
# Id validation: raise, never return a sentinel string                        #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bad", [0, -1, 55, 999])
def test_check_id_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        T.check_id(bad, T.NUM_VERTICES, "vertex")


@pytest.mark.parametrize("bad", ["1", 1.0, None, True])
def test_check_id_rejects_non_integers(bad):
    with pytest.raises(TypeError):
        T.check_id(bad, T.NUM_VERTICES, "vertex")


def test_check_id_returns_valid_ids_unchanged():
    assert T.check_id(7, T.NUM_VERTICES, "vertex") == 7


# --------------------------------------------------------------------------- #
# The generator is not hardcoded to the standard board                        #
# --------------------------------------------------------------------------- #

def test_tile_centres_generalise_to_other_row_layouts(monkeypatch):
    """A 5-6 player board is 3-4-5-6-5-4-3; the generator should not care."""
    monkeypatch.setattr(T, "ROW_LENGTHS", (3, 4, 5, 6, 5, 4, 3))
    centres = T._tile_centres()
    assert len(centres) == 30
    assert len({c for c in centres}) == 30
    assert all(isinstance(x, int) and isinstance(y, int) for x, y in centres)

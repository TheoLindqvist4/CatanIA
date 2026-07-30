"""Board geometry, generated from the row structure alone.

The only input is :data:`ROW_LENGTHS`. Tile centres, corner coordinates, vertex ids,
road ids and every incidence relation are computed from it at import time. There is no
hand-written adjacency data anywhere in this module.

Earlier versions transcribed the relations by hand (54 + 72 + 72 + 19 entries). That is
readable but unverifiable: two entries of the road->roads map were silently wrong, which
corrupted longest-road calculation. Generating them means a relation cannot disagree
with the geometry it describes, and the ids stay identical to the ones drawn in
``Images/`` — ``tests/test_topology.py`` pins that equivalence.

See ``docs/board-geometry.md`` for the diagrams and the reasoning; this docstring is the
short version.

The lattice
-----------
Pointy-top hexes on an integer lattice. ``x`` counts half hex-widths, ``y`` counts
quarter hex-heights, so every corner lands on an integer and corners can be deduplicated
by exact equality — no floating-point tolerance anywhere.

A hex centred at ``(cx, cy)`` has corners::

              (cx, cy-2)          top
       (cx-1, cy-1)  (cx+1, cy-1) upper-left, upper-right
       (cx-1, cy+1)  (cx+1, cy+1) lower-left, lower-right
              (cx, cy+2)          bottom

Rows of hexes are centred on each other and step ``ROW_PITCH = 3`` in ``y``; neighbours
within a row are ``2`` apart in ``x``.

To render at circumradius ``R`` (centre to corner)::

    px = x * R * sqrt(3) / 2
    py = y * R / 2

Numbering
---------
Ids are 1-based and assigned by position, which is what reproduces the drawings:

* **tiles** — row-major over the ragged rows, so 1..3, 4..7, 8..12, 13..16, 17..19
* **vertices** — sorted by ``(y, x)``: top to bottom, then left to right. This yields
  12 rows of 3-4-4-5-5-6-6-5-5-4-4-3 = 54.
* **roads** — sorted by ``(min y, x1 + x2)``: banded top to bottom, then left to right
  within a band. Bands alternate slanted / vertical, giving
  6-4-8-5-10-6-10-5-8-4-6 = 72.

Every table is a tuple indexed directly by id, with an unused empty slot 0, so
``VERTEX_ROADS[7]`` means vertex 7. Lookups are O(1) with no allocation (~38 ns);
the hand-written version rebuilt a dict literal per call (~5,500 ns).
"""

import itertools

# --------------------------------------------------------------------------- #
# The single input                                                            #
# --------------------------------------------------------------------------- #

#: Hexes per row, top to bottom. The standard board is 3-4-5-4-3.
ROW_LENGTHS = (3, 4, 5, 4, 3)

#: Vertical distance between hex row centres, in quarter-height units.
ROW_PITCH = 3

#: Corner offsets from a hex centre, clockwise from the top. Used as a *ring*, so
#: consecutive entries (wrapping) are the hex's six edges.
CORNER_RING = (
    (0, -2),   # top
    (1, -1),   # upper-right
    (1, 1),    # lower-right
    (0, 2),    # bottom
    (-1, 1),   # lower-left
    (-1, -1),  # upper-left
)

VERTICES_PER_TILE = len(CORNER_RING)
ENDPOINTS_PER_ROAD = 2


# --------------------------------------------------------------------------- #
# Generation                                                                  #
# --------------------------------------------------------------------------- #

def _tile_centres():
    """Hex centres in lattice coordinates, in row-major (id) order.

    Rows are centred on the widest row: a row with ``n`` hexes is indented by
    ``widest - n`` half-widths, which is what makes adjacent rows interlock.
    """
    widest = max(ROW_LENGTHS)
    return tuple(
        (widest - length + 2 * col, ROW_PITCH * row)
        for row, length in enumerate(ROW_LENGTHS)
        for col in range(length)
    )


_TILE_CENTRES = _tile_centres()

NUM_TILES = len(_TILE_CENTRES)

#: tile -> its centre in lattice coordinates
TILE_XY = ((None, None),) + _TILE_CENTRES


def _tile_corners(tile):
    """The six corners of ``tile``, in ring (clockwise) order."""
    cx, cy = TILE_XY[tile]
    return tuple((cx + dx, cy + dy) for dx, dy in CORNER_RING)


def _number_vertices():
    """Deduplicate all hex corners and order them top-to-bottom, left-to-right."""
    corners = {
        corner
        for tile in range(1, NUM_TILES + 1)
        for corner in _tile_corners(tile)
    }
    ordered = sorted(corners, key=lambda xy: (xy[1], xy[0]))
    return tuple(ordered), {xy: i + 1 for i, xy in enumerate(ordered)}


_VERTEX_COORDS, _VERTEX_ID = _number_vertices()

NUM_VERTICES = len(_VERTEX_COORDS)

#: vertex -> its lattice coordinates
VERTEX_XY = ((None, None),) + _VERTEX_COORDS


def _number_roads():
    """Collect every hex edge, deduplicate, and band them top-to-bottom.

    A hex edge joins two corners adjacent in :data:`CORNER_RING`. Interior edges are
    produced twice, once from each hex, so a set collapses them.
    """
    edges = set()
    for tile in range(1, NUM_TILES + 1):
        ring = _tile_corners(tile)
        for a, b in zip(ring, ring[1:] + ring[:1]):
            edges.add((min(_VERTEX_ID[a], _VERTEX_ID[b]),
                       max(_VERTEX_ID[a], _VERTEX_ID[b])))

    def key(edge):
        (x1, y1), (x2, y2) = VERTEX_XY[edge[0]], VERTEX_XY[edge[1]]
        return (min(y1, y2), x1 + x2, max(y1, y2))

    return tuple(sorted(edges, key=key))


_ROAD_ENDPOINTS = _number_roads()

NUM_ROADS = len(_ROAD_ENDPOINTS)

#: road -> its two endpoint vertices, ascending
ROAD_VERTICES = ((),) + _ROAD_ENDPOINTS

_ROAD_ID = {endpoints: i + 1 for i, endpoints in enumerate(_ROAD_ENDPOINTS)}


# --------------------------------------------------------------------------- #
# Derived relations                                                           #
# --------------------------------------------------------------------------- #

def _freeze(mapping, size):
    """``{id: iterable[int]}`` -> tuple indexed 1..size, each entry sorted.

    Slot 0 is an unused empty tuple so ids index the result directly. Sorting keeps
    legal-move enumeration deterministic.
    """
    table = [()] * (size + 1)
    for key, values in mapping.items():
        table[key] = tuple(sorted(values))
    return tuple(table)


def _vertex_id_of(xy):
    return _VERTEX_ID[xy]


def road_between(u, v):
    """The road joining two adjacent vertices, or ``None`` if they are not adjacent."""
    return _ROAD_ID.get((min(u, v), max(u, v)))


def _tile_rings():
    """Per tile, its corner vertices in ring order and its boundary roads."""
    vertices, roads = {}, {}
    for tile in range(1, NUM_TILES + 1):
        ring = [_vertex_id_of(c) for c in _tile_corners(tile)]
        vertices[tile] = ring
        roads[tile] = [road_between(a, b) for a, b in zip(ring, ring[1:] + ring[:1])]
    return vertices, roads


_TILE_VERTEX_RING, _TILE_ROAD_RING = _tile_rings()

#: tile -> its six corner vertices, ascending. Ascending order is also geometric
#: order (top, upper-left, upper-right, lower-left, lower-right, bottom), because
#: vertex ids are assigned by ``(y, x)``.
TILE_VERTICES = _freeze(_TILE_VERTEX_RING, NUM_TILES)

#: tile -> the six roads on its boundary
TILE_ROADS = _freeze(_TILE_ROAD_RING, NUM_TILES)


def _invert(ring, size):
    """Invert a tile -> ids mapping into id -> tiles."""
    out = {i: [] for i in range(1, size + 1)}
    for tile, ids in ring.items():
        for value in ids:
            out[value].append(tile)
    return out


#: vertex -> the 1-3 tiles touching it
VERTEX_TILES = _freeze(_invert(_TILE_VERTEX_RING, NUM_VERTICES), NUM_VERTICES)

#: road -> the 1-2 tiles it borders. A coastal road borders one.
ROAD_TILES = _freeze(_invert(_TILE_ROAD_RING, NUM_ROADS), NUM_ROADS)

#: tile -> neighbouring tiles, i.e. those sharing a road
TILE_ADJACENCY = _freeze(
    {
        tile: {t for road in TILE_ROADS[tile] for t in ROAD_TILES[road] if t != tile}
        for tile in range(1, NUM_TILES + 1)
    },
    NUM_TILES,
)


def _vertex_relations():
    """Vertex neighbours and incident roads, in one pass over the roads."""
    neighbours = {v: [] for v in range(1, NUM_VERTICES + 1)}
    roads = {v: [] for v in range(1, NUM_VERTICES + 1)}
    for road in range(1, NUM_ROADS + 1):
        u, v = ROAD_VERTICES[road]
        neighbours[u].append(v)
        neighbours[v].append(u)
        roads[u].append(road)
        roads[v].append(road)
    return neighbours, roads


_VERTEX_NEIGHBOURS_SRC, _VERTEX_ROADS_SRC = _vertex_relations()

#: vertex -> adjacent vertices (one road away)
VERTEX_NEIGHBOURS = _freeze(_VERTEX_NEIGHBOURS_SRC, NUM_VERTICES)

#: vertex -> the roads meeting at it
VERTEX_ROADS = _freeze(_VERTEX_ROADS_SRC, NUM_VERTICES)

#: road -> roads sharing an endpoint with it
ROAD_NEIGHBOURS = _freeze(
    {
        road: {
            other
            for endpoint in ROAD_VERTICES[road]
            for other in VERTEX_ROADS[endpoint]
            if other != road
        }
        for road in range(1, NUM_ROADS + 1)
    },
    NUM_ROADS,
)

#: road -> twice its midpoint, in lattice coordinates. Doubled to stay integral;
#: halve it when rendering.
ROAD_MIDPOINT_2X = ((None, None),) + tuple(
    (VERTEX_XY[u][0] + VERTEX_XY[v][0], VERTEX_XY[u][1] + VERTEX_XY[v][1])
    for u, v in _ROAD_ENDPOINTS
)

# --- coastline ------------------------------------------------------------- #
# Three different things, easy to confuse. Harbours attach to PERIMETER_VERTICES
# (30 of them), *not* to CORNER_VERTICES (18) — a harbour sits on a coastal road,
# and 12 perimeter vertices sit in the notches where two tiles still meet.

#: Roads bordering exactly one tile: the coastline itself. 30 on a standard board.
COASTAL_ROADS = tuple(
    r for r in range(1, NUM_ROADS + 1) if len(ROAD_TILES[r]) == 1
)

#: Every vertex on the coastline, i.e. an endpoint of a coastal road.
PERIMETER_VERTICES = tuple(sorted(
    {v for r in COASTAL_ROADS for v in ROAD_VERTICES[r]}
))

#: The board's outermost points — vertices touching a single tile. A subset of
#: PERIMETER_VERTICES.
CORNER_VERTICES = tuple(
    v for v in range(1, NUM_VERTICES + 1) if len(VERTEX_TILES[v]) == 1
)


# --------------------------------------------------------------------------- #
# The ragged-row view, for humans and for display                             #
# --------------------------------------------------------------------------- #

#: Index of the first tile in each row, 0-based. Derived from ROW_LENGTHS.
ROW_START_INDICES = tuple(
    itertools.accumulate((0,) + ROW_LENGTHS[:-1])
)

#: tile -> (row, col) in the ragged grid
TILE_ROWCOL = ((None, None),) + tuple(
    (row, tile - 1 - ROW_START_INDICES[row])
    for tile in range(1, NUM_TILES + 1)
    for row, start in enumerate(ROW_START_INDICES)
    if start <= tile - 1 < start + ROW_LENGTHS[row]
)

#: (row, col) -> tile
ROWCOL_TILE = {
    TILE_ROWCOL[tile]: tile for tile in range(1, NUM_TILES + 1)
}

#: Vertices grouped into their 12 horizontal rows, top to bottom.
VERTEX_ROWS = tuple(
    tuple(vertex for vertex, _ in group)
    for _, group in itertools.groupby(
        ((v, VERTEX_XY[v][1]) for v in range(1, NUM_VERTICES + 1)),
        key=lambda pair: pair[1],
    )
)


# --------------------------------------------------------------------------- #
# Public helpers                                                              #
# --------------------------------------------------------------------------- #

def tile_index(row, col):
    """(row, col) in the ragged grid -> tile id."""
    try:
        return ROWCOL_TILE[(row, col)]
    except KeyError:
        raise ValueError(f"({row}, {col}) is not a tile on the board") from None


def tile_rowcol(tile):
    """Tile id -> its (row, col) in the ragged grid."""
    return TILE_ROWCOL[check_id(tile, NUM_TILES, "tile")]


def check_id(value, size, kind):
    """Validate a 1-based id, raising instead of returning a sentinel.

    The hand-written maps returned an error *string* for out-of-range input, so a
    caller iterating the result silently looped over its characters.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{kind} must be an int, got {type(value).__name__}")
    if not 1 <= value <= size:
        raise ValueError(f"{kind} must be in 1..{size}, got {value}")
    return value


# --------------------------------------------------------------------------- #
# Import-time sanity check                                                    #
# --------------------------------------------------------------------------- #

def _validate():
    """Structural check on the generated tables (well under 1 ms).

    Cheap insurance that a change to the generator has not quietly reshaped the
    board. ``python -O`` strips these; tests/test_topology.py is the real guarantee.
    """
    # Euler-style consistency: 19 hexes, 54 corners, 72 edges
    assert NUM_TILES == sum(ROW_LENGTHS)
    assert all(len(TILE_VERTICES[t]) == VERTICES_PER_TILE
               for t in range(1, NUM_TILES + 1))
    assert all(len(TILE_ROADS[t]) == VERTICES_PER_TILE
               for t in range(1, NUM_TILES + 1))
    assert all(len(ROAD_VERTICES[r]) == ENDPOINTS_PER_ROAD
               for r in range(1, NUM_ROADS + 1))

    # the road ordering key must be a total order, or ids would be arbitrary
    assert len({
        (min(VERTEX_XY[u][1], VERTEX_XY[v][1]),
         VERTEX_XY[u][0] + VERTEX_XY[v][0],
         max(VERTEX_XY[u][1], VERTEX_XY[v][1]))
        for u, v in _ROAD_ENDPOINTS
    }) == NUM_ROADS

    # a hex board has no vertex of degree > 3, which is what makes "no repeated road"
    # imply "no repeated vertex" during path search
    assert all(2 <= len(VERTEX_NEIGHBOURS[v]) <= 3
               for v in range(1, NUM_VERTICES + 1))
    assert all(len(VERTEX_ROADS[v]) == len(VERTEX_NEIGHBOURS[v])
               for v in range(1, NUM_VERTICES + 1))
    assert all(1 <= len(VERTEX_TILES[v]) <= 3 for v in range(1, NUM_VERTICES + 1))
    assert all(1 <= len(ROAD_TILES[r]) <= 2 for r in range(1, NUM_ROADS + 1))

    # symmetry
    assert all(v in VERTEX_NEIGHBOURS[n]
               for v in range(1, NUM_VERTICES + 1) for n in VERTEX_NEIGHBOURS[v])
    assert all(r in ROAD_NEIGHBOURS[o]
               for r in range(1, NUM_ROADS + 1) for o in ROAD_NEIGHBOURS[r])
    assert all(t in TILE_ADJACENCY[o]
               for t in range(1, NUM_TILES + 1) for o in TILE_ADJACENCY[t])

    # one road per adjacent vertex pair
    assert len({frozenset(ROAD_VERTICES[r]) for r in range(1, NUM_ROADS + 1)}) == NUM_ROADS

    # Every hex contributes 6 edge-incidences; interior edges are shared by two hexes
    # and coastal edges by one. So 6*T = 2*interior + coast and interior + coast = R.
    interior = NUM_ROADS - len(COASTAL_ROADS)
    assert 2 * interior + len(COASTAL_ROADS) == VERTICES_PER_TILE * NUM_TILES
    # the coastline is a single closed loop, so it has as many vertices as edges
    assert len(PERIMETER_VERTICES) == len(COASTAL_ROADS)
    assert set(CORNER_VERTICES) <= set(PERIMETER_VERTICES)

    # the ragged-row view must agree with the lattice
    assert sum(len(row) for row in VERTEX_ROWS) == NUM_VERTICES
    assert len(ROWCOL_TILE) == NUM_TILES


_validate()

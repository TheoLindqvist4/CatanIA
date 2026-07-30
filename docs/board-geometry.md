# Board geometry

How the 19 tiles, 54 settlement positions and 72 road positions are numbered, and why the
code generates them instead of listing them.

The ids here are exactly the ones drawn in [`Images/`](../Images). Those drawings are the
contract: `tests/test_topology.py` transcribes them independently and asserts the
generated tables match. If you change the generator and the drawings stop describing the
code, those tests fail.

---

## 1. Why generated, not listed

The original code listed every relation by hand — 54 vertex-neighbour entries, 72
road-endpoint entries, 72 road-neighbour entries, 19 tile-adjacency entries. That is easy
to read one line at a time, and it was written that way deliberately so a human could
follow the logic.

The problem is that it cannot be checked. Two entries of the road→roads map were wrong:

| road | endpoints | listed neighbours | correct |
|---|---|---|---|
| 2 | (1, 5) | 1, 8 | 1, **3**, 8 |
| 51 | (35, 40) | 42, 56, 57 | 42, **43**, 56, 57 |

Road 2 and road 3 both touch vertex 5, so they are neighbours. Road 51 and road 43 both
touch vertex 35. Both omissions were silent: longest-road undercounted, and legal-move
enumeration dropped moves, only for players whose network happened to reach vertex 5 or 35.

Nothing about the listing style makes that visible. Generating the relations from
coordinates means a relation cannot disagree with the geometry it describes — the error
class stops existing rather than getting fixed once. Readability is preserved by this
document plus the drawings, which is where it belongs.

The whole geometry now derives from one line:

```python
ROW_LENGTHS = (3, 4, 5, 4, 3)
```

---

## 2. The lattice

Pointy-top hexes on an **integer** lattice, so corners shared by several hexes are
deduplicated by exact equality. No floating-point tolerance anywhere.

- `x` counts **half hex-widths**
- `y` counts **quarter hex-heights**

A hex centred at `(cx, cy)` has six corners:

```
                 (cx, cy-2)                    top
                     ______
                    /      \
      (cx-1, cy-1) /        \ (cx+1, cy-1)     upper-left, upper-right
                  |          |
                  |  centre  |
                  |          |
      (cx-1, cy+1) \        / (cx+1, cy+1)     lower-left, lower-right
                    \______/
                 (cx, cy+2)                    bottom
```

Consecutive hexes in a row are `2` apart in `x`. Consecutive rows are `ROW_PITCH = 3`
apart in `y` — which is the correct 3/4-of-height overlap for pointy-top hexes, and is
what makes the rows interlock.

A row of `n` hexes is indented by `widest - n` half-widths, centring it on the widest row.
For 3-4-5-4-3 that gives indents of 2, 1, 0, 1, 2.

**Rendering.** At circumradius `R` (centre to corner):

```
px = x * R * sqrt(3) / 2
py = y * R / 2
```

---

## 3. Tiles — 19, row-major

```
          1   2   3
        4   5   6   7
      8   9  10  11  12
       13  14  15  16
         17  18  19
```

`ROW_START_INDICES = (0, 3, 7, 12, 16)`, derived from `ROW_LENGTHS`.

Tile ids ascend row by row, left to right, so the ragged-row view is pure presentation.
`Board` stores tiles in **flat arrays indexed by tile id** and rebuilds the rows only for
display — see [decision 0002](decisions/0002-flat-tile-arrays-not-ragged-rows.md).

`tile_index(row, col)` and `tile_rowcol(tile)` convert between the two when a human-facing
view needs it.

---

## 4. Settlement positions — 54 vertices

Every hex corner, deduplicated, then **sorted by `(y, x)`**: top to bottom, then left to
right. That ordering is what reproduces the numbering in
[`Images/Catan_settlement_positions.png`](../Images/Catan_settlement_positions.png).

The result is 12 horizontal rows with a hexagonal profile of **3-4-4-5-5-6-6-5-5-4-4-3**:

```
 row  1   y=-2            1    2    3
 row  2   y=-1         4    5    6    7
 row  3   y= 1         8    9   10   11
 row  4   y= 2      12   13   14   15   16
 row  5   y= 4      17   18   19   20   21
 row  6   y= 5   22   23   24   25   26   27
 row  7   y= 7   28   29   30   31   32   33
 row  8   y= 8      34   35   36   37   38
 row  9   y=10      39   40   41   42   43
 row 10   y=11         44   45   46   47
 row 11   y=13         48   49   50   51
 row 12   y=14            52   53   54
```

Rows come in pairs with the same size (4-4, 5-5, 6-6, 5-5, 4-4) because each hex row
contributes two vertex rows: its upper corners and its lower corners.

`VERTEX_ROWS` exposes this grouping.

### Corner ordering within a tile

Because ids ascend by `(y, x)`, sorting a tile's six corners ascending is *also* their
geometric order — top, upper-left, upper-right, lower-left, lower-right, bottom. Tile 1:

```
                    1                     top          (2, -2)
                  _____
        road 1   /     \  road 2
              4 /       \ 5               upper L / R  (1,-1) (3,-1)
               |         |
       road 7  |  tile 1 |  road 8
               |         |
              8 \       / 9               lower L / R  (1, 1) (3, 1)
       road 12  \_____/  road 13
                    13                    bottom       (2,  2)
```

`TILE_VERTICES[1] == (1, 4, 5, 8, 9, 13)` and `TILE_ROADS[1] == (1, 2, 7, 8, 12, 13)`.

### Vertex degree

18 vertices have 2 neighbours, 36 have 3. **No vertex exceeds degree 3.** This matters for
longest-road search: a path that never reuses a road also cannot pass through a vertex
more than… well, see [decision 0006](decisions/0006-longest-road-intersection-reuse.md) —
degree 3 is exactly why that edge case exists at all.

---

## 5. Road positions — 72 edges

Every pair of corners adjacent in the hex ring, deduplicated (interior edges are produced
twice, once per hex), then **sorted by `(min y, x₁ + x₂)`** — banded top to bottom, then
left to right within a band.

Roads fall into alternating bands:

| band | roads | y span | kind |
|---|---|---|---|
| 1 | 1–6 | -2 → -1 | slanted |
| 2 | 7–10 | -1 → 1 | vertical |
| 3 | 11–18 | 1 → 2 | slanted |
| 4 | 19–23 | 2 → 4 | vertical |
| 5 | 24–33 | 4 → 5 | slanted |
| 6 | 34–39 | 5 → 7 | vertical |
| 7 | 40–49 | 7 → 8 | slanted |
| 8 | 50–54 | 8 → 10 | vertical |
| 9 | 55–62 | 10 → 11 | slanted |
| 10 | 63–66 | 11 → 13 | vertical |
| 11 | 67–72 | 13 → 14 | slanted |

Sizes: **6-4-8-5-10-6-10-5-8-4-6 = 72**.

*Slanted* bands are the short roads between a hex row's top and upper corners. *Vertical*
bands are the tall sides of a hex row. Vertical bands span `y` by 2 (`-1 → 1`), slanted
bands by 1 (`-2 → -1`), which is why sorting by `min y` separates them cleanly — no two
bands share a minimum.

`road_between(u, v)` inverts `ROAD_VERTICES`, returning `None` if the two vertices are not
adjacent.

---

## 6. Coastline

Three related sets that are easy to confuse. Getting them mixed up would break harbours,
so they are named apart:

| constant | count | definition |
|---|---|---|
| `COASTAL_ROADS` | 30 | roads bordering exactly one tile — the coastline itself |
| `PERIMETER_VERTICES` | 30 | every vertex on a coastal road |
| `CORNER_VERTICES` | 18 | vertices touching exactly one tile — the board's points |

The count follows from edge incidences: 19 hexes × 6 edges = 114 incidences; interior edges
are counted twice and coastal edges once, and interior + coastal = 72. So
`2·interior + coastal = 114` gives **coastal = 30**.

The coastline is a single closed loop — every perimeter vertex has exactly two coastal
roads — so it has as many vertices as edges.

`CORNER_VERTICES` is a *strict* subset: the other 12 perimeter vertices sit in the notches
where two tiles still meet.

```
CORNER_VERTICES  1  2  3  4  7 12 16 22 27 28 33 39 43 48 51 52 53 54
notches          5  6  8 11 17 21 34 38 44 47 49 50
```

**Harbours (Phase 2) attach to `PERIMETER_VERTICES`, not `CORNER_VERTICES`** — a harbour
sits on a coastal road, and both its endpoints are legal settlement spots that should get
the port.

---

## 7. A strategic consequence worth knowing

The tiles meeting at any vertex are **pairwise adjacent**. Combined with the
balanced-generation house rule ("no equal numbers on adjacent tiles"), this makes
double-production vertices *structurally impossible*: no settlement can ever collect twice
from a single roll.

The "double 6" spot that exists in official Catan simply does not occur on this board. That
is a real difference in the game an agent learns, not an implementation detail. See
[decision 0005](decisions/0005-balanced-board-generation.md).

---

## 8. Table reference

All tables are tuples indexed directly by 1-based id, with an unused empty slot 0, so
`VERTEX_ROADS[7]` means vertex 7. Lookups are O(1) with no allocation (~38 ns); the
hand-written version rebuilt a dict literal per call (~5,500 ns).

| table | maps |
|---|---|
| `TILE_XY`, `VERTEX_XY` | id → lattice coordinates |
| `ROAD_MIDPOINT_2X` | road → twice its midpoint (integral; halve to render) |
| `TILE_VERTICES` | tile → its 6 corners |
| `TILE_ROADS` | tile → its 6 boundary roads |
| `TILE_ADJACENCY` | tile → tiles sharing a road |
| `VERTEX_TILES` | vertex → the 1–3 tiles touching it |
| `VERTEX_NEIGHBOURS` | vertex → adjacent vertices |
| `VERTEX_ROADS` | vertex → roads meeting there |
| `ROAD_VERTICES` | road → its 2 endpoints |
| `ROAD_NEIGHBOURS` | road → roads sharing an endpoint |
| `ROAD_TILES` | road → the 1–2 tiles it borders |
| `VERTEX_ROWS` | the 12 horizontal vertex rows |
| `TILE_ROWCOL`, `ROWCOL_TILE` | tile ↔ (row, col) |

`topology._validate()` runs at import and asserts the structural invariants (counts,
symmetry, degree bounds, the edge-incidence identity, and that the road sort key is a total
order). It costs well under a millisecond. `python -O` strips it; the tests are the real
guarantee.

# 0001 — Generate the board geometry from the row structure

**Status:** accepted · Phase 0

## Context

The board's geometry was written out by hand: 19 tile→vertex entries, 72 road→vertex
entries, 54 vertex→vertex entries, 54 vertex→road entries, 72 road→road entries and 19
tile→tile entries. About 440 lines of data. This was a deliberate choice — the commit
message "Explications des positions choisis pour les routes et les villages" says as much
— so that a human could follow which position is where.

It worked for reading, but it could not be checked. Two entries of the road→road map were
wrong (see [the audit](../audit-2026-07-30.md), B1): road 2 was missing neighbour 3, road
51 was missing neighbour 43. Both omissions were invisible on inspection and silently
corrupted longest-road calculation and legal-move enumeration for any player reaching
vertex 5 or vertex 35.

A second problem: each lookup rebuilt its dict literal on every call — 5.49 µs measured —
which is fatal for a project whose point is 10⁵–10⁶ RL steps.

## Decision

Generate everything from **one input**, `ROW_LENGTHS = (3, 4, 5, 4, 3)`.

Place hex centres on an integer lattice, compute the six corners of each hex, deduplicate
them by exact coordinate equality, and assign ids by position:

- **vertices** sorted by `(y, x)` — top to bottom, then left to right
- **roads** sorted by `(min y, x₁ + x₂)` — banded top to bottom, then left to right

Then derive every relation from `TILE_VERTICES` and `ROAD_VERTICES`. Freeze the results
into tuples indexed by id at import time.

Integer coordinates are load-bearing: they let shared corners be deduplicated by equality
rather than by floating-point tolerance.

The generated ids are **identical** to the hand-written ones, so
[`Images/`](../../Images) remains an accurate reference.
`tests/test_topology.py` transcribes those drawings independently and asserts the match —
the drawings, not the code, are the contract.

## Consequences

**Good**

- The bug class disappears. A relation cannot disagree with the geometry it describes,
  because it is computed from it. Nothing to keep in sync.
- 440 lines of data → 0.
- 5,490 ns → 38 ns per lookup (145×); board construction 2.4× faster.
- Real coordinates fall out for free (`VERTEX_XY`, `TILE_XY`, `ROAD_MIDPOINT_2X`),
  which rendering needs and which the AI can use for spatial features.
- Coastline detection becomes derivable, which Phase 2 harbours need.
- Generalises: a 5–6 player board is `(3, 4, 5, 6, 5, 4, 3)` and nothing else changes.

**Bad**

- Reading `topology.py` no longer tells you which vertex is where. That knowledge moved to
  [`docs/board-geometry.md`](../board-geometry.md) and the drawings, which is where it
  belongs — a table of 72 entries was never actually readable as a whole.
- The numbering now depends on two sort keys being correct. Both are pinned against the
  drawings by tests, and `_validate()` asserts the road key is a total order at import.

## Alternatives considered

- **Patch the two bad entries.** Rejected: fixes two instances, leaves the class. There is
  no way to know the remaining 438 lines are right by reading them.
- **Keep the tables, add tests that check them.** The tests would have to derive ground
  truth from coordinates anyway — at which point the tables are redundant.
- **Axial/cube hex coordinates.** More standard, but the ragged 3-4-5-4-3 row form is what
  the drawings and the display use, and the doubled-integer lattice makes corner
  deduplication exact. Axial adds a conversion for no gain here.

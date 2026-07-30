# 0002 — Store tiles in flat arrays, not ragged rows

**Status:** accepted · Phase 0

## Context

`Board` stored tiles the way they are drawn: `grid` and `tile_grid` were lists of 5 lists
with lengths 3-4-5-4-3. Every tile access went through a `(row, col)` translation
(`get_flat_index`, `get_coordinates_from_index`).

That shape leaked into the logic. Number placement validated a tile against its already-
placed neighbours, and the way it knew which neighbours were placed was that
`get_coordinates_from_index` returned `(None, None)` when `col >= len(self.grid[row])` —
i.e. it used the *partial length of a half-filled row* as an implicit progress marker.
Correct, but subtle enough that changing generation risked silently changing which
constraints were checked.

## Decision

Store `tile_numbers[1..19]` and `tile_resources[1..19]` as flat lists indexed by tile id,
with slot 0 unused.

Tile ids are already row-major, so the row layout is pure presentation. `grid` and
`tile_grid` become read-only properties that rebuild the rows on demand for display, and
`display_board()` uses them.

Number placement now iterates tiles `1..19` and tests `placed[other] is not None`, which
says what it means.

## Consequences

**Good**

- The generation invariant is explicit instead of encoded in a list length.
- Tile lookup is a direct index; `get_flat_index` / `get_coordinates_from_index` are gone.
- `Board` no longer imports `tile_index` / `tile_rowcol` at all — the row view is confined
  to two properties and one display method.
- Flat arrays indexed by id are also the shape Phase 3's observation encoder wants, so this
  is one fewer conversion later.

**Neutral**

- `grid` / `tile_grid` still exist and still return the ragged rows, so display and any
  human-facing view are unchanged. They now return copies, which is a small correctness
  improvement.

**Cost**

- Two properties rebuild small lists on access. Only display calls them; the hot paths use
  the flat arrays.

## Note on reproducibility

The rewrite consumes the injected RNG in exactly the same order as the ragged version —
one shuffle of the number pool per attempt, then one shuffle of the resource pool — and
fills tiles in the same row-major order. A given seed therefore produces the same board
before and after. `test_pinned_layout_for_a_fixed_seed` pins seed 42 as a canary against
unintended drift.

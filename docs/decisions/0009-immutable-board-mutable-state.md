# 0009 — The board is immutable; everything mutable is in `GameState`

**Status:** accepted · Phase 1

## Context

The old `Board` owned both the layout (which number and resource sit on each tile) and the
occupancy (`settlement_positions`, `road_positions`). Occupancy was maintained by *deleting*
from an "available" set, which is the root of audit finding B5: *empty*, *blocked by the
distance rule* and *occupied by player N* became indistinguishable, and no owner was
recorded.

Phase 1 needed a clean split, and there is a second reason to want one:
`GameState.clone()` is on the hot path for search. Copying the layout, the 54-entry
production map and the 11-entry payout index on every clone would be pure waste, since none
of it ever changes.

## Decision

**`catan.board.Board` is immutable after construction.** It holds only what a seed
determines: `tile_numbers`, `tile_resources`, the per-vertex production records, and the
payout index. It has no ownership fields and no robber.

**`catan.state.GameState` holds everything that changes**, as parallel arrays indexed by
topology id: `vertex_owner`, `vertex_piece`, `edge_owner`, plus hands and piece supplies.

`clone()` therefore shares the board **by reference** and copies only the mutable arrays.

`Board` gets value `__eq__` and `__hash__` over its layout. Identity comparison was
tempting — clones share the object, so `is` would be free — but it is wrong: replaying a
seed builds an equal board in a *new* object, and those two games are the same game. This
was caught by a reproducibility test failing while every mutable field matched.

### The robber lives in `GameState`

Even though a robber feels like a board thing, it moves during play, so it belongs to the
state. Phase 2 adds it there. `Board.desert_tile` is where it starts, which *is* a property
of the layout.

## Consequences

**Good**

- Availability is derived from ownership rather than stored, so the three cases B5 conflated
  stay distinct. `respects_distance_rule` reads the neighbours; nothing is destroyed.
- Cloning skips ~54 production records, a 19-tile layout and an 11-roll index.
- A whole class of aliasing bug is impossible: no game can corrupt a board another clone is
  reading.
- `Board` is hashable, which Phase 3 can use to key caches per layout.

**Bad**

- The "board" a player sees is spread across two objects, so rendering needs both. That is
  the correct split — a renderer is a view over layout *and* position — but it does mean
  `Board.render()` shows only the tiles.
- Nothing enforces immutability at runtime; it is a convention plus tests. Freezing the
  arrays would cost more than it is worth here.

## Enforced by

`test_the_board_carries_no_ownership_or_robber_state`,
`test_a_board_is_unchanged_by_a_full_game` (snapshots the layout, plays a full 3-player
game, compares),
`test_a_clone_shares_the_immutable_board_but_nothing_mutable`,
`test_boards_compare_by_layout_not_identity`,
`test_empty_blocked_and_occupied_are_now_distinguishable`.

# 0005 — Balanced board generation (a house rule)

**Status:** accepted · pre-existing, documented in Phase 0 · **revisit in Phase 2**

## Context

`Board.is_number_valid` rejects a number token if the tile has a neighbour with:

- the **same** number, or
- a **6** when placing an 8, or an 8 when placing a 6, or
- a **2** when placing a 12, or a 12 when placing a 2.

Official Catan does not do this. The rulebook gives a fixed spiral ("Variable" setup places
tokens in a spiral from a corner), and the beginner layout is a specific fixed board.
Neither constrains adjacency the way this does.

## Decision

Keep it for now, and **document it as a deviation** rather than let it look like a rule.
Placement is greedy with reshuffle-on-dead-end, bounded at 100 attempts (median 2, max 5
observed).

Phase 2 should make generation mode a config flag, with the official spiral available.

## Consequences

### The important one: double production becomes impossible

The tiles meeting at any vertex are **pairwise adjacent** — a geometric fact, asserted by
`test_all_tiles_meeting_at_a_vertex_are_pairwise_adjacent`. Combined with "no equal
numbers on adjacent tiles", this means **no vertex can ever touch two tiles with the same
number**, so no settlement can ever collect twice from a single roll.

Verified: zero double-production vertices across 3,000 generated boards.

The "double 6" or "double 8" spot that exists in official Catan therefore does not exist
here at all. That is a **real change to the game**, not an implementation detail:

- one of Catan's classic opening heuristics is unavailable
- the variance of resource income is lower than in the real game
- an agent trained here will not learn to value or contest those spots, and will be
  mis-calibrated against a real board or a real opponent

This is the main reason the flag matters. If the goal is an agent that plays *Catan*, it
should eventually train on official layouts.

### Other effects

**Good**

- Boards are more even, so games are less decided by setup. Arguably better for early
  training signal — less noise from board luck.
- Extreme clusters (three high-probability numbers around one vertex) cannot occur.

**Bad**

- Generation can dead-end and must retry, so it is not a single pass.
- The reachable board distribution is narrower than official Catan's, which is a form of
  train/test mismatch if the agent is ever evaluated on real boards.

## Enforced by

`test_balanced_generation_rule_holds`,
`test_the_balanced_rule_makes_double_production_impossible`,
`test_all_tiles_meeting_at_a_vertex_are_pairwise_adjacent`.

The second of those must become conditional on the generation mode when the spiral option
lands — it encodes a consequence of this house rule, not a property of Catan.

# 0004 — Model the desert as the tile numbered 7

**Status:** accepted · pre-existing, documented in Phase 0

## Context

In official Catan the desert carries **no** number token: 18 tokens are distributed across
the 18 non-desert tiles, and the desert is placed separately.

This implementation puts a 7 in the token pool, giving 19 tokens for 19 tiles, and makes
whichever tile draws the 7 the desert.

## Decision

Keep it. Verified as an invariant over 200+ generated boards: the desert is always the
7-tile and the 7-tile is always the desert.

The number distribution is otherwise exactly standard — one 2, two each of 3–6 and 8–11,
one 12 — so 18 real tokens plus the 7 marker.

## Consequences

**Good**

- One shuffle places both the numbers and the desert. No separate desert step.
- 7 is already the roll that produces nothing, so "the desert's number is 7" is
  self-consistent rather than a hack: `producers_for(7)` is empty because the desert is
  excluded from the payout index, which makes a 7 **structurally** inert rather than
  something every caller must remember to filter.
- The desert position is uniformly random across tiles, which is a reasonable variant.

**Different from official Catan**

- The desert's placement is coupled to the token shuffle rather than chosen independently.
  In practice this only means the desert is uniformly distributed, which official random
  setup also gives.
- The balanced-adjacency rule ([0005](0005-balanced-board-generation.md)) sees the 7 as a
  number. Since there is exactly one 7, "no equal neighbours" is trivially satisfied for
  it, so this has no effect.

**Watch out**

- Anything that treats `number_at(tile)` as "a roll that pays out" must exclude 7, or use
  `producers_for()` / `Board.ROBBER_ROLL` instead of hardcoding.
- When the robber arrives (Phase 2) it starts on `Board.desert_tile`, which is derived from
  the resource array rather than from the number.

## Enforced by

`test_the_desert_is_always_the_seven_tile`, `test_rolling_a_seven_produces_nothing`,
`test_desert_never_appears_in_any_payout`.

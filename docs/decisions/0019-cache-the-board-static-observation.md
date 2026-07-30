# 0019 — Cache the board-static half of an observation

**Status:** accepted · **Date:** 2026-07-30 · **Phase:** 8

## The problem

Profiling the first training rollout put `catan.encoder.encode` at **57% of total runtime** —
more than the game rules, the network forward pass and the PPO update combined.
`_encode_vertices` alone was 24.7s of 55.7s, with 2.37 million generator-expression calls and
978,000 `sum()` calls in a 14,614-encode sample.

It was recomputing, for every observation, quantities that had not changed since the board was
generated.

## The decision

Split an observation into the part that depends on the *layout* and the part that depends on
*play*, and compute the first one once per `Board`.

Board-static: which resource sits on a tile, its number token, its payout odds, which harbours
a vertex can reach, and a vertex's pip potential. That is roughly 40% of the vector. It is
cached on the board instance — `Board` is immutable and shared across clones, so one template
serves an entire training run — and `encode` starts from a copy of it.

Two further hoists in the dynamic half:

- `_encode_tiles` now writes **one** value, the robber flag, instead of walking 19 tiles.
- The two buildability flags were 108 per-vertex calls into `rules.respects_distance_rule` and
  `rules.touches_own_road`. Both are now derived in one pass over what is *owned* — at most
  ten settlements and fifteen roads — rather than by interrogating fifty-four vertices.

## Verified, not assumed

An optimisation that changes the observation by one float silently changes what every agent
sees and invalidates every checkpoint. So the check was not "the tests still pass" but
"the output is identical":

The pre-refactor `encoder.py` was loaded from git as a second module and both were run on the
same states. **3,200 encodings across 8 seeds and both players: 0 mismatches.** The 36
existing encoder tests pass unchanged, including
`test_buildability_flags_agree_with_the_rules`, which cross-checks the hoisted flags against
`catan.rules` for every vertex of every board — that test is what keeps the shortcut honest as
the rules evolve.

| | µs per encode |
|---|---|
| before | 276 |
| after | 162 |

(Measured under load, same script, same process — the ratio is the meaningful part.)

## Why the rules keep the flags too

`respects_distance_rule` and `touches_own_road` were not deleted. They are the legality
authority, used by `catan.rules` itself, and they answer a question about *one* vertex, which
is the right shape for that job. The encoder needs the answer for all fifty-four at once, which
is a different computation with the same meaning. Two implementations of one rule is exactly
the thing [0001](0001-generate-geometry-from-the-row-structure.md) argues against — the
cross-check test is the price of keeping both, and it is worth paying at 57% of training time.

## See also

- [0009 — immutable board, mutable state](0009-immutable-board-mutable-state.md) — what makes
  caching on the board safe
- [0017 — PPO self-play](0017-ppo-self-play.md) — why the encoder was on the hot path at all

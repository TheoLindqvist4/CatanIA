# 0008 — `apply` mutates; copy with `clone()`

**Status:** accepted · Phase 1

## Context

[`ROADMAP.md`](../../ROADMAP.md) originally specified `rules.apply(state, action) -> state`
alongside the requirement that `rules.py` be "pure". Read strictly, *pure* means returning a
new state and leaving the input untouched, which is the usual functional-core design and is
easy to reason about.

But the consumer is a search algorithm. MCTS applies an action, descends, and unwinds
thousands of times per decision. A copy-on-every-apply engine pays the copy for every node
whether or not the caller needed the old state — and callers usually do not: they clone once
at a branch point and then play forward destructively.

## Decision

`apply(state, action)` **mutates** `state` and returns it for chaining.
`GameState.clone()` is the explicit way to keep the old one:

```python
after = rules.apply(state.clone(), action)   # non-destructive, when you want it
rules.apply(state, action)                   # the fast path
```

"Pure" is kept in the sense that matters for correctness and for training: no I/O, no global
state, no module-level mutable data, and no use of the global `random` module. Given
`(state, action)` and the state's own generator, the result is fully determined.

`apply` raises `IllegalAction` rather than returning a status, and validates through the
same `can_*` predicates `legal_actions` uses — so a move can never be offered by one and
rejected by the other. That was the original bug: `Game.check_valid_*` computed legal
positions and `place_*` ignored them.

## Consequences

**Good**

- Search does not pay for copies it does not want.
- Cloning is deliberate and visible at the call site, so the cost is where the reader can
  see it.
- One code path. A functional `apply` that internally cloned would still need the mutating
  version underneath.

**Bad**

- A caller that forgets to clone silently loses the previous state. Mitigated by the naming
  and by `apply` returning the same object it was given, so `a = apply(b, x)` makes
  `a is b` — which is at least consistent rather than surprising.
- Not directly usable in a persistent/immutable-history design. Nothing needs that here.

## Measured

Cloning a mid-game state: **~17 µs** with a snapshotted generator, **~1.3 µs** with a
shared one.

The Mersenne Twister snapshot (625 words) is ~92% of the default cost — for a property
rollouts do not want anyway, since sibling rollouts should *diverge*. So
`GameState.clone()` takes an optional `rng`:

```python
child = state.clone(rng=state.rng)   # search: share the stream, 13x cheaper
snap  = state.clone()                # a true point-in-time copy, replays identically
```

The default is the safe one; the fast one is one keyword away.

## Enforced by

`test_apply_accepts_exactly_what_legal_actions_offers` — the single-authority guarantee,
checked by offering every action to a clone and rejecting a sample of everything not
offered.

`test_mutating_a_clone_does_not_touch_the_original`,
`test_a_clone_replays_identically_by_default`,
`test_a_clone_can_share_a_stream_so_rollouts_diverge`,
`test_sharing_the_stream_makes_cloning_much_cheaper`.

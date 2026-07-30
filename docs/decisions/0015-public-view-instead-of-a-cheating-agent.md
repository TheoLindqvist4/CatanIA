# 0015 — Agents see a `PublicView`, not the state

**Status:** accepted · **Date:** 2026-07-30 · **Phase:** 7

## The problem

The heuristic opponent needs to reason about the board: which vertex produces most, what the
opponent already holds, where the robber hurts. A trained policy gets the 1808-float
observation and physically cannot see more. A hand-written heuristic cannot use that vector —
decoding it back into "which tiles touch vertex 23" to write an evaluation function would be
absurd — so it needs the objects.

Handing it `GameState` gives it `state.hands[opponent]` and `state.dev_deck`. Nothing stops a
line of heuristic code from reading them, and such a line would not look wrong: it would look
like a good heuristic. An opponent that quietly cheats is worse than a weak one, because it
teaches the human the wrong lessons and, later, would silently inflate any win rate measured
against it.

## The decision

Agents receive `info["view"]`, a `catan.view.PublicView` — a wrapper with an **explicit
allow-list**:

```python
FORWARDED = frozenset({"board", "vertex_owner", "bank", "robber_tile", ...})

def __getattr__(self, name):
    if name in FORWARDED:
        return getattr(self._state, name)
    raise AttributeError(f"{name!r} is not public — a player may not see it")
```

Hidden things are reachable only through counting methods: `hand_size(p)`,
`dev_card_count(p)`, `dev_deck_size`. `my_hand` returns a **copy**, so an agent cannot edit
the game it is playing.

Two properties follow, and they are the reason for the design:

- **A new field on `GameState` defaults to hidden.** It has to be added to `FORWARDED`
  deliberately, by someone thinking about whether it is public. The failure mode of a
  deny-list — add a field, forget to hide it — cannot happen here.
- **Cheating is impossible rather than tested for.** `view.hands` raises `AttributeError`.
  There is no line of heuristic code that could read an opponent's cards by accident.

The same reasoning as filtering hidden information server-side in `interfaces/web/api.py`
(decision 0014) and as generating the geometry rather than hand-writing it (0001): make the
wrong thing unrepresentable instead of writing a test that hopes to catch it.

## What it costs

265 ns to construct, so the environment builds a fresh one for every agent on every step
(`tests/test_view.py` pins this). Some duplication: `players`, `opponents` and the card
counts are reimplemented rather than forwarded. That is the point — each is a decision about
what a player may know.

An agent driven by hand without a view still works: `HeuristicAgent` falls back to greedy
play rather than crashing.

## Also tested

`PublicView` makes leaks impossible *by construction*, but the argument is only as good as
the wrapper being the sole route in. So `tests/test_heuristics.py` plays a full game and, at
every one of the ~180 decisions, asks the agent again from a state where the opponent's
hidden cards have been rewritten — same hand *size*, different resources; same development
card *count*, different cards; both decks reversed. The move must be identical.

A companion test asserts the scramble actually changes something, so the leak test cannot
pass by mutating nothing.

## What was rejected

- **Trusting the agent.** Works until someone writes `state.hands[opponent]` in a heuristic
  at 1am and it looks reasonable in review.
- **A deny-list of hidden fields.** Wrong default: every new field on `GameState` is public
  until someone remembers.
- **Deep-copying a censored state.** Correct, but copies 54 + 72 + N arrays per agent per
  step, which at ~3,700 steps/sec is a cost paid on every training step for nothing.
- **Making the agent read the observation vector.** What a trained policy will do, and the
  right answer for it. For hand-written code it means re-deriving the board from floats.

## See also

- [0014 — the AI surface](0014-ai-surface.md)
- [`docs/ai-surface.md`](../ai-surface.md)

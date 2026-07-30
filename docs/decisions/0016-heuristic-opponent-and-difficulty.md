# 0016 — A heuristic opponent, with difficulty as noise

**Status:** accepted · **Date:** 2026-07-30 · **Phase:** 7

## The problem

`GreedyAgent` orders *what* to build sensibly — city, then settlement, then development card,
then road — and then picks *where* at random. A person beat it 16–3 on the first try. Random
placement is not a weak strategy so much as no strategy: the two setup settlements decide most
1v1 games, and greedy chooses them by coin flip.

Training a policy was the obvious next step, but it needs a baseline to be measured against,
and "better than random placement" is not a baseline. So: judgement first, learning after.

## The decision

`catan/heuristics.py` — pure functions scoring positions — plus `HeuristicAgent`, which uses
them to answer *where*.

### Value is marginal

The one idea that matters. A settlement is not worth the sum of its tiles; it is worth what
those tiles add to what you **already produce**:

```python
value += rate / (DIMINISH + have[resource])       # DIMINISH = 0.25
```

At 0.25, a first source of a resource is worth about 4×, a second about 1.3×. A spot covering
three resources you lack beats a richer one covering a fourth wheat. That single line is most
of what separates a plausible opening from a bad one, and it is why the agent does not stack
its two settlements on the same two numbers.

Resources are not equal either. Wheat (1.20) and ore (1.15) build cities, which are the
efficient route to 15 points — two points on ground already held. Brick (1.10) is scarce, only
three tiles, and gates early expansion. Sheep (0.80) buys the least.

### One step of lookahead, no search

A road is worth *half* the best settlement spot it brings within reach. Half, because reaching
a spot is not owning it; and a road that reaches nothing scores zero, which is what stops the
agent laying track across the board for its own sake.

No game tree. With a shuffled development deck, a shuffled dice deck and hidden hands, search
would need belief sampling to mean anything — which is Phase 8's problem, and is better solved
by learning than by a deeper hand-written evaluation.

### Difficulty is noise, not amputation

One knob: Gaussian noise added to every evaluation before comparing options.

```python
DIFFICULTY = {"easy": 0.9, "medium": 0.35, "hard": 0.0}
```

An easy opponent **misjudges which spot is best** — which is how a weaker human plays. The
alternative, taking rules away (no cities, never plays knights), produces an opponent that
behaves in ways no player would, and teaches the human nothing.

## Measured

60 games per pairing, seats swapped every other game, `seed=500`:

| | wins | rate | truncated |
|---|---|---|---|
| hard vs random | 59 – 1 | 98.3% | 0 |
| hard vs greedy | 58 – 2 | 96.7% | 0 |
| medium vs greedy | 56 – 2 | 96.6% | 2 |
| easy vs greedy | 55 – 5 | 91.7% | 0 |
| hard vs medium | 42 – 15 | 73.7% | 3 |
| hard vs easy | 48 – 12 | 80.0% | 0 |
| medium vs easy | 41 – 18 | 69.5% | 1 |
| *(greedy vs random, for scale)* | 45 – 15 | 75.0% | 0 |

The ladder is monotone — hard beats medium beats easy — which is what a difficulty setting has
to be to mean anything.

**One honest limit.** Noise degrades *position* choice only; the action-type priority chain is
untouched, so even at extreme noise the agent still builds cities before roads. Past noise ≈1.5
the strength stops falling — measured 16.7% at 0.8, 3.3% at 1.5, 3.4% at 12.0 against hard. So
`easy` cannot be made much weaker than roughly greedy strength by this knob alone. `greedy` and
`random` stay in the dropdown as the rungs below it.

## A bug this found

`robber_damage` iterated `VERTEX_TILES[tile]` — indexing the *vertex*→tiles table with a tile
id. It returned tiles where vertices were expected, so robber placement was being scored
against an unrelated set of positions. Both tables are keyed by integers, so nothing complained.
Writing the test that a city suffers twice a settlement's loss is what surfaced it.

Fixing it moved `easy` from 82.5% to 91.7% against greedy: the robber is worth more than the
difference between good and bad *settlement* placement at that noise level.

## What was rejected

- **Reusing the encoder's pip-potential feature** (the roadmap suggested it). It is absolute
  production, not marginal, and marginal is the whole idea. The encoder computes what a network
  should *see*; heuristics compute what a player should *think*. Different jobs.
- **Expectimax over the dice.** ~700 µs per node and a branching factor in the hundreds, before
  the hidden information makes the values wrong anyway.
- **Difficulty by rule removal.** Produces a bad opponent, not an easy one.

## See also

- [0015 — agents see a `PublicView`](0015-public-view-instead-of-a-cheating-agent.md) — the
  agent cannot read the opponent's cards, and a leak test proves the wrapper is the only way in.
- [0014 — the AI surface](0014-ai-surface.md)

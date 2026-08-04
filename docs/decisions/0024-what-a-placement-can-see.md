# 24. What a placement can see

Date: 2026-08-04
Status: accepted

## Context

A knowledgeable player watched the AlphaZero champion and reported two things: its opening
settlements are poor, and it never uses harbours. Both turned out to be measurable, and one of
them turned out to be a gap in the observation rather than in the policy.

**Measured first.** 30 games, champion at 32 simulations against the heuristic:

| | AlphaZero | heuristic |
|---|---:|---:|
| pips left on the table at placement | **0.005** | 0.024 |
| distinct resources per opening spot | 2.48 / 3 | 2.58 / 3 |
| harbours owned at game end | **0.87** | 1.43 |
| 2:1 trades per 30 games | **26** | 95 |
| 4:1 trades | 283 | 268 |
| development cards per game | **13.5** | 4.9 |

The agent is not placing at random. It places to **maximise total pips**, essentially perfectly
— and that is the only placement signal the observation gives it. It also trades heavily at the
worst available rate while owning half the harbours the heuristic does, and it has converged
hard onto the ore-wheat-sheep development-card strategy.

## What the observation actually contained

Per vertex, 16 floats: owner one-hot, city flag, **which harbour is here** (7), **pip
potential** (1), buildable, reachable-by-my-roads.

- `pip potential` is the summed odds of the adjacent tiles, **resource-blind**. A corner with
  an 8 on ore, a 6 on wheat and a 5 on sheep and a corner with three sheep totalling the same
  are *the same number*. The structured network recovers a little by averaging adjacent tile
  rows, but the pairing — which resource carries which number — is lost, and `CLAUDE.md` has
  called this "the largest known gap" since before this work.
- A vertex knew whether it **was** a harbour. It had no way to know one was two roads away,
  which is the form the information is used in: you settle *near* a port and build toward it.
- The player block encoded trade **rates** but not production **rates**. Nothing said "I make
  no brick", which is what "keep the hand actionable" means.

So the agent maximised the one placement number it had. That is not a policy failure.

## Decision

Four additions, all **public information**, all appended so every existing offset is unchanged.
`encoder.SIZE` 1884 → **2503**.

| where | floats | what |
|---|---:|---|
| per vertex | 5 | expected cards per roll **of each resource** |
| per vertex | 6 | nearness of the closest harbour of each kind, 1 here → 0 at six roads |
| per player | 5 | expected cards per roll of each resource, from their buildings |
| global | 5 | what the whole board makes of each resource — scarcity |

**Information, not advice.** This is the distinction that mattered: nothing here tells the
agent that ore-wheat-sheep is a strategy, that a 2:1 port is worth building toward, or that a
balanced opening is good. It says *what is on the board*. The two strategies a person would
name are computable from these numbers, and the agent is free to discover that they exist,
weight them differently, or find something else. A hand-written "port value" term would have
been advice, and would have baked in whatever the author believed.

**Where each lives.** All of it except per-player production is board-static, so it is computed
once per `Board` and cached in `encoder._static_template` — no per-encode cost. Per-player
production is derived from ownership and costs one pass over the vertices.

**One definition each.** `Board.expected_production`, `Board.resource_scarcity`,
`Board.harbour_distances` and `rules.production_rates` are the only implementations;
the encoder and `training.alphazero.study` both call them. The study had its own copy of the
pip arithmetic for about an hour, which is exactly how two authorities start.

**Leak check.** Every input is the board layout, or ownership, or a public count. The existing
scramble-at-constant-public-counts tests cover it unchanged, and pass.

## The graft, generalised

Changing `encoder.SIZE` used to invalidate every checkpoint, and `CLAUDE.md` records that the
promotion gate is ungated by construction when no champion loads. Record 0023 added a graft for
a *non-positional* block; this change grows the vertex and player rows, which the old one
explicitly refused.

`training/alphazero/layouts.py` now records the observation's block shapes, and
`network.graft` rebuilds every observation-width layer by gathering old columns and leaving new
ones at **zero** — `tile_embed`, `vertex_embed`, `road_embed` and `context_mlp.0`. A zero
column contributes nothing, so a grafted network computes exactly the function it did before.

- New checkpoints record their own layout, so they are self-describing forever.
- Older ones are looked up in `layouts.HISTORICAL`, a table of the three sizes this repository
  has shipped. **Never edit an entry**: it is a statement about a file that already exists.
- An unknown size with no recorded layout **raises**. A checkpoint bent into the wrong shape
  loads, runs, and plays nonsense.

**Verified, not asserted.** The champion scored 74.7% [70.2, 78.8] over 400 games before the
change and **75.6% [69.1, 81.2]** over 200 games after being grafted onto the new observation.

`champion.load` now grafts automatically instead of returning `None`, so an observation change
is no longer a breaking change for whoever is playing. That is the failure recorded in 0023 —
both interfaces silently lost their learned opponent — made structurally impossible.

## What the openings study says, honestly

`training.alphazero.study` plays games and records the opening against the outcome. At 40
games it showed a clean monotonic result: the more pip-maximising and the more ore-heavy the
opening, the worse it did. **At 240 games that did not hold.**

| ore-wheat-sheep share | 40 games | 240 games |
|---|---:|---:|
| lowest band | 100% | 69.5% |
| second | 81.8% | **83.0%** |
| third | 63.6% | 65.5% |
| highest | 60.0% | 66.1% |

The 240-game picture is a *hump* — balanced openings best, both extremes worse — not a slope,
and every band is within about ±11 points of the others. **So "balanced openings win" is
suggestive and not established**, and the 40-game version was the "confidence intervals, not
point estimates" rule in `CLAUDE.md` catching the author of this record rather than somebody
else.

What *is* established, because it was measured twice on independent samples: the agent owns
half the harbours the heuristic does and makes a quarter as many 2:1 trades. The case for
these features does not rest on the win-rate study at all — it rests on the observation
provably not containing the information.

## Consequences

- **Nothing has been retrained yet.** The features exist and are encoded; a network has to be
  trained to use them. The champion plays exactly as before until then.
- The observation is 33% larger. The static template absorbs most of the cost, but the
  per-encode float16 replay row grows with it — a 220,000-position buffer is now about 1.1 GB.
- `catan.encoder.VERTEX_OFFSETS` now names the fields in a vertex row. A test computed
  `VERTEX_FEATURES - 2` for the buildability flags, which was right until something was
  appended and then silently wrong.
- Two visualisation tools were added alongside, both read-only and in their own processes so
  they cannot disturb a run: `training.alphazero.dashboard` (a self-contained HTML page) and a
  watch mode in the web interface that plays two agents against each other at five decisions a
  second.

# Documentation

Reference material for CatanIA. The goal of the project is a rules-correct Catan engine
whose entire state is machine-readable, so an agent can be trained on it; the playable
interface consumes that engine rather than being part of it.

For *what happens next*, see [`../ROADMAP.md`](../ROADMAP.md). This folder is for what has
been established: how things work, and why they were decided that way.

## Contents

| Document | What it covers |
|---|---|
| [board-geometry.md](board-geometry.md) | How the 19 tiles, 54 settlement positions and 72 road positions are numbered, the coordinate lattice, and the coastline. **Start here** to understand the board. |
| [engine.md](engine.md) | How the engine fits together: the layers, the state model, and how to drive a game. **Start here** to use the code. |
| [audit-2026-07-30.md](audit-2026-07-30.md) | Full audit of the codebase at commit `e0f91a3`: verified bugs, missing rules, AI-readiness blockers, and what Phase 0 measured. |

## Decisions

Numbered, immutable records. A decision is superseded by a new record, not edited away —
the reasoning at the time is the point.

| # | Decision | Status |
|---|---|---|
| [0001](decisions/0001-generate-geometry-from-the-row-structure.md) | Generate the board geometry from the row structure | accepted |
| [0002](decisions/0002-flat-tile-arrays-not-ragged-rows.md) | Store tiles in flat arrays, not ragged rows | accepted |
| [0003](decisions/0003-io-free-core-and-injected-randomness.md) | I/O-free core, injected randomness, instance state | accepted |
| [0004](decisions/0004-desert-as-the-seven-tile.md) | Model the desert as the tile numbered 7 | accepted |
| [0005](decisions/0005-balanced-board-generation.md) | Balanced board generation (a house rule) | accepted, revisit Phase 2 |
| [0006](decisions/0006-longest-road-intersection-reuse.md) | Longest road: strict simple path, and an opponent's building breaks a road | accepted |
| [0007](decisions/0007-package-layout-rewrite-vs-incremental.md) | Build `catan/` fresh | accepted |
| [0008](decisions/0008-mutating-apply-plus-clone.md) | `apply` mutates; copy with `clone()` | accepted |
| [0009](decisions/0009-immutable-board-mutable-state.md) | The board is immutable; mutable state lives in `GameState` | accepted |

## Worth knowing

Consequences that are easy to trip over:

- **A closed loop of six roads counts as five**
  ([0006](decisions/0006-longest-road-intersection-reuse.md)). A simple path can visit a
  hex ring's six vertices but only five of its roads. Consistent with the strict ruling, but
  not obvious from it.
- **Double-production vertices cannot exist**
  ([0005](decisions/0005-balanced-board-generation.md)). The "double 6" spot of official
  Catan does not occur on this board, so an agent trained here never learns to value one.
- **Search should clone with a shared stream**
  ([0008](decisions/0008-mutating-apply-plus-clone.md)). `state.clone(rng=state.rng)` is
  13× cheaper than the default snapshot, and divergent rollouts are what sampling wants.
- **Harbours attach to `PERIMETER_VERTICES`, not `CORNER_VERTICES`**
  ([board-geometry.md](board-geometry.md#6-coastline)). 30 versus 18 — the 12 notch
  vertices are on the coast too.
- **Phase 1 games usually stall — only 4 of 40 random games reach 10 points**
  ([engine.md](engine.md#-phase-1-games-usually-stall-and-trading-is-why)). A settlement
  needs four different resources and most players' buildings reach only three, so they end
  up holding 100+ useless cards. The engine is not deadlocked, but a win-based reward is
  almost always zero, so **this is not yet a trainable environment.** Phase 2's trading is
  the unblocking item.

## Deviations from official Catan

Kept together so they are not mistaken for bugs:

- **[0004](decisions/0004-desert-as-the-seven-tile.md)** — the desert carries a 7 token.
  Equivalent for production; official Catan gives it no token.
- **[0005](decisions/0005-balanced-board-generation.md)** — balanced generation forbids
  equal adjacent numbers and 6/8, 2/12 pairs. Official Catan uses a fixed spiral. This has
  a real strategic consequence: **double-production vertices cannot occur**, so the "double
  6" spot does not exist on this board.

## Conventions

- The ids in [`../Images/`](../Images) are the contract for board numbering.
  `tests/test_topology.py` transcribes them independently and asserts the generated tables
  match, so the drawings and the code cannot diverge silently.
- Tests are the enforcement mechanism for decisions. Each record lists the tests that pin
  it; changing the behaviour means changing a named test, deliberately.

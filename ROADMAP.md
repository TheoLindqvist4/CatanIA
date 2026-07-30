# CatanIA — Roadmap

**Goal.** A complete, rules-correct Catan engine whose entire state is machine-readable, so a
reinforcement-learning agent can be trained on it. A playable human interface is a *consumer* of
that engine, never a part of it.

Ordering principle: **engine first, rules second, AI surface third, UI last.** Every phase leaves
the repo in a working, tested state.

This file is the *plan*. For what has already been established — how the geometry works, the
initial audit, and the reasoning behind each decision taken — see **[`docs/`](docs/README.md)**.

---

## Status

| Phase | Scope | State |
|---|---|---|
| **0** | Unblock: correctness, performance, determinism, tests | ✅ **done** |
| **1** | Real state model + economy | ✅ **done** |
| **2** | Complete the rules | ✅ **done** |
| **3** | AI surface (action space, observations, env) | ⬜ next |
| **4** | Interfaces (CLI, web API) | ⬜ not started |

417 tests. `python -m pytest -m "not slow"` runs the fast ones in ~5s.

---

## Target architecture

```
catan/
  topology.py      # ✅ Geometry GENERATED from ROW_LENGTHS, frozen at import. O(1) lookups.
  resources.py     # ✅ the five resources; costs as fixed-width vectors
  board.py         # ✅ one layout: numbers, resources, production index. IMMUTABLE.
  state.py         # ✅ GameState: vertex_owner[54], vertex_piece[54], edge_owner[72],
                   #    hands, supplies, phase, turn.  clone() / __eq__
  actions.py       # ✅ Action = (type, position, extra)
  rules.py         # ✅ legal_actions / apply — the single legality authority
  encoder.py       #    Phase 3: to_vector(state, perspective_player), hidden-info masked
  env.py           #    Phase 3: Gymnasium-style reset(seed) / step(action)
  agents/          #    Phase 3: random, heuristic, then the network
interfaces/
  cli.py           #    Phase 4: the ONLY place print()/input() may appear
  api.py           #    Phase 4: adapter for the FullStackCatan front end
tests/

Board.py  Player.py  Deck.py  Dice.py  Game_2_players.py
                   # DEPRECATED legacy engine. Kept only so `python Game_2_players.py`
                   # still runs; deleted in Phase 4 with interfaces/cli.py. No new work.
```

See [`docs/engine.md`](docs/engine.md) for how the layers fit together and how to drive a
game.

**Non-negotiables**

1. `rules.py` is pure and I/O-free. `legal_actions` is the *single* legality authority — no
   duplicated `check_*` logic in a `Game` class.
2. Per-game `random.Random(seed)`. Never the global `random` module.
3. **Perspective rotation** in the encoder: "me" is always player index 0, so one network plays
   every seat.
4. **Hidden-information masking**: opponents expose hand *size* and played dev cards, never
   contents.
5. `GameState.clone()` must be cheap — MCTS and self-play both depend on it.

---

## Phase 0 — Unblock ✅

Prerequisite work that everything else was blocked on.

- [x] **Generate the geometry instead of hand-writing it.** The whole board now derives from one
      line — `ROW_LENGTHS = (3, 4, 5, 4, 3)`. Hex centres go on an integer lattice, corners are
      deduplicated by exact equality, and ids are assigned by position: vertices by `(y, x)`, roads
      by `(min y, x₁+x₂)`. ~440 lines of hand-maintained tables → 0, and the generated ids are
      *identical* to the ones drawn in `Images/`, pinned by tests. This structurally eliminates the
      bug found at old `Board.py:324` / `Board.py:373` (roads 2 and 51 were each missing a
      neighbour, corrupting longest-road and legal-move enumeration).
      → [decision 0001](docs/decisions/0001-generate-geometry-from-the-row-structure.md),
      [docs/board-geometry.md](docs/board-geometry.md)
- [x] **Freeze the topology.** Lookups were rebuilding a 54- or 72-entry dict literal on every
      call (~5.5 µs). Now module-level tuples indexed in O(1) with zero allocation.
- [x] **Flat tile arrays instead of ragged rows.** `Board` indexes tiles by id; the 3-4-5-4-3 row
      view is rebuilt on demand for display only. Number placement no longer uses a half-filled
      row's length as an implicit progress marker.
      → [decision 0002](docs/decisions/0002-flat-tile-arrays-not-ragged-rows.md)
- [x] **Real production structure.** `positions_grid` (`{vertex: [{number: resource}, …]}`, which
      discarded tile identity) is replaced by
      `vertex_production: vertex → tuple[Production(tile, number, resource)]` plus a payout index
      keyed by roll. A dice payout is now one dict lookup rather than a 54-vertex scan, and a 7
      pays nobody *structurally* because the desert is excluded from the index.
- [x] **Coastline derived.** `COASTAL_ROADS` (30), `PERIMETER_VERTICES` (30) and `CORNER_VERTICES`
      (18) are named apart so Phase 2 harbours cannot attach to the wrong set.
- [x] **Remove I/O from the core.** No `print` in `Board.__init__`; `display_board()` returns a
      string; no module-level game instantiation; `if __name__ == "__main__"` guard.
- [x] **Determinism.** Injectable `random.Random` through `Board`, `Dice`, `Game`. Bounded
      board-generation retries. All legal-move enumerations return sorted lists.
- [x] **Instance state, not class state.** `player_order`, `turn_number`, `dice_value` and the deck
      counts were class attributes shared across every instance — fatal for parallel self-play.
- [x] **Ids raise instead of returning a sentinel.** The adjacency lookups used to return the
      *string* `"The number must be between 1 and 54."` on bad input, so a caller iterating the
      result silently looped over its characters.
- [x] **Test suite.** 129 tests covering the numbering drawn in `Images/`, geometry invariants,
      the lattice, the coastline, board-generation invariants, determinism, I/O-freedom, two
      performance floors, and regressions pinning roads 2/51 and the longest-road branching case.
- [x] **`docs/`** — the [audit](docs/audit-2026-07-30.md), a
      [geometry reference](docs/board-geometry.md), and numbered
      [decision records](docs/decisions/).

Three items were pulled forward from Phase 1 because they sat inside code being rewritten anyway,
and leaving a known-wrong version in place was worse than the small scope increase:

- [x] `get_available_road_from_settlement` now filters against occupancy, and the setup-phase road
      loop no longer `break`s on a rejected choice. Together these were costing a player their
      starting road: the offered list included roads that were already built, and picking one
      consumed the prompt without placing anything.
- [x] The resource supply in `Deck` was 21 per type; standard Catan is 19.
- [x] The production data structure (listed under Phase 1 as replacing `positions_grid`), since the
      board's layout code was being rewritten around it.

**Measured effect** (same board layouts, same longest-road answers):

| | before | after | |
|---|---|---|---|
| adjacency lookup | 5,490 ns | 38 ns | 145× |
| board construction | 2,753/s | 6,637/s | 2.4× |
| `find_longest_path` (8 roads) | 319 µs | 81 µs | 3.9× |
| hand-maintained geometry data | ~440 lines | 0 | — |

`find_longest_path` is still an exponential search; memoising it is Phase 3 work, once
`GameState` exists to key a cache on.

## Phase 1 — Real state model + economy ✅

The blocking issue was that the board could not say *who owns what*: `settlement_positions` was one
flat "available" list, so **empty**, **blocked by the distance rule** and **occupied by player N**
all collapsed into a single bit.

- [x] **`catan/` package** built fresh, per
      [decision 0007](docs/decisions/0007-package-layout-rewrite-vs-incremental.md).
      `topology.py` moved in unchanged and is the only module shared with the legacy engine —
      deliberately not copied, since two copies of the geometry would drift.
- [x] **`GameState`** with `vertex_owner[54]`, `vertex_piece[54]`, `edge_owner[72]`, hands and
      supplies as parallel arrays. Availability is *derived* from ownership; nothing is deleted.
- [x] **The board is immutable**, so `clone()` shares it —
      [decision 0009](docs/decisions/0009-immutable-board-mutable-state.md).
- [x] **One legality authority.** `legal_actions` and `apply` share the same `can_*` predicates,
      so a move can never be offered by one and rejected by the other.
      `apply` mutates; `clone()` is the explicit copy —
      [decision 0008](docs/decisions/0008-mutating-apply-plus-clone.md).
- [x] **Resources charged on build.** One brick + one wood now builds exactly one road.
- [x] **Connectivity enforced**, including the rule that a road may not be extended through an
      opponent's building.
- [x] **Cities**: 2 wheat + 3 ore, must upgrade an own settlement, double production, 2 VP, and the
      settlement returns to the supply.
- [x] **Victory points + the win at 10**, derived from the board so they cannot drift.
- [x] **Setup** as an atomic action per placement, snake order, and the second settlement's payout.
- [x] **Longest road** under [decision 0006](docs/decisions/0006-longest-road-intersection-reuse.md):
      strict simple path, and an opponent's building breaks a chain. The *award* is Phase 2.
- [x] **2–4 players.**
- [x] **Naming normalised**: `'Weat'` is gone, resources are a `Resource` IntEnum, hands are
      vectors rather than dicts keyed by misspelled strings.
- [x] **147 new tests**, including whole-game fuzzing that re-checks every state invariant after
      every mutation.

**Found while building it:** `Action.__repr__` crashed on a bogus action type — inside the error
message raised *because* the type was bogus. And `GameState.__eq__` required board *identity*, so
replaying a seed compared unequal; `Board` now has value equality and a hash.

## Phase 2 — Complete the rules ✅

Trading was moved to the front of this phase because Phase 1 measured only **4 of 40** random
games reaching 10 points: a settlement needs four *different* resources, most players' buildings
reach only three, and there was no way to convert a surplus. That made the reward signal
almost always zero.

- [x] **The bank**: 19 cards per resource, with cards conserved — every card is either in the
      bank or in a hand, so the total is always 95. Paying for a build returns the cards.
      Production is bank-limited with the official shortage rule: if the bank cannot cover
      everything owed of a resource, nobody gets any unless exactly one player is owed it.
- [x] **4:1 bank trading**, and **harbours** at 3:1 and 2:1. Nine harbours on coastal roads,
      evenly spaced, both endpoints granting the port; positions fixed, types shuffled by seed —
      [decision 0010](docs/decisions/0010-harbour-placement.md), which records that the
      positions are *not* the official ones.
      → **39 of 40 games now finish**, median 286 turns.
- [x] **The robber**, on `GameState` since it moves during play. Starts on the desert, must
      always move, blocks its tile's production for everyone, and steals one card drawn
      uniformly over the victim's *cards* — not their resource types. Cannot rob you, or anyone
      holding nothing.
- [x] **Discard on 7**: everyone over 7 cards gives up half, rounded down, in seat order from
      the roller. **One card per action**, so the choice stays a small discrete action instead of
      a multiset. The count is fixed when the 7 is rolled — recomputing it from the shrinking
      hand moved the target and stopped discards at 7 cards instead of at half, which is a bug
      this caught.
      → still **40 of 40** random games finish, median 393 turns (up from 286; the robber
      slows the game, as it should).
- [x] **Harbours randomised** per board rather than fixed, after confirming the rules sanction
      it and that no official coastal-edge list is published —
      [decision 0010](docs/decisions/0010-harbour-placement.md). 280 distinct position sets,
      gaps always 3 or 4 roads so they never cluster.
- [x] **Development cards**: the 25-card deck, buying, and both timing rules — one per turn,
      and not the turn you bought it. All five cards, including play *before* rolling, which is
      how a Knight blocks a tile before it produces. Victory Point cards are never played: they
      count while held and stay hidden, so `public_victory_points` exists alongside
      `victory_points` for what an opponent can see.
      → [decision 0012](docs/decisions/0012-development-card-modelling.md)
- [x] **Largest Army** (3+ knights) and **Longest Road** (5+ segments), 2 points each, sharing
      one implementation: a minimum to qualify, sole leadership to take it, holder keeps it on a
      tie. Rechecked after *every* build — a settlement can break an opponent's road and take
      Longest Road off them.
      → **40 of 40** random games finish, median 349 turns. Largest Army is held in 39 and
      Longest Road in 37, so both are contested. Winners' points: buildings 203, VP cards 92,
      army 58, road 50.
- [ ] **Official spiral layout** as an alternative to balanced generation, behind a config flag.
      The house rule makes double-production vertices impossible, so an agent trained only on it
      never learns to value a "double 6" spot —
      [decision 0005](docs/decisions/0005-balanced-board-generation.md).

**Out of scope for this version:** player-to-player trading
([decision 0011](docs/decisions/0011-no-player-to-player-trading.md)). The target is 1v1, where
trading hands resources to the only opponent who can beat you, and an unrestricted offer is a
multiset-for-multiset exchange that does not flatten into a discrete action space. Revisit when
adding 3–4 player training or human play.

## Phase 3 — AI surface ⬜

- [ ] `actions.py`: flat discrete codec on top of the existing `Action`, plus
      `legal_action_mask() -> bool[N]`. e.g. `0` end turn · `1..72` road · `73..126` settlement ·
      `127..180` city · `181` buy dev · dev-card plays · 19 robber placements · discards · trades.
- [ ] `encoder.py`: fixed-length observation, perspective-rotated, hidden-info masked.
- [ ] `env.py`: `reset(seed)` / `step(action) -> (obs, reward, done, info)`, auto-rolling in the
      `ROLL` phase; `clone(rng=state.rng)` for MCTS.
- [ ] **Performance.** `legal_actions` is the bottleneck at ~174 µs (~5,700/s) because it rescans
      72 roads and 54 vertices twice. Track a build frontier incrementally, and memoise longest
      road against a state key. Worth doing once the action space exists to shape it, not before.
- [ ] Random + heuristic baseline agents.
- [ ] Scale `tests/test_selfplay.py` up to the 10k-game harness. The invariant checks are written;
      it is currently 60 games at 1,500 actions.

## Phase 4 — Interfaces ⬜

- [ ] `interfaces/cli.py` over the pure core — the only place `print`/`input` may appear.
- [ ] **Delete the legacy engine** (`Board.py`, `Player.py`, `Deck.py`, `Dice.py`,
      `Game_2_players.py`) once the CLI replaces it as the playable entry point.
- [ ] `interfaces/api.py`; wire up
      [FullStackCatan](https://github.com/TheoLindqvist4/FullStackCatan).

---

## Decisions

All nine records in [`docs/decisions/`](docs/decisions/) are settled. Two were resolved going into
Phase 1: longest road uses a **strict simple path** with an opponent's building breaking a chain
([0006](docs/decisions/0006-longest-road-intersection-reuse.md)), and `catan/` was **built fresh**
([0007](docs/decisions/0007-package-layout-rewrite-vs-incremental.md)).

Consequences that are easy to trip over are collected under
[Worth knowing](docs/README.md#worth-knowing) — including that a closed loop of six roads counts as
five, and that double-production vertices cannot exist on this board.

Deviations from official Catan are recorded in
[`docs/README.md`](docs/README.md#deviations-from-official-catan) so they are not mistaken for
bugs — most importantly that balanced generation makes double-production vertices impossible.

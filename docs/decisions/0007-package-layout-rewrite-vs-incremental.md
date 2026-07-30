# 0007 — Build `catan/` fresh, or refactor the flat modules in place?

**Status:** ✅ **accepted — build `catan/` fresh** · underway in Phase 1

## Context

Phase 1 introduces the real state model: `GameState` with `vertex_owner[54]`,
`vertex_kind[54]`, `edge_owner[72]`, plus a pure `rules.py` where `legal_actions` is the
single legality authority. Phase 3 adds `actions.py`, `encoder.py` and `env.py`.

The current modules are flat at the repo root: `topology.py`, `Board.py`, `Player.py`,
`Deck.py`, `Dice.py`, `Game_2_players.py`. `Game_2_players` still mixes driver, rules,
legality checks and terminal I/O in one class.

Two ways forward.

## Options

### A — Build `catan/` fresh, keep the old modules as reference until parity

```
catan/
  topology.py    # moves unchanged
  board.py  state.py  actions.py  rules.py  encoder.py  env.py
  agents/
interfaces/
  cli.py  api.py
```

- The new code is written against `GameState` from the start, with no accommodation for
  the "flat available-set" model it replaces.
- The old modules keep working, so the demo and the tests stay green throughout and can be
  diffed against the new engine for parity.
- Delete the old ones once parity holds.

### B — Refactor in place

- Change `Board` to hold ownership arrays, then rework `Game_2_players` around it.
- One code path at all times, no duplication.
- But every intermediate commit has to keep a half-migrated `Board` working, and the two
  things most in the way — the flat availability sets ([audit](../audit-2026-07-30.md) B5)
  and the driver/rules tangle — are exactly what has to change.

## Decision

**Option A.** The deciding factor is B5. `settlement_positions` collapses *empty*, *blocked
by the distance rule* and *occupied by player N* into one bit, and
`delete_settlement_position` destroys information rather than recording it. Ownership is not
an addition to that model; it replaces it. Migrating in place would have meant keeping both
models consistent for the duration.

`topology.py` moved into `catan/` **unchanged**, as predicted. It is the only module shared
by the old and new code — there is deliberately no second copy, because two copies of the
geometry would drift, which is the exact failure this project already had once.

## What was built

```
catan/topology.py    moved, unchanged
catan/resources.py   the five resources; costs as fixed-width vectors
catan/board.py       one layout. Immutable -> shareable across clones
catan/state.py       GameState: vertex_owner / vertex_piece / edge_owner, hands, supplies
catan/actions.py     Action = (type, position)
catan/rules.py       legal_actions / apply — the single legality authority
```

Naming normalisation landed with it, as this record anticipated: `'Weat'` is gone, resources
are a `Resource` `IntEnum`, and hands are vectors rather than dicts keyed by misspelled
strings.

## The legacy modules

`Board.py`, `Player.py`, `Deck.py`, `Dice.py` and `Game_2_players.py` stay for now, marked
deprecated in their docstrings, importing the moved `catan.topology`. They keep
`python Game_2_players.py` working — the only human-playable entry point until
`interfaces/cli.py` lands.

They are **deleted in Phase 4** when that CLI arrives. Until then they must not accumulate
new work: new rules go in `catan.rules`. The one change made to them was applying decision
0006's strict ruling, so the repository does not hold two contradictory answers.

## Consequences

**Good**

- The new code has no compatibility shims for the model it replaces.
- The old engine still runs, so the two can be compared.
- `print`/`input` never entered the new package at all.

**Cost**

- Two engines coexist until Phase 4. Mitigated by the deprecation notices and by there
  being exactly one shared module.
- Test files nearly doubled. The Phase 0 geometry tests needed only an import change,
  because they assert against the drawings rather than against module paths.

## Outcome

The plan held. `Board.py`, `Player.py`, `Deck.py`, `Dice.py` and `Game_2_players.py` were
deleted in Phase 4, along with their tests, once `interfaces/cli.py` replaced them as the
playable entry point. They coexisted with `catan/` for three phases and accumulated no new
work in that time beyond applying decision 0006's strict longest-road ruling, which was done
to avoid the repository holding two contradictory answers.

`topology.py` moved into the package unchanged, as predicted, and never needed a second copy.
Git history keeps the old engine if it is ever wanted.

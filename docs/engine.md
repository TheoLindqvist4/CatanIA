# The engine

How `catan/` fits together, and how to drive a game. For the board's numbering see
[board-geometry.md](board-geometry.md).

---

## Layers

Each layer depends only on the ones above it. Nothing performs I/O; nothing touches the
global `random` module.

| Module | Holds | Mutable? |
|---|---|---|
| `catan.topology` | the geometry: 19 tiles, 54 vertices, 72 roads, every incidence relation | no — generated once at import |
| `catan.resources` | the five resources, and what things cost | no |
| `catan.board` | one layout: numbers, resources, production index | **no** — immutable after construction |
| `catan.state` | `GameState`: ownership, hands, supplies, phase, turn | yes — this is the only mutable thing |
| `catan.actions` | `Action = (type, position)` | — |
| `catan.rules` | `legal_actions`, `apply`, `roll_dice`, scoring, longest road | pure functions |

The board/state split is [decision 0009](decisions/0009-immutable-board-mutable-state.md):
because the board never changes, `clone()` shares it by reference and copies only the arrays
that move.

---

## The state model

Parallel arrays indexed by 1-based topology id, slot 0 unused:

```python
state.vertex_owner[20]   # 0 (NO_OWNER) or a player number
state.vertex_piece[20]   # Piece.NONE / SETTLEMENT / CITY
state.edge_owner[30]     # 0 or a player number
state.hands[1]           # [wood, brick, sheep, wheat, ore]
state.settlements_left[1], state.cities_left[1], state.roads_left[1]
```

Arrays rather than sets of positions, because the Phase 3 encoder wants fixed-width vectors
and `clone()` is on the hot path.

**Availability is derived, never stored.** The old engine deleted from an "available" set,
which collapsed three different facts into one bit and recorded no owner
([audit B5](audit-2026-07-30.md#b5--the-board-could-not-say-who-owns-what)). Now:

```python
state.vertex_owner[v] != NO_OWNER          # occupied — and by whom
rules.respects_distance_rule(state, v)     # buildable: v and all neighbours empty
```

so "empty", "blocked by the distance rule" and "occupied by player N" stay distinct.

`hands` and the supply lists are indexed by player, with slot 0 unused, so a player number
indexes directly everywhere.

---

## Phases

```
SETUP_SETTLEMENT ──▶ SETUP_ROAD ──▶ (repeat 2n times) ──▶ ROLL ⇄ BUILD ──▶ GAME_OVER
```

The two setup phases alternate per placement, so **every setup step is one atomic action** —
which is what an RL agent needs. Placement order is a snake: round one in `player_order`,
round two reversed. The second settlement pays out its adjacent tiles immediately.

`ROLL` is not a decision. `legal_actions` returns `[]` there and the driver calls
`roll_dice`, because a dice roll is environment stochasticity, not a move. Phase 3's `env`
will do this automatically.

---

## Driving a game

```python
from catan.state import GameState, Phase
from catan import rules

state = GameState(num_players=3, seed=42)

# Bound the loop. A Phase 1 game can legitimately never reach 10 points — see below.
while state.phase is not Phase.GAME_OVER and state.turn_number < 500:
    if state.phase is Phase.ROLL:
        rules.roll_dice(state)          # environment stochasticity, not a move
        continue
    actions = rules.legal_actions(state)
    if not actions:
        break
    rules.apply(state, pick(actions))   # your agent chooses

print(state.winner, rules.scores(state))
```

Two things to know about `pick`:

* `END_TURN` is always offered during `BUILD`, so a player can never be stuck — but that
  also means an agent that blindly takes `actions[0]` will end every turn and never build.
  Choose deliberately.
* Building is only possible after paying, so early turns often offer nothing but
  `END_TURN`.

`apply` **mutates**. Clone first if you need the old state
([decision 0008](decisions/0008-mutating-apply-plus-clone.md)):

```python
after = rules.apply(state.clone(), action)      # non-destructive
child = state.clone(rng=state.rng)              # search: 13x cheaper, and rollouts diverge
```

`legal_actions` and `apply` share one set of `can_*` predicates, so **a move can never be
offered by one and rejected by the other**. `apply` raises `IllegalAction` for anything else.

---

## What is implemented

✅ Setup with the snake order and the second-settlement payout · the distance rule ·
road connectivity, including *not* building through an opponent's building · resource costs
actually charged · production, with cities yielding double · city upgrades returning the
settlement to the supply · piece limits · victory points and the win at 10 · longest-road
measurement · 2–4 players · full determinism from a seed.

❌ **Not yet** (Phase 2): the robber and 7-handling · discarding above 7 cards · development
cards · Largest Army · the Longest Road *award* (the measurement exists; the 2 points do
not) · harbours · trading of any kind · bank limits.

### ⚠️ Phase 1 games usually stall, and trading is why

Measured over 40 random 2-player games (4,000 actions each): **only 4 reached 10 points.**

The cause is resource coverage. A player's two starting settlements touch at most six
tiles, and across 80 sampled players the number of *distinct* resources reachable from
their buildings was:

| distinct resources | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| players | 4 | 14 | **45** | 13 | 4 |

A settlement costs wood + brick + sheep + wheat — four different resources. Most players can
never assemble that, and with no way to convert a surplus they stop permanently. One
observed end state:

```
player 1: 0 wood, 32 brick, 52 sheep, 29 wheat,  0 ore   -> cannot build anything
player 2: 36 wood, 0 brick, 58 sheep,  0 wheat,  0 ore   -> cannot build anything
player 3: 42 wood, 2 brick,  0 sheep,  0 wheat, 33 ore   -> out of road pieces
```

Player 1 holds 113 cards and is completely stuck.

The engine is not deadlocked — `END_TURN` stays legal forever, which
`test_games_that_stall_do_so_for_a_legitimate_reason` asserts. But it does mean **Phase 1 is
not yet a trainable environment**: a reward signal based on winning is almost always zero.
Phase 2's trading (4:1 bank, harbours, player-to-player) is the unblocking item, not the
robber or dev cards.

---

## Measured

Mid-game, single-threaded:

| | |
|---|---|
| `legal_actions` | ~174 µs (~5,700/s) — **the bottleneck**; scans 72 roads + 54 vertices twice |
| `apply` | negligible beside the above |
| `clone()` snapshot | ~17 µs |
| `clone(rng=state.rng)` | ~1.3 µs |
| `longest_road_length` | ~76 µs (exponential search, ≤15 roads) |
| `victory_points` | ~2.2 µs |
| topology lookup | ~38 ns |

Phase 3 should attack `legal_actions` — tracking a build frontier incrementally instead of
rescanning, and memoising longest road against a state key. Not worth doing before the
action space exists to shape it.

---

## Testing

```sh
python -m pytest                  # everything
python -m pytest -m "not slow"    # skip whole-game fuzzing (~2s)
```

`tests/test_selfplay.py` drives random legal games and asserts the state stays coherent after
**every** mutation: ownership consistent with pieces, the distance rule never violated,
piece accounting adding up (settlements on the board plus supply equals 5, allowing for
cities handing theirs back), no negative hands, no orphaned roads, and phase bookkeeping
sane. It is the cheap version of the 10k-game harness Phase 3 wants.

`tests/helpers.py` holds the shared fixtures. Helpers like `put_building` write straight
into the state, bypassing the rules, so a test can construct a position without playing 40
legal moves to reach it — tests of the *rules* go through `rules.apply`.

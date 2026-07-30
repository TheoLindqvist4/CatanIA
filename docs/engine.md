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
| `catan.actions` | `Action = (type, position, extra)` | — |
| `catan.rules` | `legal_actions`, `apply`, `roll_dice`, scoring, longest road | pure functions |

The board/state split is [decision 0009](decisions/0009-immutable-board-mutable-state.md):
because the board never changes, `clone()` shares it by reference and copies only the arrays
that move.

An `Action` is a `(type, position, extra)` triple. Two plain ints rather than a variable
payload, because Phase 3 has to flatten every action into one discrete index and a fixed
arity makes that a lookup rather than a parse:

| type | `position` | `extra` |
|---|---|---|
| `END_TURN` | — | — |
| `BUILD_ROAD` | road id | — |
| `BUILD_SETTLEMENT` / `BUILD_CITY` | vertex id | — |
| `TRADE_WITH_BANK` | resource given | resource received |
| `MOVE_ROBBER` | tile id | player to rob, 0 for nobody |
| `DISCARD` | resource discarded | — |
| `BUY_DEV_CARD` | — | — |
| `PLAY_KNIGHT` / `PLAY_ROAD_BUILDING` | — | — |
| `PLAY_YEAR_OF_PLENTY` | first resource | second resource (ascending) |
| `PLAY_MONOPOLY` | resource demanded | — |

---

## The state model

Parallel arrays indexed by 1-based topology id, slot 0 unused:

```python
state.vertex_owner[20]   # 0 (NO_OWNER) or a player number
state.vertex_piece[20]   # Piece.NONE / SETTLEMENT / CITY
state.edge_owner[30]     # 0 or a player number
state.hands[1]           # [wood, brick, sheep, wheat, ore]
state.settlements_left[1], state.cities_left[1], state.roads_left[1]
state.bank               # the supply; hands + bank is always 95
state.robber_tile        # 1..19
state.discards_owed[1]   # cards player 1 still owes for the current 7
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
SETUP_SETTLEMENT ⇄ SETUP_ROAD  (2n placements, snake order)
        │
        ▼
      ROLL ─────── anything but a 7 ──────▶ BUILD ──▶ GAME_OVER
        │  ▲                                 ▲
        │  └──────────┐                      │
        └── a 7 ──▶ DISCARD ──▶ MOVE_ROBBER ─┘
                   (0+ players,
                    1 card each)
```

`MOVE_ROBBER` returns to `BUILD` after a 7, but to `ROLL` when a **Knight** sent it there
before the dice — `state.rolled_this_turn` is what decides.

**Every phase asks for exactly one atomic action.** That is why setup alternates
settlement/road per placement rather than asking for a pair, and why discarding is one card
per action rather than a chosen multiset — a multiset does not flatten into a discrete action
space, and one-card steps do.

Setup order is a snake: round one in `player_order`, round two reversed. The second
settlement pays out its adjacent tiles immediately.

`roll_dice` is how you advance out of `ROLL` — a dice roll is environment stochasticity, not
a move, and Phase 3's `env` will call it automatically. But `legal_actions` is **not** empty
in `ROLL`: the rules allow playing a development card before the dice, so it returns those
plays (usually none). A driver must branch on `phase is Phase.ROLL` rather than on
"`legal_actions` came back empty".

`DISCARD` is the one phase where **`current_player` is not the player whose turn it is** —
it is whoever owes cards, which is usually an opponent. Use `state.turn_player` when you
mean the turn holder and `state.current_player` when you mean "who must act now".

---

## Driving a game

```python
from catan.state import GameState, Phase
from catan import rules

state = GameState(num_players=3, seed=42)

# Bound the loop: a game can still fail to reach 10 points, just far less often now.
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

**Every rule of base-game Catan except player trading.** Setup with the snake order and the
second-settlement payout · the distance rule · road connectivity, including *not* building
through an opponent's building · resource costs charged · production, with cities yielding
double and the bank's shortage rule · city upgrades returning the settlement to the supply ·
piece limits · the bank, with cards conserved · 4:1 bank trading and harbours at 3:1 and 2:1 ·
the robber, 7-handling, discarding and stealing · all five development cards with both timing
rules · Largest Army · Longest Road · victory points and the win at 10 · 2–4 players · full
determinism from a seed.

Random 2-player games: **40 of 40** reach 10 points, median 349 turns. Largest Army is held in
39 of them and Longest Road in 37, so both are contested rather than decorative. Winners' points
came from buildings 203, Victory Point cards 92, Largest Army 58, Longest Road 50.

🚫 **Deliberately out of scope**: player-to-player trading
([decision 0011](decisions/0011-no-player-to-player-trading.md)).

⬜ **Next** (Phase 3): the flat action space and legality mask, the observation encoder with
hidden-information masking, and the Gymnasium-style environment.

### Trading is what made games finishable

Before bank trading existed, only **4 of 40** random games reached 10 points. The cause was
resource coverage: a settlement costs wood + brick + sheep + wheat — four *different*
resources — while across 80 sampled players the number of distinct resources reachable from
their buildings was:

| distinct resources | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| players | 4 | 14 | **45** | 13 | 4 |

Most players could never assemble four, and with no way to convert a surplus they stopped
permanently. One observed end state had player 1 holding **113 cards** and unable to build
anything:

```
player 1:  0 wood, 32 brick, 52 sheep, 29 wheat,  0 ore
player 2: 36 wood,  0 brick, 58 sheep,  0 wheat,  0 ore
player 3: 42 wood,  2 brick,  0 sheep,  0 wheat, 33 ore   (out of road pieces)
```

With 4:1 bank trading and harbours: **39 of 40** games now finish, in a median of 286 turns.

A stalled game was never a *deadlock* — `END_TURN` stays legal — and
`test_almost_every_game_now_finishes` still checks that for any game that does not finish.

## Trading and the bank

`state.bank` holds 19 of each resource. **Cards are conserved**: every card is either in the
bank or in a hand, so `sum(hands) + sum(bank) == 95` always
(`test_cards_are_conserved`). Paying for a build returns the cards to the bank; without that
the bank would drain and production would stop.

Production is bank-limited, with the official shortage rule: if the bank cannot cover
everything owed of a resource, **nobody** gets any of it — unless exactly one player is owed
it, in which case they take what remains. So `distribute` tallies the whole payout per
resource before any card moves, and returns `{player: [received per resource]}`.

`trade_rates(state, player)` gives how many of each resource the player must hand over for
one card back:

| | rate |
|---|---|
| no harbour | 4 |
| any generic (3:1) harbour | 3 on everything |
| the matching specific (2:1) harbour | 2 on that resource |

A harbour sits on a coastal *edge*, so a building on **either** endpoint grants it — two spots
to aim for. But those two vertices are adjacent, so the distance rule means the first building
to claim one **excludes the other**: each harbour serves at most one player.

Harbour positions and types are both randomised per board, kept to gaps of 3 or 4 roads so
they never cluster. The rules sanction randomising them, and no official coastal-edge list is
published — [decision 0010](decisions/0010-harbour-placement.md).

There is **no player-to-player trading**, deliberately —
[decision 0011](decisions/0011-no-player-to-player-trading.md). In 1v1 it hands resources to
the only person who can beat you, and an unrestricted offer is a multiset-for-multiset
exchange that does not flatten into a discrete action space.

## The robber

`state.robber_tile` starts on `board.desert_tile` and moves during play, so it lives on the
state, not the shared board ([decision 0009](decisions/0009-immutable-board-mutable-state.md)).

Rolling a **7** pays nobody. Instead:

1. Everyone holding **more than 7** cards discards **half, rounded down** — so 9 cards means
   losing 4 and keeping 5. The count is fixed when the 7 is rolled and stored in
   `state.discards_owed`; recomputing it from the shrinking hand would move the target and
   stop the discards early. Order starts at the roller and follows seat order.
2. The roller then moves the robber. It **must** move — the tile it is on is not a legal
   destination — and may rob one player with a building on the destination tile.

A robbed card is drawn uniformly over the victim's **cards**, not over their resource types,
so a hand of five wood and one ore gives up wood five times in six. You cannot rob yourself,
and cannot rob a player holding nothing — if the destination offers no victim, the action
carries `extra = 0`.

The tile under the robber **produces nothing for anyone** until it moves again.

## Development cards and the awards

A 25-card deck: 14 Knight, 5 Victory Point, 2 each of Road Building, Year of Plenty and
Monopoly. Buying costs sheep + wheat + ore and draws the top card.

Two timing rules: **one card per turn**, and **not the turn you bought it**
(`state.dev_cards_new` tracks this turn's purchases). A card may be played **before rolling**,
which is how a Knight blocks a tile before it produces.

| card | effect |
|---|---|
| Knight | move the robber and steal; counts toward Largest Army |
| Victory Point | **never played** — worth a point while held, and hidden from opponents |
| Road Building | two free roads, as credit (`state.free_roads`); unused credit lapses at end of turn |
| Year of Plenty | two resources from the bank, as a **sorted** pair so one action means one outcome |
| Monopoly | every opponent hands over all of one resource; the bank is untouched |

**Largest Army** (3+ knights) and **Longest Road** (5+ segments) are each worth 2 points and
share one implementation: you need the minimum to qualify, sole leadership to take it from
someone, and the holder keeps it on a tie. If the holder falls behind into a tie, nobody holds
it until someone is clearly ahead.

`update_awards` runs after **every build**, not only after a road — a settlement or city can
*break* an opponent's road and take Longest Road off them.

`victory_points` counts buildings + both awards + Victory Point cards.
`public_victory_points` is the same minus the hidden VP cards, which is what an opponent can
see — the encoder needs both.

Why each of these is shaped the way it is:
[decision 0012](decisions/0012-development-card-modelling.md).

---

## Measured

Mid-game, single-threaded:

Mid-game, with a full hand and a card of each type held:

| | |
|---|---|
| `legal_actions` | ~218 µs (~4,600/s) — **the bottleneck**; 68 actions offered |
| `apply` | negligible beside the above |
| `update_awards` | ~71 µs — almost all of it `longest_road_length`, once per player |
| `longest_road_length` | ~76 µs (exponential search, ≤15 roads) |
| `clone(rng=state.rng)` | ~2.1 µs |
| `clone()` snapshot | ~17 µs |
| `trade_rates` | ~2.8 µs |
| `victory_points` | ~2.2 µs |
| topology lookup | ~38 ns |

**Phase 3's two optimisation targets**, in order:

1. **`longest_road_length`.** It is an exponential path search, and `update_awards` runs it
   per player after *every* build. Memoise it against a road-set key — the answer only
   changes when roads or blocking buildings change.
2. **`legal_actions`.** It rescans 72 roads and 54 vertices from scratch each call. Track a
   build frontier incrementally instead. `can_play_road_building` also scans all 72 roads
   just to answer "is there anywhere to build".

Neither is worth doing before the action space exists to shape the caching around. Adding 20
candidate trades cost nothing measurable, because `trade_rates` is computed once and shared
across all 20 — the same trick applies elsewhere.

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

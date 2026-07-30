# The AI surface

How to train against the engine. For the rules themselves see [engine.md](engine.md); for why
each piece is shaped this way see
[decision 0014](decisions/0014-ai-surface.md).

```
catan/action_space.py    324 flat indices  +  legal_mask(state)
catan/encoder.py         1808-float observation, perspective-rotated, hidden-info masked
catan/env.py             reset(seed) / step(index)
catan/agents.py          random and greedy baselines  +  play_match
```

---

## The loop

```python
from catan.env import CatanEnv

env = CatanEnv(num_players=2)              # ranked 1v1 rules by default
obs, info = env.reset(seed=0)

while not info["done"]:
    action = my_agent(obs, info["mask"])   # an index; must be legal
    obs, reward, terminated, truncated, info = env.step(action)

print(info["winner"], info["scores"])
```

An agent is any callable `(observation, info) -> index`. That is the whole interface — a network
fits it, and so does `RandomAgent`.

### What `info` carries

| key | |
|---|---|
| `player` | **who must act** — not always the turn holder |
| `mask` | `bytearray` of 324 flags, 1 where legal |
| `legal` | the same thing as a list of indices |
| `phase`, `turn`, `last_roll` | where the game is |
| `scores` | true victory points, including hidden cards |
| `public_scores` | what an opponent can see |
| `winner`, `done` | `winner` is `None` if truncated |

**Read `info["player"]`; do not assume turn order.** During a discard the decision belongs to
whoever is over the hand limit, which is usually an opponent. Assuming alternation is the
classic multi-agent environment bug, and the observation is built from *that* player's view.

---

## The action space

324 indices, in contiguous blocks by type — so `mask[SLICES[ActionType.BUILD_ROAD]]` is exactly
the roads:

| block | indices | |
|---|---|---|
| `END_TURN` | 0 | |
| `BUILD_ROAD` | 1–72 | road id |
| `BUILD_SETTLEMENT` | 73–126 | vertex id |
| `BUILD_CITY` | 127–180 | vertex id |
| `TRADE_WITH_BANK` | 181–200 | 20 ordered pairs of distinct resources |
| `MOVE_ROBBER` | 201–295 | 19 tiles × (nobody, or one of four players) |
| `DISCARD` | 296–300 | one per resource |
| `BUY_DEV_CARD` | 301 | |
| `PLAY_KNIGHT` | 302 | |
| `PLAY_ROAD_BUILDING` | 303 | |
| `PLAY_YEAR_OF_PLENTY` | 304–318 | 15 **sorted** pairs, doubles included |
| `PLAY_MONOPOLY` | 319–323 | one per resource |

The size is **independent of the player count**, so weights transfer between 1v1 and 4-player
and evaluation code does not branch.

`encode` / `decode` convert between an `Action` and its index. `encode` raises on an
inexpressible action rather than dropping it — a silent drop would make that move permanently
unreachable, and would look like a policy that simply never learns it.

---

## The observation

1808 floats, always. `LAYOUT` gives the named spans and `SHAPES` the row/column counts, so a
graph or convolutional model can reshape rather than being forced through an MLP:

```python
from catan import encoder

obs = encoder.encode(state, me)
tiles = encoder.block(obs, "tiles")        # 19 rows of 19 features
tiles[3]                                   # tile 4
```

| block | rows × features | contents |
|---|---|---|
| `tiles` | 19 × 19 | resource, number, production odds, robber |
| `vertices` | 54 × 16 | owner, city flag, harbour, pip potential, buildability |
| `roads` | 72 × 6 | owner, reachable-by-me |
| `players` | 4 × 29 | hands and holdings, masked for opponents |
| `global` | 35 | phase, last roll, bank, ruleset, turn bookkeeping |

Every value sits in `[0, 1]`.

### Perspective rotation

`encode(state, me)` puts **me in player slot 0**, opponents following in turn order. So one
network plays every seat, and a position encodes identically whichever player number holds it.
Absent players leave their slot zeroed.

### Hidden information

An observation contains only what that player may see:

| hidden | public instead |
|---|---|
| an opponent's hand *composition* | its size — cards are countable |
| an opponent's dev-card *composition* | how many they hold, and Knights played |
| the dev deck order and contents | how many remain |
| the Balanced Dice deck | nothing |

This is tested by mutating the hidden thing and asserting the observation does not move — swap
an opponent's three Knights for three Victory Points and nothing changes.

Use `rules.public_victory_points` wherever you mean "what an opponent can see"; `scores` in
`info` is the true total and includes hidden cards.

---

## Baselines

```python
from catan.agents import GreedyAgent, RandomAgent, play_match

play_match({1: GreedyAgent(0), 2: RandomAgent(0)}, games=40, seed=100)
# {1: 29, 2: 11, 'truncated': 0}
```

`RandomAgent` is the floor — anything that cannot beat it is broken. `GreedyAgent` picks the
highest-priority *action type* available, with no idea *where* to build, and still wins about
70% against random.

`play_match` **swaps seats every other game**, because Catan's first-player advantage is real
and large; a fixed-seat result measures the seat as much as the agent.

---

## Performance

Per env step, typical mid-game: **~270 µs (~3,700 steps/sec)**, single-threaded.

| | |
|---|---|
| `encoder.encode` | ~250 µs — **the floor** |
| `legal_mask`, poor hand | ~16 µs |
| `legal_mask`, rich hand | ~236 µs |
| `update_awards` | ~1 µs (memoised) |
| `clone(rng=state.rng)` | ~2 µs |

`encode` dominates. Further gains want numpy or incremental updates — worth doing once a
training loop shows it matters, not before.

---

## ⚠️ Search needs to sample hidden state, and does not yet

`state.clone(rng=state.rng)` diverges for plain dice, but **three pieces of hidden state are
copied verbatim** and therefore replay identically:

- `dice_deck` — with Balanced Dice, the next ~24 rolls are already determined
- `dev_deck` — the next purchases are already determined
- opponents' `dev_cards`

That is *correct*: these are hidden, not random. But it means a rollout from a cloned state is
not a sample of the future — it is the same future. **Before building MCTS, reshuffle the unseen
parts**, which is belief-sampling and depends on the algorithm, so it is deliberately left out.
`test_with_balanced_dice_a_clone_replays_the_same_rolls_even_sharing_the_rng` pins the
behaviour so it is not discovered by surprise.

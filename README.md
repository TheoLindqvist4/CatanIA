# CatanIA

A Catan engine built so that a machine can learn to play it, and a person can play against
what it learned.

The engine is the point. It is dependency-free Python, exhaustively tested, and every rule
lives in exactly one place — so an agent and a human are always playing the same game, and
"the interface disagreed with the rules" is not a bug this project can have.

```sh
python -m interfaces.web        # then open http://127.0.0.1:8000
```

![Board](Images/Catan_tile_positions.png)

---

## What is here

| | |
|---|---|
| **A complete Catan implementation** | Ranked 1v1 rules by default: 15 points, hand limit 9, Friendly Robber, Balanced Dice |
| **A playable web interface** | Click the board to build. Painted artwork, resources as cards, full game log |
| **A hand-written opponent** | Positional judgement from marginal value — beats a naive greedy agent 96.7% |
| **A trained opponent** | PPO self-play, warm-started by cloning the heuristic |
| **The machinery to improve it** | 1,884-float observation, 325 discrete actions, parallel rollouts, a promotion gate |
| **753 tests** | Including leak detectors that prove no agent can see hidden information |

## Quick start

```sh
git clone https://github.com/TheoLindqvist4/CatanIA.git
cd CatanIA
python -m interfaces.web                       # play in the browser

python -m interfaces.cli                       # or in the terminal
python -m interfaces.cli --agents hard easy    # or watch two bots
```

The engine needs **no dependencies at all**. Only training needs PyTorch:

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## The three ideas this project is built on

### 1. One source of truth, always

The rules live in `catan/rules.py` and nowhere else. `legal_actions` and `apply` share the
same `can_*` predicates, so an action is legal for exactly one reason. The browser draws what
the server sends and reports clicks; it holds no rules, no board generation, no scoring. The
last time board logic existed in JavaScript it was a second implementation that could
disagree with the engine.

The same principle removed 440 lines of hand-written geometry. `catan/topology.py` generates
every vertex, road and adjacency from one line:

```python
ROW_LENGTHS = (3, 4, 5, 4, 3)
```

The generated ids are **identical** to the ones drawn in `Images/`, which the tests check — so
the diagrams and the code cannot drift apart. Two entries in the old hand-written road table
were wrong, which had been silently corrupting the Longest Road calculation.

### 2. Hidden information is hidden by construction, not by care

An agent receives a `PublicView` with an explicit allow-list. Reading an opponent's hand
raises `AttributeError`. A new field on `GameState` is invisible to agents until someone adds
it deliberately — the opposite of a deny-list, where forgetting once leaks forever.

The observation vector, the web responses and the game log are all filtered the same way, and
each has a **leak test**: rewrite the opponent's hidden cards at constant public counts, and
demand that nothing observable changes.

### 3. Measure it, or do not claim it

Every performance claim in this repository has a number behind it, and several turned out to
be the opposite of what seemed obvious. The records in `docs/decisions/` exist so the
reasoning survives — including the things that did not work.

---

## The opponents

| name | what it is |
|---|---|
| `hard` / `medium` / `easy` | The heuristic, with noise added to its evaluations. Difficulty is *misjudgement*, not amputated rules |
| `greedy` | Sensible build order, random placement |
| `random` | Uniform over legal moves |
| `learned` | The trained champion, when one is installed |

The heuristic's central idea is **marginal value**: a settlement is worth what its tiles add
to what you already produce, not the sum of its pips. A third wheat is worth far less than a
first ore.

Its resource weights are tuned for **two-player, 15-point** play, which inverts four-player
folklore. Competitive 1v1 data on this exact ruleset gives the win rate for a player who
starts with no production of a resource:

```
brick 36%    wood 40%    sheep 42%    ore 43%    wheat 49%      (50% = even)
```

Missing brick is the worst thing that can happen to an opening; missing wheat is nearly free.
With no player-to-player trading, a resource you do not produce costs 4:1 at the bank, so
*expansion* is what gates a 15-point run.

---

## Training

```sh
python -m training.clone --games 300 --net structured     # imitate the heuristic, ~4 min
python -m training.train --resume checkpoints/cloned.pt --workers 12 --lr 3e-5
python -m training.champion promote checkpoints/best.pt   # only if measurably better
```

**PPO, not AlphaZero.** Search needs a state you can roll forward, and `clone()` copies the
development deck, the dice deck and opponents' hands verbatim — so a rollout replays the same
future instead of sampling one. Belief sampling is the prerequisite, and it is not built.

**Cloning first.** Self-play from random spends millions of steps rediscovering things
`catan/heuristics.py` states outright. Cloning reaches useful play in four minutes.

**The network knows the board has a shape.** `training/structured_net.py` shares weights
across all 54 vertices, 72 roads and 19 tiles, and produces per-position logits from each
position's own embedding. Against a flat MLP on the same data: held-out agreement 69.6% →
**80.3%**, overfitting gap 13.9 → **2.2 points**, with 7.3x fewer parameters.

### The champion, and why training cannot break your game

```
checkpoints/   scratch. A run owns it and rewrites it. Not in git.
models/        the champion. Changes only by promotion. In git.
```

The interfaces read `models/champion.pt` and nothing else, so a fine-tune in progress cannot
disturb a game in progress. Promotion is earned: a candidate plays 400 games against the
reigning champion and is refused unless the Wilson lower bound clears 50%. It is also checked
against the fixed heuristic, so a policy that beat the champion by learning its habits while
getting worse at the game is rejected — self-play is non-transitive and that is where it
shows.

This is not theoretical. The gate has already refused a finished training run that scored
48.2% against the champion.

### Recorded games

Games you play in the browser are written to `games/`. The record is the **seed and the move
indices**, which is complete rather than a summary: the engine is deterministic, so replaying
reproduces the board, the decks, every roll and every observation. Each decision also keeps
*what else was on offer*, because "it had fourteen options and chose that one" is the question
a lopsided game needs answered.

```sh
python -m interfaces.web.recorder --margin 5 --verify    # the lopsided ones
```

Only games a person actually played are recorded — tests and scripts drive the same code and
leave no trace.

---

## Layout

```
catan/                 the engine — no dependencies
  topology.py            geometry, generated from ROW_LENGTHS
  board.py               one immutable layout
  state.py               everything that changes during a game
  rules.py               legal_actions / apply — the only legality authority
  action_space.py        325 flat indices + a legality mask
  encoder.py             the 1,884-float observation
  view.py                PublicView — what a player may see
  heuristics.py          position evaluation
  agents.py              the baseline agents and a match harness
  env.py                 reset / step

interfaces/            the only parts that display anything
  render.py              board -> PNG
  cli.py                 play or watch in a terminal
  web/                   the browser game, the recorder, a stdlib HTTP server

training/              the only package that imports PyTorch
  net.py structured_net.py    the policy/value networks
  rollout.py ppo.py pool.py   self-play, the update, the opponent pool
  clone.py                    warm start by imitating the heuristic
  champion.py                 the model the game plays, and the promotion gate

docs/decisions/        22 records of why things are the way they are
```

---

## Things that turned out to be the opposite of obvious

Each cost real time to discover, and all are written up in `docs/decisions/`.

- **The opening evaluator was a pip count in disguise.** `settlement_value` never accumulated
  within a vertex, so with an empty hand it collapsed to exactly 4x weighted pips — identical
  on 54 of 54 vertices. A spot with three wheat tiles rated as highly as one with wheat, ore
  and brick.
- **Fixing it won no more games.** The road threshold sat at the 88th percentile of road
  values: on 85.2% of decisions where a road was legal, *every* option was below it. The agent
  could not expand, so a better opening had nothing to express. Fixing both: **70.7%** against
  the old agent, and truncated games fell from ~90 per 800 to 8.
- **The winner is always the player who just acted**, so `step()`'s reward is always `+1` and
  the loser's never arrives. A learner that consumed it would train on winners only, its
  critic would converge to `V = 1`, and nothing would crash.
- **`encoder.encode` was 57% of training time**, recomputing per-vertex harbours and pip
  potential for a board that had not changed in 14,000 calls.
- **A `\b` inside a JavaScript template literal is a backspace**, not a word boundary — a
  pattern that matches nothing while reading as though it should work.
- **1-ply lookahead does not help.** Leak-safe and correct, and 53.4% against 52.2% over 800
  games. Recorded because an unwritten negative result gets re-attempted.

---

## Where it stands

The engine and both interfaces are complete and tested.

The heuristic recently got substantially stronger — 70.7% against its previous self over 800
games — and that moved the yardstick. The champion, trained against the *old* heuristic, now
scores **55.0% against the new one** with the interval [49.3, 60.5] straddling 50%: it is no
longer clearly ahead of the hand-written opponent it used to beat comfortably. A retrain from
the improved teacher is under way, and the promotion gate decides whether it replaces the
champion.

The lesson worth carrying: **win rates recorded against the heuristic before and after that
change are not comparable.** Any claim of the form "the bot reached X%" has to say which
baseline it was measured against.

What is known to be missing, in order of expected value:

1. **Which numbers a vertex touches.** The observation gives an aggregate "pip potential", so
   "an 8 on ore" is blended with the two tiles beside it.
2. **Roads have one step of lookahead**, no plan. There is no notion of a route.
3. **Belief sampling.** Every remaining search idea needs it.

Build costs used to head that list — the observation said nothing about what a road cost, and
affordability was inferred only from which actions happened to be legal. The `affordability`
block now encodes how far the hand is from each purchase and what closing the gap would cost
at the bank, which is the part that actually varies
([decision 0022](docs/decisions/0022-affordability-features.md)).

See [`ROADMAP.md`](ROADMAP.md) for the phase history, [`CLAUDE.md`](CLAUDE.md) for working
notes, and [`docs/`](docs/) for the decision records.

## Tests

```sh
python -m pytest tests -q       # 753 tests, about 90 seconds
```

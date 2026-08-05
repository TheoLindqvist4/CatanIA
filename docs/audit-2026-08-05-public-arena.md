# Audit — 5 August 2026: a public arena

What it would take to let strangers submit agents, rank them, and let people play the best
of them in a browser.

State of the codebase at commit `b1e285e` (branch `alphazero-self-play`), working tree
carrying only the resource-card art change. Line references point at that commit.

Findings were verified by execution. Every containment claim below was re-tested by writing
the exploit and running it; every capacity number was measured rather than read off a
docstring, which turned out to matter — see [§6](#6-measurements).

**Verdict.** The tournament is largely built and the boundary is not. `arena.compete` is
already a reproducible, seat-swapped, worker-count-independent match runner, and
`champion.promote` is already a Wilson-gated ladder — together that is most of a leaderboard.
What does not exist is any way to run code you did not write. An agent handed `info` today
holds the opponent's hand, the development deck and the next dice, and that is not fixable
in-process. Beneath it sits a problem that no submission format escapes: the hidden state is
a public function of a low-entropy seed whose values are constants in the source.

⚠️ [`ROADMAP.md`](../ROADMAP.md) currently scopes this out in writing — *"Multiplayer over a
network. One human against the AI, locally. Accounts, lobbies and matchmaking are a different
project."* This audit does not overturn that. It is the survey you would want before deciding
whether to.

---

## Contents

1. [What already exists](#1-what-already-exists)
2. [Blockers, verified by execution](#2-blockers-verified-by-execution)
3. [The seed is the whole hidden state](#3-the-seed-is-the-whole-hidden-state)
4. [The engine as a competitive benchmark](#4-the-engine-as-a-competitive-benchmark)
5. [Rating](#5-rating)
6. [Measurements](#6-measurements)
7. [The web surface](#7-the-web-surface)
8. [Proposed shape](#8-proposed-shape)
9. [Decisions to take before writing code](#9-decisions-to-take-before-writing-code)
10. [Claims in this repository that are now known false](#10-claims-in-this-repository-that-are-now-known-false)
11. [What was not measured](#11-what-was-not-measured)

---

## 1. What already exists

The reuse inventory is unusually good, because the project has repeatedly built the general
thing rather than the specific one.

| Need | Already built | Where |
|---|---|---|
| Tournament runner | paired seeds, seat swap every other game, per-game re-seeding, result independent of worker count | `training/alphazero/arena.py::compete` |
| Submission model | agents are **picklable specifications, not objects** — a submission is one new `kind` | `arena.py::build_agent` |
| Accept/reject rule | Wilson lower bound over 400 games, anti-regression against the fixed heuristic, `forced` + `forced_reason` audit trail | `training/alphazero/champion.py::promote` |
| Statistics | Wilson interval; "shown better, not merely ahead" | `training/evaluate.py:21`, `alphazero/evaluator.py:75` |
| Match record | seed + move indices, with `replay()` and `verify()` proving exact reconstruction | `interfaces/web/recorder.py` |
| Spectating | both seats agents, one decision per call, server-set pace, already leak-filtered | `interfaces/web/api.py::Game(watch=…)`, `advance()` |
| Leaderboard page | self-contained HTML, inline SVG, no JavaScript, no dependency, already draws Wilson bars | `training/alphazero/dashboard.py` |
| Surviving an encoder change | block signatures, a historical table, zero-column widening | `alphazero/layouts.py`, `network.graft` |
| Playable-opponent registry | `name -> factory(seed)`, served to the browser rather than hardcoded | `api.py::OPPONENTS`, `opponents()` |
| Path-traversal guard | probed with `../`, encoded traversal, backslashes, `CON`, absolute paths — all refused | `interfaces/web/server.py::safe_path` |
| A field to launch with | 23 AlphaZero snapshots, 2 champions, 5 built-in agents, with orderings already measured | `checkpoints/az_snapshots*/`, `catan/agents.py::DIFFICULTY` |

Two consequences worth stating plainly.

**The agent protocol is the whole leverage.** Because an agent is any
`callable(observation, info) -> int`, a sandboxed agent that talks to a child process over a
pipe implements the same signature and drops into `play_game`, `evaluate`, `arena.compete`,
`interfaces.cli` and the browser's opponent list *with no change to any of them*. Design the
sandbox as an agent, not as a new execution path.

**The leaderboard should not gain a dependency.** `dashboard.py` already demonstrates a
static HTML report with inline SVG and no build step, and `sqlite3` is stdlib. The whole
arena fits inside `stdlib + pytest + pillow (+ torch)`.

---

## 2. Blockers, verified by execution

### B1 — `info["view"]._state` is the live `GameState` 🔴

`catan/view.py:43`. `__slots__ = ("_state", "me")` installs a slot *descriptor*, and Python
only calls `__getattr__` when normal lookup **fails**. Slot lookup succeeds, so `FORWARDED`
— the allow-list — is never consulted for `_state`.

Reproduced at a real mid-game position (seed 7, 60 heuristic moves):

```
view._state is env.state -> True
opponent hand            -> [4, 0, 1, 1, 0]
dev deck top 3           -> [KNIGHT, ROAD_BUILDING, KNIGHT]
next dice                -> [(5, 5), (2, 6), (6, 5)]
```

Under Balanced Dice `dice_deck` is the literal next rolls. Writes work too — `state.hands[1]`
was set to `[50, 50, 50, 50, 50]` through this reference, and `state.winner`,
`state.robber_tile` and `state.vertex_owner` are equally reachable. `view._state.clone()` is
callable, which is unlimited perfect-information lookahead over exactly the action set
`training.agent.DETERMINISTIC_TYPES` exists to forbid (decision
[0020](decisions/0020-parallel-rollouts-and-lookahead.md)).

`__setattr__` raising *"a view is read-only"* is irrelevant: the attacker does not rebind the
view's attributes, it mutates the object the slot points at.

**This is not a bug in `PublicView`.** Two in-repo agents read `_state` deliberately
(`training/agent.py:178`, `training/alphazero/agent.py:99`) and are right to — decision
[0015](decisions/0015-public-view-instead-of-a-cheating-agent.md) designed the view as a
correctness aid for first-party code, and as that it works. The finding is that the docstring's
*"impossible rather than merely tested for"* is true against accident and false against intent,
and the arena is where intent arrives.

### B2 — an agent that never touches the view reaches the same object 🔴

`sys._getframe(1)` from inside `__call__` lands in `catan/agents.py::play_game`, whose
`f_locals` contains `env`. Verified end to end: opponent hand, deck order and next dice read,
then a write. `gc.get_referrers(observation)`, `info["view"].__class__.__init__.__globals__`
and plain `import catan.rules; catan.rules.apply = evil` are equally available.

**In-process isolation of CPython is not achievable.** No allow-list, no view redesign and no
audit of the `info` dict changes this. Any hardening of `PublicView` against a hostile
submitter is wasted effort — and worse, it reads as a fix.

Two proposed hardenings were tested and both fail:

- A `__getattribute__` refusing underscore names is **not implementable as written** — every
  `PublicView` property reaches through `self._state` (view.py:66, 70, 77, 81, 85, 89, 93, 98,
  103, 106, 109, 112, 115).
- `object.__setattr__(view, "me", opponent)` — the same call `__init__` uses at view.py:46-47,
  using no underscore attribute and no frame walk — succeeds, after which the *allow-listed*
  `my_hand`, `my_dev_cards` and `my_playable_dev_cards` return the opponent's cards.

The transferable lesson for any wire format: **per-seat private data must be absent from the
payload, not addressed by an index the recipient controls.**

### B3 — `info["scores"]` leaks hidden Victory Point cards 🔴

`catan/env.py:166` puts `rules.scores(state)` (which counts `DevCard.VICTORY_POINT`) next to
`public_scores`. One subtraction gives the opponent's exact hidden VP count — the fact that
decides whether they are one build from 15.

This is not a sandbox escape. It is a leak in the documented interface, readable by a
well-behaved agent using only public keys, and `tests/test_env.py:306-308` currently *pins*
the difference as intended. `scramble_hidden_state` does not cover it, because the existing
leak tests point at the observation and the view, not at `info`.

### B4 — loading a submitted checkpoint is remote code execution 🔴

Seven production sites pass `weights_only=False`:

```
training/agent.py:42, :93
training/alphazero/agent.py:70
training/alphazero/champion.py:136, :291
training/alphazero/network.py:209
training/train.py:88
```

It executes inside the `ProcessPoolExecutor` initializer (`arena.py:185`), so it runs on the
host once per worker, before a single game is played. `champion.py:291` is the worst of them:
`_install()` unpickles the **candidate** file and re-saves its `config` and `weights`, so a
submission validated safely at match time is unpickled unsafely at promotion time.

Two corrections to the obvious fix:

- `weights_only=True` **already loads both shipped champions unchanged** (verified on torch
  2.13.0+cpu; the permitted types cover `dict`, `OrderedDict`, `int`, `str`). No metadata
  sidecar and no migration script are needed — it is one word per site.
- `weights_only=True` is not the whole trust story. `network.py:210-215` does
  `config = dict(checkpoint["config"])` then `build(config)`, so hostile dimensions are an
  allocation bomb *before* any tensor is inspected — and `champion.load` wraps it in a bare
  `except Exception: return None` ("never raises"), so the failure is indistinguishable from
  "no champion present". Validate config **ranges**, not just its key set, in a loader that
  raises.

Fix it once rather than seven times: a single `training/loading.py::load_checkpoint(path, *,
trusted)` as the only module allowed to call `torch.load`, plus a test that greps the tree and
fails on a new call site. Same discipline the repo already applies to geometry.

### B5 — truncation is a free non-loss 🟠

`arena.py:198` computes `decided = wins + losses` and drops truncated games from the
denominator; `training/evaluate.py:47-50` does the same. Catan has no draws, so truncation is
the only escape from a loss, and it is reachable by an agent that simply refuses to build.
Measured truncation rates: heuristic **2 / 8,000**, greedy **153 / 2,000**, random
**264 / 2,000**.

The interval is also computed over the reduced denominator while `games` reports the full
count — a 400-game match with 100 truncations publishes a Wilson interval bought with 300.

Scoring a truncation as half a win *halves* the incentive without removing it: for an agent
that is behind, 0.5 > 0, so refusing to build is still strictly better than playing on.
**Resolve on `public_scores` at the truncation ply** — ahead wins, level is a loss for both.
The information is already in `info` at every step. Add it as a new field; leave `win_rate`
untouched, because `champion.py:219`'s regression check compares against recorded history.

A passive probe (an agent that only ends turns) went 0-38 with 2 truncations, so *passivity*
is not the exploit. Whether an **active** denial strategy exists — road-blocking expansion,
holding Longest Road, robbing the best number every turn — is unmeasured, and a public
competition is exactly where it would be found. Measure it before opening.

### B6 — nothing has a timeout, and one bad agent destroys the whole match 🟠

`pool.map` is called with no timeout (`arena.py:187`); `_play` loops with no deadline; a
`ProcessPoolExecutor` task cannot be cancelled once running. An exception inside `_play`
propagates out of the `with` block, so **a crash in game 397 of 400 discards 396 completed
games** — hours of MCTS-class compute, and no rating movement at all where a partial one was
perfectly valid (the games are independently seeded).

`arena.py:76-80` already records this failure happening once, from a mismatched model inside
the pool initializer.

### B7 — `reseed` duck-types, and silently no-ops 🟠

`arena.py:120-134` sets `agent.rng` and `agent._stream` if present and returns silently
otherwise. A submitted agent seeding from a closure, a module global, `numpy.random`, torch or
the clock is not re-seeded and not detected — so the module docstring's central claim, *"the
result does not depend on how many workers ran it"*, becomes false with no test failing.

There is a real trade here that must be decided rather than discovered: a long-lived child per
worker (which is what makes spawn cost affordable — see [§6](#6-measurements)) sees every game
in its slice and *can* retain state across them. You cannot have cheap spawn **and** the
worker-count invariance `tests/test_alphazero.py` pins, against an agent that chooses to ignore
a reseed frame. Keep the per-worker child, drop the invariance claim for submission rows, and
stamp those rows with the worker count so they are reproducible by construction.

### B8 — the web server is a local single-player app, correctly 🟠

Everything here is right for what it is and disqualifying for the internet.

| | |
|---|---|
| `api.py:231`, `:243` | game ids are `itertools.count(1)` — sequential and enumerable |
| `api.py:641` | `view()` reveals seat 1's hand unconditionally, so `GET /api/game/2` reads a stranger's cards and `POST …/action` plays their moves |
| `server.py:48`, `:173` | `ThreadingHTTPServer` over shared mutable `Game` objects, no lock anywhere; the only thing preventing interleaved `/advance` calls is `state.busy` **in the browser**, which by this project's rules decides nothing |
| `api.py:715-736` | `Games` never evicts — ~136 KB per completed game, forever |
| `server.py:130-137` | `Content-Length` is read with no cap; no socket timeout, unbounded threads per connection |
| `recorder.py:43` | filenames are `%H%M%S`, so two games in one second collide and one is silently lost; `save()` rewrites the whole JSON after every move (~8 MB of writes per game) |
| `api.py:111-113` | every `POST /api/game` re-reads the checkpoint and rebuilds the network — measured **1.94 s of CPU on the request thread** |

The `api.py` / `server.py` split is genuinely good and the docstring's claim that swapping the
server "would touch nothing else" holds **for routing** and fails **for the process model**:
`Games` is a process-local dict, so `--workers 2` splits it and half the requests 404.

### B9 — no version identity on any recorded number 🔴

`grep -rn '__version__|ENGINE_VERSION'` over `catan/`, `training/`, `interfaces/` finds
nothing. `models/champion_az.json` records `observation` and `actions` and nothing about the
rules. `recorder.py:98` writes `"version": 1`, which is the *record schema* version.

This has already bitten, and CLAUDE.md records it: commit `e4b0441` changed pre-roll dev-card
legality after a champion was promoted, turning a recorded 71.6% into a measured 49.3%.

It has bitten a second time, unnoticed. Running `recorder.verify` over all 19 files in
`games/`: **11 replay exactly, 6 are unfinished, and 2 raise** — `unknown action ROLL(0)` and
`must roll the dice before PLAY_YEAR_OF_PLENTY(ore, ore)`. Seed-plus-moves is the arena's audit
primitive and its cheapest storage format, and it is silently version-scoped.

Related: `recorder.replay` hardcodes `num_players=2` and never passes `max_turns`, so it
rebuilds under `DEFAULT_MAX_TURNS = 5000` regardless of what was played. Since
`_advance_to_decision` stops auto-rolling at the cap, replaying a long game under a different
cap changes which states are offered as decisions — and `verify()` then reports a mismatch that
is the replayer's fault, not the record's.

---

## 3. The seed is the whole hidden state

This is the finding that changes the plan, and it defeats every submission format anyone
proposed.

**The board is a perfect fingerprint of the seed.** 20,000 seeds enumerated → 20,000 distinct
`board.layout` values, zero collisions. One `GameState` construction costs 0.379 ms, so a core
scans 2,637 seeds/second.

**The seeds are constants in the source.** `compete(seed=10_000)` (arena.py:160),
`rank(seed=30_000)` (:211), `promote(seed=41_000)` (champion.py:175), with game *g* at
`seed + g`. A rung's search space is 400 wide — **0.2 CPU-seconds**.

**Both were exploited.** Seed 30137 recovered in **63 ms** from `board.layout`; seed 41213
recovered in **360 ms** from the observation's `tiles` block alone (`encoder.LAYOUT["tiles"]`,
slice 0-361). In both cases `dev_deck` and `dice_deck` compared identical to the real game and
the next rolls were read off exactly.

The board must be shown to anything that plays Catan. So this is not closed by a process
boundary, by weights-only submission, or by an observation-only contract — all three hand over
the board. `"board"` is the first entry in `catan/view.py::FORWARDED`; the allow-list itself
blesses the vector.

Three defences that look right and are not:

| Proposed | Why it fails |
|---|---|
| "assert no frame emitted during play contains the seed" | the attack never needs the seed transmitted; it recomputes it. The test passes green while the break works — worse than no test, because it will be cited as proof |
| reuse `scramble_hidden_state` on the wire payload | **blind by construction**: scrambling destroys the seed→deck correlation the exploit uses, so the payload tests byte-identical while remaining fully invertible. Shuffling `dev_deck` and `dice_deck` leaves `board.layout` unchanged |
| an observation-only contract is "safer" | it is the same break. Smaller, yes; safer, no — the tiles block *is* the board |

**What actually works.**

1. **Entropy.** Draw ranked base seeds from `random.SystemRandom().getrandbits(128)` — verified
   `GameState` accepts an arbitrary-width int — recorded in the match record and nowhere else.
   Never reuse a seed set across submissions.
2. **Stream splitting.** `catan/state.py` seeds one `random.Random` for board generation, both
   decks *and* every subsequent draw. Split it into a board stream and a hidden-deck stream, so
   a disclosed board seed does not determine the decks. This also makes replay-after-publication
   safe.
3. **Canary games — the only detection story.** Reshuffle the unseen tail of `dev_deck` and
   `dice_deck` at a random ply in a fraction of ranked games. Publicly undetectable by
   construction, because only the multiset is public — which is exactly what
   `training/alphazero/determinize.py` already formalises in `unseen_resources` /
   `unseen_dev_cards`. An honest agent is unaffected; a seed-replaying agent's inferences go
   stale from that ply on, and the per-agent win-rate delta between canary and non-canary games
   is testable with the Wilson interval that already exists.

⚠️ **Publishing replays and paired seeds are individually reasonable and jointly a total
break.** A published replay of the promotion rung at seeds 41,000-41,399 hands every future
submission the exact hidden state of the games it will itself be graded on. Pick one: rotate to
a fresh high-entropy seed set per submission (the pairing is worth only ×1.16 — see
[§5](#5-rating) — so it is cheap to lose), or publish replays only after a seed set is
permanently retired.

A recorder file already carries the seed at top level, and every decision entry stores the
**mover's full legal-action list** including the opponent's, which bounds their hand at each
decision. A replay is not a safe public artefact as written.

---

## 4. The engine as a competitive benchmark

Determinism is genuinely excellent and is the hard half. Verified: `reset(seed)` alone pins the
board, both decks and turn order; it is independent of who plays; the board is byte-identical
across two games with the seats swapped; and `randomize_order` defaults `False` and is called
nowhere outside one test. A perfectly paired two-seatings-on-one-board match is available for
free.

Four things are not ready.

**`max_turns` is not part of the ruleset and takes four values.** `env.py:41` 5000,
`evaluate.py:36` 1000, `arena.py:161` / `evaluator.py:37` / `champion.py:107` 800,
`mcts.py:182` / `alphazero/agent.py:57` / `config.py:36` 400. The cap decides which games are
draws, so it is part of the rules, and nothing records it. Move it into `RuleSet` and delete
the four scattered defaults.

**`MCTSAgent` silently stops searching at turn 400.** Its horizon (400) is shorter than the
arena's game length (800). Past the cap, `_classify` returns TERMINAL, `searchable` is False,
and `agent.py:106` falls back to `_policy_only` with no announcement. CLAUDE.md records that
the raw policy and the same weights with search move in *opposite* directions, so this
systematically corrupts the tail of long games — which are disproportionately the close ones.
A figure labelled "champion at 32 simulations" is not what was played in those games.

**Seat swapping is not paired to the seed.** `shift = game % 2` and `seed = base + game` vary
together, so the two seatings are played on different boards. Balanced in expectation, never
mirrored. Note this is a *statistical* miss, not a large one — see [§5](#5-rating).

**3-4 player is a different season, not a feature.** Measured with four `HeuristicAgent`s at
`max_turns=1000`: `BASE_GAME` 4p finished **12/12** (median 110 turns), `BASE_GAME` 3p **12/12**,
`RANKED_1V1` 4p **5/12**. And `mcts.Search` refuses `num_players != 2` by design — the seat-1
value frame is the design, not a flag — so no learned model could enter without a per-seat
value vector. Combined with decision
[0011](decisions/0011-no-player-to-player-trading.md) (no player trading), 3-4 player here is
a materially different and arguably degenerate game.

---

## 5. Rating

### First-player advantage is zero in this engine

Measured, because nothing in the repo had measured it. Mirror match, `RANKED_1V1`,
`max_turns=800`, **fixed** seats:

| agents | seat 1 wins | games |
|---|---|---|
| `HeuristicAgent(noise=0)` | **49.8% [48.7, 50.9]** | 8,000 (2 truncated) |
| `GreedyAgent` | 51.9% [49.6, 54.1] | 2,000 |
| `RandomAgent` | 51.3% [48.9, 53.6] | 2,000 |

The cause is structural: `state.py:230` `setup_sequence` is a snake `[1,2,2,1]`, so seat 2 gets
picks 2 and 3. The claim that Catan's first-player advantage is "real and large" — at
`catan/agents.py:414`, `docs/ai-surface.md:146`, decision 0014 and a test docstring — is
4-player folklore and is **false for this engine**. Keep the seat swap; it costs nothing and
±1.1 points is not zero. Stop justifying it with a number that is wrong.

### Pairing is worth much less than expected

Over 900 boards played both seatings, mirrored-pair outcome correlation was +0.001 (heuristic),
+0.013, +0.044 (greedy) — indistinguishable from independent. Shared seed pools across
*candidates* give paired s.e. 1.32 points against unpaired 1.43 over 1,797 games: a **×1.16**
game saving.

The reason is structural and cannot be fixed cheaply: `reset(seed)` seeds one RNG for the board
*and* every subsequent roll, so two agents share the board and diverge on dice at the first
differing action. The board is a small part of the variance.

### Use a batch Bradley-Terry fit, not Elo, Glicko-2 or TrueSkill

All three are online filters whose K-factor and rating-deviation inflation exist to **forget**
old evidence about drifting human strength. Submitted agents are frozen files: strength is
exactly stationary, old evidence never goes stale, and an order-dependent rating would break
the reproducibility property this repo enforces everywhere else.

Recommended shape:

- MAP fit over the whole game log, **refit from scratch on every update**, so the table is
  order-independent and is a pure function of an append-only JSONL.
- `HeuristicAgent(noise=0)` **pinned at 0**, so every rating is a statement about one reference.
- The seat term fitted as a free parameter and published as a monitored row — it is 49.8% today,
  and if a rules change makes it non-zero the table says so instead of absorbing it.
- Intervals from the observed Fisher information, and a `difference(a, b)` with its own
  interval, because that is the quantity a reader actually cares about.

**Precision to be honest about.** dElo/dp = 400/(ln10 · p(1−p)) = **695 Elo per unit win rate**
at p = 0.5, so s.e. ≈ 347/√n: 400 games is **±34 Elo**, 1,000 games is ±21.5. The gap between
the current champion and its predecessor (51.8% head to head) is about **12 Elo** — inside the
interval of a 400-game match. A table sorted on point estimates reorders itself on noise every
refit. The interval must be the primary visual.

### Two structural problems

**The gate and the ladder will disagree by construction.** `MAX_BASELINE_REGRESSION`
(champion.py:61, 272-279) refuses a candidate that beats the champion but falls more than 5
points against the fixed heuristic — which is precisely the entry a head-to-head ladder ranks
first. So "top of the ladder" and "the model that ships" are different objects, and
`DEFAULT_OPPONENT` picks from `models/`, not from any ranking. Say in the record that they
answer different questions and never merge them into one table.

*(Separately: that regression check compares two 400-game point estimates against a fixed
5-point threshold. The s.e. of the difference is ≈3.5 points, so the rule sits inside its own
noise — it both refuses good candidates and passes real regressions. It is the one place in the
gate that violates the project's own intervals-not-point-estimates rule. The current champion's
`forced_reason` already computed a two-proportion z by hand; codify that.)*

**The anchor drifts.** The gate plays candidates against `{"kind": "mcts", "path": CHAMPION}` —
a *path* — so "against the champion" means whatever file is there today.
`models/champion_az.json` records three promotions on **one day** (15:11, 18:23, 21:43), the
last one `forced: true`. Ranking submission A against the 15:11 champion and submission B
against the 21:43 one produces two rows measured against different opponents. Freeze content,
not paths: `models/anchors/<sha256>.pt`, with the sha in every row. A row is
`(submission, anchor, rules, engine commit)` or it is not comparable to anything.

### Anti-gaming

Seed-fishing is the live one and is addressed in [§3](#3-the-seed-is-the-whole-hidden-state).
Beyond it: derive match seeds from `HMAC(secret, season:purpose:i)` with a gitignored secret and
a **published** practice pool so submitters can self-test on statistically identical boards they
cannot game; identify a submission by `weights_hash` plus a behavioural hash over a fixed probe
set, so a renamed copy of the champion merges into one row rather than appearing twice; and keep
a mandatory quota of games against the anchor set so the comparison graph never fragments into
cliques, which is the shape collusion takes.

---

## 6. Measurements

Machine: i7-12700H, 14 physical / 20 logical cores, 16.8 GB, Windows 11, Python 3.14.3,
torch 2.13.0+cpu. Games are `RANKED_1V1`, 2 players, `max_turns=800`, seats swapped, one
warm-up game discarded, torch pinned to one thread.

### Throughput

| | s/game | decisions/game | 400-game match |
|---|---|---|---|
| heuristic(0) vs heuristic(0) | **0.161** (sd 0.045) | 258 | 64 CPU-s — 12 s wall on 16 workers |
| greedy vs greedy | 0.31-0.32 | 763 | 3/40 truncated |
| random vs random | 0.34-0.35 | 828 | 4/40 truncated |
| AZ champion (32 sims) vs heuristic | **8.9-9.2** | 229 | ~3,600 CPU-s — 10-13 min wall |
| AZ vs AZ (extrapolated, not run) | ~17.4 | — | ~7,000 CPU-s |

One searching game costs **56×** a heuristic one. That ratio should drive all scheduling. A full
`promote()` (three rungs) is **~17 minutes**, not "a couple of minutes".

The cost ordering is **inverted from the strength ordering**: the heuristic is the cheapest
agent per *game* and the most expensive per *decision* (0.64 ms vs random's 0.42 ms), because
random play needs 3.2× as many decisions to reach 15 points.

### Micro-costs

`encoder.encode` 219 µs · `env.step` 497 µs amortised (83% of a heuristic game's wall clock) ·
`legal_actions` 292 µs · `legal_mask` 295 µs · `GameState.clone()` 54 µs · `PublicView(...)`
0.77 µs (it copies nothing) · `CatanEnv() + reset()` 1.42 ms.

`cProfile` over 20 random games: `legal_actions` is **59%** of engine time, `apply` 25%.

### Startup and memory

torch import in a fresh subprocess **3.39-3.49 s** — so "half a second per container" is
fantasy for anything carrying torch, and **one child per worker, not per game** is the only
affordable shape. `from interfaces.web import api` costs 3.35 s because it pulls torch in
through `training.champion`. Arena pool spawn: 3.4 s (1 worker) → 6.2 s (16).

`arena.py:107` imports torch **unconditionally** in every child, so a heuristic-only match pays
16 × 3.4 s of import against 64 s of play — **46% overhead** — plus 3.0 GB of RSS for a library
it never calls. Making it conditional on the specs is a two-line change.

RSS: bare interpreter 15.0 MB → +engine 18.3 MB → +torch 209.9 MB → +champion 218.9 MB. **torch
is 191 MB per process; the whole engine is 3.3 MB.** A 16-worker ranked match with sandboxed
children is ~7 GB on a 16.8 GB box, so **the machine runs exactly one ranked match at a time** —
state that as a documented constant and an exclusive lockfile, not as prose.

### Storage

A recorded game is **316 B/decision**, ~70 KB per 223-decision game, fitting `499 + 318 × moves`.
But `recorder.py:71` stores the full legal-index list for every decision, while `replay()`
rebuilds the game from **seed + moves alone** — so every legal set is recomputable, and storing
it is both the disk cost and a second copy of a fact `rules.legal_actions` is supposed to be the
sole authority for. A `decisions=False` mode is ~30 B/decision, a **12×** saving, for about an
hour of work. `verify()` reads only seed/moves/result, so it keeps working unchanged.

Checkpoints: `champion_az.pt` 811 KB, `champion.pt` 781 KB, `checkpoints/best.pt` 2.4 MB
(it carries optimizer state — archive the `models/`-shaped file, not the checkpoint).

### ⚠️ Every performance constant in this repository is 2.4-3.7× optimistic

| stated | where | measured |
|---|---|---|
| heuristic game "costs 56 ms" | `arena.py:5` | 161 ms |
| MCTS game "5.2 seconds" | `arena.py:5` | 8.9 s |
| "about 32 ms a decision" at 32 sims | `api.py:60` | 91.5 ms (median 93.8, p90 102.4) |
| "64 simulations costs about 25 ms" | `alphazero/agent.py:24` | — |
| "~36,000 actions/sec", 56 ms/game | `benchmark.py:13` | 14,123/s, 136 ms |

A capacity plan built on those under-provisions by ~3×. Correct them in one commit before
anyone plans against them — the repo's own *measure before claiming* rule already demands it,
and `arena.py`'s multiprocess design is only more justified at the true numbers.

### ⚠️ The measurements themselves are unstable to 3.5-5.4×

Identical work — same script, same 40 games, same seeds, decision counts matching to the digit —
measured 0.877 s/game in one round and 0.161 s/game in two later ones, with CPU/wall = 0.99 in
the fast round (so it was not descheduling). The cause was not isolated; a second 16-worker job
was resident throughout.

The operational consequence is load-bearing: **a wall-clock deadline on a shared box
manufactures forfeits**, and forfeits are not missing at random — an agent that times out only
in long, complicated positions forfeits exactly the games it was winning or losing, and the
Wilson interval does not know. Budget in **CPU time** (`RLIMIT_CPU` is cumulative and immune to
machine load); keep a wall-clock kill only as a liveness backstop for a sleeping agent, at ~10×
the CPU budget. And make it a rule that a rung whose forfeit rate exceeds a threshold is
discarded and re-run, not published.

Nothing in the repo coordinates the arena with the training loop, and they are the same 14
cores. A machine lock that `train.py`, `compete()` and any ladder runner all take is the missing
piece.

### Hosting

One Hetzner box, systemd, SQLite in WAL on local NVMe, Caddy for TLS, hourly `sqlite3.backup()`
to R2/B2: **€10-30/month** (CX32 ~€7-8, CCX23 dedicated ~€25). "Popular" — 20k ranked games/day
— is ~11 cores continuous, ~€105/month on Hetzner against ~$340-850 elsewhere. Serverless is
8-12× the price for sustained CPU, and a 520 MB torch install makes cold starts painful.
List prices are quoted from memory; the load-bearing part is the **ratio** (dedicated vCPU 1× ·
DO/Fly/Railway 3-4× · Lambda ~10×).

Note `arena.compete`'s default `workers = cpu_count() - 4` yields **1 worker on a 4-vCPU VPS** —
an 8× surprise the first time it runs on a small server. Pass `workers` explicitly.

---

## 7. The web surface

Beyond [B8](#2-blockers-verified-by-execution), one measurement changes
the deployment design.

**Torch threads are never pinned in the web process.** Same machine, same agent, same 32
simulations:

| | median | p90 | max |
|---|---|---|---|
| default (14 threads) | **1,758 ms** | 17,983 ms | 22,444 ms |
| `torch.set_num_threads(1)` | **179 ms** | — | 237 ms |

Batch-1 forward passes spend everything in the OpenMP barrier. CLAUDE.md already records that
the pool is sized at import and that thread env vars must be set in the parent — the training
code got that treatment and the web process never did. It is a one-line fix worth 10× median
and ~75× tail, and without it the site reads as *"the AI is slow"* rather than *"the deployment
is misconfigured"*.

**Playing against a submitted agent is nearly free.** `OPPONENTS` is `name -> factory(seed)`,
`Game.__init__` calls it in exactly one place, and the browser already fills its dropdown from
the server — so a rated submission becomes selectable with **one dict entry and no client
change**. Two constraints: the registry is built at *import* (`_register_learned()` at
api.py:116), so new entries mean "on restart"; and `RemoteAgent.__call__` must return within
budget **always**, falling back to a legal move, so a hostile agent can never block the thread
serving a human's click. At the pinned 179 ms median, a 500 ms interactive budget has headroom.

**Replays are closer than they look.** `api.view()` touches only nine attributes of `Game`
(`state`, `info`, `id`, `opponent_name`, `rules_name`, `watching`, `watcher_name`, `log`,
`awaiting_opponent`), so a `Replay` class that duck-types them reuses `view()` verbatim and
therefore the entire existing renderer. The only new UI is a scrubber; cache an `env.clone()`
every 20 plies so seeking is not O(n) per frame.

But extract `drawBoard` out of `app.js` (lines 270-455, bound to a module-level `state` and
fixed DOM ids) **before** writing a second page. Copying 190 lines of SVG into a second file is
the second-implementation problem this project already removed from JavaScript once.

Deployment shape: keep `server.py` exactly as the zero-dependency local game — that is a real
asset — and add `interfaces/web/asgi.py` (Starlette; you already produce dicts, so pydantic buys
nothing) as a separate target, calling `safe_path` rather than reimplementing it, with
`Cache-Control`/`ETag` on the 1.64 MB of board art that currently ships with no caching headers
at all.

---

## 8. Proposed shape

### Tier 0 — a local ladder (2-3 focused days)

`python -m ladder run` ranks agents on the owner's machine. Seeded from the 23 AlphaZero
snapshots and 5 built-in agents already on disk, anchored on content hashes, appending per-game
rows to `ladder/games.jsonl`, publishing a static HTML table with the engine fingerprint in
every row. No auth, no sandbox, no uptime, no LICENSE decision needed to start.

It delivers the ranking, violates nothing in CLAUDE.md, is the same shape as
`training/alphazero/report.py` and `checkpoints/dashboard.html` which already exist — and it can
sit dormant for a year and still be correct when it wakes.

Acceptance: worker-count independence for a ladder pairing · a row carries a fingerprint and the
report **refuses** to pool two · a stalling entrant does not out-score `RandomAgent` · an entrant
reaching for hidden state is caught · a recorded game replays exactly.

### Tier 1 — weights-only submissions by pull request (+2-3 days)

A submission is a `.pt` plus a JSON sidecar in `submissions/`, arriving as a pull request,
validated by GitHub Actions: sidecar parses, `obs_size == encoder.SIZE`,
`num_actions == action_space.NUM_ACTIONS`, `torch.load(weights_only=True)` succeeds, tensor
shapes match a freshly built net, config values within bounds.

**No submitter code ever executes**, so the entire sandbox / nsjail / abuse-surface branch drops
off the critical path. GitHub supplies identity, rate-limiting and abuse handling for zero
dependencies and zero operating cost — which answers "who authenticates", the open question
every part of this audit flagged as most likely to be underestimated.

State the restriction honestly: it is a **weights competition, not an agent competition**,
because everyone submits into `training.net.build`'s architecture.

### Tier 2 — subprocess submissions (the cliff)

Line-delimited JSON over the child's stdin/stdout, one long-lived child per worker, stderr
captured verbatim into the match record (it is the only debugging a submitter gets). Frames echo
`game` and `ply` so a late reply is rejected rather than silently answering the wrong position.
Illegal index, timeout, malformed frame or crash → a recorded forfeit, never a substituted move.
Hard byte cap on reads, because `json.loads` on an unbounded line is a one-line OOM.

Costs to accept before starting:

- A Linux host you do not develop on. Windows has no `setrlimit`, no seccomp, no namespaces —
  so this is *acquire and operate a second machine*, an ongoing commitment, not a task.
- **UID separation ships with the subprocess tier, not after it.** A same-UID child can read the
  parent's `GameState` out of `/proc/<ppid>/mem` (or `ReadProcessMemory` on Windows), so during
  the phase where the boundary is called "structural" it is not a boundary at all. Encode it:
  a child running as the parent's UID is enforcement level `none`, not `weak`, and ranked play
  refuses below `full`.
- Budget sized from the measured champion, not a round number: **250 ms/decision** (2.7× its
  91.5 ms) and **30 s/game** (3.4× its 8.9 s). The 1 s/decision figure that looks generous is a
  factor-of-eleven capacity bomb — 400 games × 129 decisions × 1 s / 16 workers is **54 minutes
  per rung** against a champion that needs 3.7.

### What to skip

**HTTP-hosted submissions.** Reject them on *reproducibility*, not latency: a submitter-hosted
endpoint mutates under you, so a published number becomes unauditable and "comparable within one
version of the engine" silently gains a third clause nobody can check. (The latency argument
that looks compelling is wrong — a rung is ~51,600 round trips, but `arena` divides them 16
ways, so 50 ms RTT is ~2.7 minutes, not half an hour. An argument that is wrong for its stated
reason gets re-litigated by the first person who does the division.)

---

## 9. Decisions to take before writing code

Each is expensive to reverse once a row is published.

1. **Seed entropy and rotation.** Everything else is downstream of
   [§3](#3-the-seed-is-the-whole-hidden-state). Ranked seeds must be 128-bit and released only
   in the post-match record.
2. **Weights or code?** Weights-only is most of the value for a fraction of the work and makes
   the safety claim trivially true. If it is the long-term product, the sandbox work is
   unnecessary and the effort belongs in the validator instead.
3. **Seasons.** `encoder.SIZE` went 1808 → 1868 → 1884 → 2503 in six days. A digest that changes
   weekly gives one leaderboard row per week. Pin a season to a tagged commit, let `main` keep
   moving, re-run at a boundary — and **budget it**: ~24 h of the whole box for a 20-entry
   re-run. That tax is levied on the thing the project exists to do (improve the engine) by the
   thing built to attract other people, and it should be a decision rather than a surprise.
4. **Truncation scoring** ([B5](#2-blockers-verified-by-execution)). Fix it before the first
   row is written; changing it later renumbers every published result.
5. **A frozen anchor.** `HeuristicAgent(noise=0)` plus a content-hashed champion copy. The
   corollary bites: the heuristic can never be retuned mid-season, which conflicts directly with
   CLAUDE.md's note that improving it raises the ceiling for training. Pick one per season and
   say so.
6. **Rating identity is `(weights, simulations, budget)`.** `champion.py:53-57` already makes
   this argument. Let submitters pick their own simulation count and the ladder ranks compute
   spend, not skill.
7. **One serializer, not three.** The wire view, a seat-parameterised `api.view()`, and a
   protocol shim were each proposed separately — three serialisations of the hidden-information
   filter, in the one area this project has been most careful about. One function builds the
   public payload; the browser and the wire are two callers. Decide it before either module is
   written.
8. **What happens to the local web app.** It is the only part of this repo with users. Decide
   explicitly that it stays local and unchanged and the ladder is a separate deployment sharing
   only `catan/` — rather than taking a regression risk on `view()` for a platform that may
   never be built.

---

## 10. Claims in this repository that are now known false

Recorded here so they are corrected deliberately rather than inherited.

- **"Catan's first-player advantage is real and large"** — `catan/agents.py:414`,
  `docs/ai-surface.md:146`, decision 0014, `tests/test_agents.py:123`. Measured 49.8%
  [48.7, 50.9] over 8,000 games. See [§5](#5-rating).
- **The four performance constants** — `arena.py:5`, `alphazero/agent.py:24`, `api.py:60`,
  `benchmark.py:13`. All 2.4-3.7× optimistic. See [§6](#6-measurements).
- **`rank()`'s "the comparison between them is paired rather than two independent samples"**
  (`arena.py:211`). The *design* is paired; the *analysis* is not — it reports each candidate's
  unpaired Wilson interval and sorts on the point estimate. There is no valid interval for a
  **difference** anywhere in the codebase, which is exactly what a leaderboard publishes.
- **`view.py:9`'s "impossible rather than merely tested for"** — true of the observation, true
  against accident, false against intent. That sentence is the one that stops anyone looking.
  Amend it: an allow-list constrains which *names* resolve, not which *seat* they resolve for,
  and both escapes in [§2](#2-blockers-verified-by-execution) live in that gap.
- **README says 854 tests, ROADMAP says 598, actual is 865.** The two documents a newcomer reads
  first disagree on the headline number.
- **Two of the 13 finished games in `games/` no longer replay.** See
  [B9](#2-blockers-verified-by-execution).

Also absent and load-bearing for anything public: **no `.github/`** (no CI, on a repo the plan
turns into one that accepts pull requests), **no `LICENSE`** (so publishing `catan/` as an SDK
ships all-rights-reserved code that submitters legally cannot build against — the SDK is a
non-starter until that is decided), **no `pyproject.toml`**, no `CONTRIBUTING`, no `SECURITY.md`.
The board art under `interfaces/static/images/` is vendored from FullStackCatan; copying it into
a local app is what the ROADMAP signed off on, and serving it from a public site is a different
act with a different answer.

One implementation trap found by execution and worth recording: **do not name the package
`platform/`.** It shadows the stdlib module, and torch imports it during initialisation —
verified, `import torch` fails with `module 'platform' has no attribute 'machine'`. It would
surface as a torch failure and read as an environment problem. Use `ladder/`.

---

## 11. What was not measured

- **AlphaZero versus AlphaZero.** The ~17.4 s/game and ~7,000 CPU-second figures are
  extrapolated from the measured 91.5 ms searched decision, 16.8% forced fraction and 229
  decisions/game. The forced fraction in particular will differ when both seats search.
- **The seat effect for a searching agent.** The 8,000-game result used
  `HeuristicAgent(noise=0)`. It almost certainly transfers — greedy, random and the heuristic
  span an enormous strength range and all sit within 2 points of 50% — but a searching agent
  plans over the setup snake differently, and it is the anchor of the whole scheme. 400 games of
  the champion against itself at fixed seats is ~40 minutes on 16 workers. If it comes back
  outside [45, 55] the seat term stops being a monitoring row and becomes load-bearing.
- **Whether an active denial strategy exists.** See [B5](#2-blockers-verified-by-execution).
  Passivity is not the exploit; nobody built the adversary that road-blocks and robs.
- **The sandbox's own cost.** Per-decision round-trip overhead against the measured 91.5 ms
  decision, and whether a 400-game rung still fits its budget. Per the project's own rule,
  measure it before building the leaderboard on it — and warm up first.
- **Parallel efficiency on a quiet box.** 16 workers delivered 10.3× on heuristic games and 5.9×
  on MCTS games here, but another job held part of the machine throughout. Both are floors.
- **The torch virtual-address high-water mark**, which is what `RLIMIT_AS` must be sized from.
  Sizing it from the 210 MB RSS would kill every torch submission at import.

---

## The cheapest way to de-risk all of it

Run a private invitational **by hand**: `arena.compete` against three friends' agents, printed
table, one afternoon. It costs nothing and answers the question no amount of engineering
addresses — whether anyone will write an agent at all. *Nobody submits* is the highest-probability
failure of this entire plan.

Second cheapest: seed the table from the 23 snapshots already on disk and publish
`DIFFICULTY`'s easy/medium/hard as named milestones — "beats random", "beats greedy", "beats
hard". The rungs exist and are already measured monotone. Without them a newcomer's first agent
scores near zero against every entry (heuristic-hard beats random 98.3%, the AlphaZero champion
beats hard 74.7%), gets an interval the width of the table, and learns nothing — which is how a
ladder with fifty submitters still feels dead.

# Working notes

Things that are true about this repository and cost time to discover. Read this before
changing the engine, the observation, or the agents.

`docs/decisions/` holds the full reasoning; this is the short version plus the traps.

---

## Ground rules this codebase actually holds to

**One source of truth.** If a fact exists in two places it will diverge. Geometry is
generated from `ROW_LENGTHS`, not written down. The browser draws what the server sends and
decides nothing. The web client is told which image file goes with which resource, because
the art set calls wheat `weat` and ore `stone` and a second copy of that mapping in
JavaScript would be a bug waiting to happen.

**Hidden information is impossible, not merely absent.** `PublicView` is an allow-list — a
new field on `GameState` is invisible to agents until someone adds it deliberately. Anything
that could leak has a test that mutates the hidden state at constant public counts and demands
the observable output not move. `tests/helpers.py::scramble_hidden_state` is the shared
scrambler; use it rather than writing a weaker one.

**Measure before claiming.** Several "obvious" improvements in this project measured as no
better, and one measured as the opposite. Numbers or it did not happen.

**Confidence intervals, not point estimates.** 200 games gives about ±7 points, which cannot
tell 45% from 55%. `training/evaluate.py` has a Wilson interval; use it. Several results in
this session flipped meaning between 200 and 800 games.

---

## Engine invariants the training code depends on

All three are undocumented properties of `CatanEnv`, all three are load-bearing, and all three
fail *silently*. Pinned in `tests/test_training.py`.

1. **The winner is always the player who just acted.** So `step()`'s reward is always `+1`
   and `LOSS_REWARD` is unreachable on the normal path. A learner that consumed the returned
   reward trains on winners only, its critic converges to `V = 1`, every advantage collapses
   to zero, and nothing crashes. Read `info["winner"]` and write both seats.
2. **At `GAME_OVER`, `current_player` becomes the winner**, so the terminal observation is
   the *winner's* view. Bootstrapping the loser's trajectory from it is an exact sign flip on
   half the data.
3. **The terminal mask is all-zero.** A masked softmax over nothing.

Also: the game is **not alternating** — during a discard the decision belongs to whoever is
over the hand limit. `info["player"]` is the authority.

---

## The observation

2,503 floats. `LAYOUT` maps a name to its slice; `VERTEX_OFFSETS` names the fields inside a
vertex row, because counting to them by hand is how a test broke.

```
tiles         19 x 19   resource, number, odds, robber
vertices      54 x 27   owner, city, harbour, pip potential, buildability,
                        per-resource production, harbour nearness
roads         72 x  6   owner, buildability
players        4 x 34   hands and holdings (masked for opponents), production rate
affordability  4 x  4   my hand against each purchase, priced through my trade rates
history        4 x 12   production, spending, purchases, idleness  (the public record)
rolls             12   how often each total has come up
global            40   phase, last roll, bank, ruleset, bookkeeping, board scarcity
```

**About 40% of it never changes during a game** and is cached on the `Board`
(`encoder._static_template`). If you add a board-static feature, put it there — the per-encode
path was 57% of training time before this existed. A feature that depends on a *hand* cannot
go there: the `Board` is shared across every clone and every observer, so a cached per-observer
value is both stale and a leak.

**Encoding a constant is worth nothing.** The cost table is identical in every state, so it
folds into a bias in one gradient step. What the `affordability` block encodes is the
*state-dependent* part: cards short, and what closing the gap would cost at the bank given my
own harbours. See `docs/decisions/0022-affordability-features.md`, which is also where the
argument for four columns rather than twenty lives.

**What is NOT in it, and people assume is:**
- ~~**Which numbers a vertex touches.**~~ Fixed in record 0024: a vertex now carries its
  expected cards *per resource*, and how near the closest harbour of each kind is. The old
  resource-blind `pip potential` is still there and still first — the agent that maximised it
  placed at 0.005 pips off the best available spot every time, which is what a resource-blind
  signal gets you.
- **Adjacency**, for the flat MLP. It must infer that vertex 23 neighbours 24 from
  correlations, though `topology.py` knows.
- **Any estimate of an opponent's hand.** Deliberate — a robber steal moves a card only two
  players ever see. The `history` block gives cumulative production and spending per player,
  which bounds it; deriving the bound is left to the network.

Changing `encoder.SIZE` no longer invalidates a checkpoint, as long as the change **appends**:
`training/alphazero/layouts.py` records the block shapes, new checkpoints carry their own, and
`network.graft` widens every observation-width layer with zero columns. Verified rather than
assumed — the champion measured 74.7% before the 1884 → 2503 change and 75.6% after. Add an
entry to `layouts.HISTORICAL` keyed by the new size, and **never edit an existing one**: it is
a statement about a file already on disk. The interfaces check `obs_size` *and*
`num_actions` before offering a model, because a stale one loads fine and then fails on the
first move.

---

## The action space

325 flat indices. **Append, never insert** — every existing index keeps its meaning, and the
structured network's final concatenation slices `other[:, 21:]` open-ended so an appended
action lands correctly with no code change.

The positional blocks are **not contiguous**: `TRADE_WITH_BANK` sits between the city and
robber blocks. `training/structured_net.py::_validate` checks this at import; it has caught
two mistakes already, including one of its own.

---

## Traps

**`VERTEX_TILES` and `TILE_VERTICES` are both keyed by plain integers.** Swapping them
type-checks, runs, and silently produces nonsense. It happened in `robber_damage`. Name the
variable after the id it holds.

**Applying an action to a `clone()` can reveal hidden information.** The dev deck, dice deck
and opponents' hands are copied verbatim, so a lookahead over `BUY_DEV_CARD`, `MOVE_ROBBER`,
`PLAY_KNIGHT`, `END_TURN` or `PLAY_MONOPOLY` sees the *real* outcome.
`training.agent.DETERMINISTIC_TYPES` is a correctness boundary, not an optimisation. The
AlphaZero package answers the same problem the other way — `training/alphazero/determinize.py`
resamples everything hidden from what is public, so its search may go as deep as it likes.
**Anything that searches must go through one of those two doors.**

**`models/champion.pt` has not loaded since the affordability block landed.** It was promoted
at `encoder.SIZE == 1868`; the encoder is now 1884, so `training.champion.load()` returns
`None` and both interfaces silently fell back to the heuristic for a while. Nothing raised.
`training.alphazero.network.graft` reconciles it — the only wrong-shaped tensor is
`context_mlp.0.weight`, and inserting the 16 new columns as **zeros** reproduces the original
function exactly — and the interfaces now offer the grafted model. Check `champion.load()` is
not `None` after any observation change.

⚠️ **`models/champion.json` says 71.6% against the heuristic; it measures 49.3% today**
(150 games, [41.4, 57.3]). Not a broken graft — that was checked three ways. Commit `e4b0441`
restricted pre-roll development-card plays to the Knight *after* the champion was promoted, so
the recorded number belongs to a slightly different game. This is the rule below about the
fixed yardstick, biting for real: **a `beat_heuristic` figure is only comparable within one
version of the rules.** Re-measure before trusting any number in a champion record.

**Benchmark persistent collectors with warm-up calls.** Transitions bank in bursts when games
finish, so a short measurement measures luck. An early benchmark reported 4 workers as faster
than 8 for exactly this reason.

**Windows spawns rather than forks.** Worker functions must be module-level; a script run from
a heredoc cannot be a multiprocessing parent because the child cannot re-import `__main__`.
Torch sizes its OpenMP pool at import, so thread env vars must be set in the *parent* before
the pool is created.

**A `\b` inside a JavaScript template literal is a backspace**, not a word boundary. The
resulting pattern matches nothing while reading as though it works. There is a test for stray
control characters in `app.js`.

**`python -m` with a heredoc and a pipe buffers output.** Use `python -u`, and read
`checkpoints/metrics.jsonl` for training progress rather than the piped stdout. `| tail` is
worse than buffering — it prints *nothing* until the process exits, so a three-hour run looks
hung. Redirect to a file and read the file.

**Self-play samples bank in cohorts, not continuously.** A worker's `envs_per_worker` games
advance in lockstep — one simulation each per round — so they also *finish* together. Nothing
comes back for the first ~30 s of worker time and then several thousand positions arrive at
once. Two consequences, both measured:

* a `generate(positions=N)` call takes a wildly variable time (5 s for one worker, 46 s for
  another on the same request), and `pool.map` waits for the slowest, so most of the pool
  idles. **Time-box the slice instead** — `generate(seconds=...)`. Count-based sharing measured
  172 positions/sec across 14 workers where the clock-based one gives ~400.
* `envs_per_worker` is a latency knob as much as a batch-size knob.

---

## Agents and training

**The heuristic is the behaviour-cloning teacher**, so its judgement is where the trained
policy starts. Improving it raises the ceiling for both. It is also the fixed yardstick — if
you change it, *win rates recorded against it before and after are not comparable*. Say so
when reporting.

**Resource weights are tuned for 2-player 15-point play and invert 4-player folklore.**
Missing brick is the worst opening deficiency (36% win rate), missing wheat nearly free (49%).
With no player trading, a missing resource costs 4:1 at the bank, so expansion gates the game.
Do not "fix" these back toward wheat-and-ore without 1v1 evidence.

**Fine-tuning from a clone is not training from scratch.** Use `lr 3e-5`–`1e-4` and
`entropy 5e-3`. The defaults will destroy what cloning learned before the outcome signal
replaces it.

**Warm-started runs look like failures for their first ~100 iterations.** One went
27.8 → 21.9 → 31.6 → 31.1 before climbing to 77.6%. Another started *below* its predecessor
and overtook at iteration 79. Judged at iteration 59 both would have been abandoned. Check
whether the intervals overlap before concluding anything.

**Two lineages, two champions.** `models/champion.pt` is PPO; `models/champion_az.pt` is
AlphaZero. They are separate files on purpose and the interfaces offer both. Do not train one
into the other's file.

**The AlphaZero package is where search lives, and its shape follows from the game.** Dice are
chance nodes keyed by roll *total*, resampled per visit. Values propagate in seat 1's frame
rather than flipping by depth, because the game is not alternating — during a discard the mover
may be either player. `Search` refuses `num_players != 2` for that reason. Forced moves
(≈30% of decisions) are collapsed during descent and never recorded: a one-hot policy target
teaches nothing and costs a network evaluation.

**The AlphaZero run is warm-started, and that is a choice, not a default.** At the simulation
counts a CPU affords, MCTS is a modest improvement over its prior, so starting from a policy
that already plays is what makes a few hours worth anything. `--cold` does it the guide's way.
Say which one produced a number.

⚠️ **The AlphaZero loop's win-rate column measures the raw policy, and the champion plays with
search. They move in opposite directions — twice measured, in both directions.** In the run on
the 2,503-float observation the policy score went 62.5 → 60.4 → 50.5 → 51.0 and read as "peaked
at iteration 20". Ranked *with search* on identical games, the order was almost exactly
reversed: iteration 95 best at 74.7%, iteration 20 **worst** at 69.0%. A declining policy
column is not evidence a run has peaked. The earlier measurement: On identical games, between two checkpoints of one
run: raw policy 47.4% → 43.8%, the same weights *with 32 simulations* 57.1% → **60.8%**. The
value head is what search leans on hardest and it was still improving while the policy's
argmax drifted. So `checkpoints/alphazero/best.pt` is best-*policy*, not best-*player* —
re-measure candidates with `training/alphazero/arena.py` before promoting, and never read the
in-loop curve as the run's progress.

⚠️ **When the policy will not learn, measure the labels — not the loss curve.** A run at 48
simulations with AlphaZero's noise of 0.25 sat flat for 60 iterations. Everything looked
plausible and everything *was* correct. The answer came from asking how often the recorded
target picks the same move as a clean 400-simulation search:

```
 48 sims, noise 0.25   62%      <- what it was training on
 48 sims, noise 0.10   68%
 96 sims, noise 0.10   78%      <- now the default
160 sims, noise 0.10   80%
raw policy argmax      58%      <- what it already knew
```

Four points of signal. Root noise alone flipped 24% of the top moves — 0.25 is calibrated for
800 simulations, where it barely perturbs the visit counts, and is destructive at 48. Loss
curves and win rates cannot tell "learning slowly" from "learning from nothing"; this
measurement takes three minutes and does.

**The champion is not the newest model.** `models/champion.pt` changes only through
`training.champion promote`, which requires the Wilson lower bound over 400 games to clear
50% *and* no regression against the fixed heuristic. The gate has already refused a completed
run. Never copy a checkpoint into `models/` by hand.

⚠️ **The gate does not run when there is no loadable champion.** `promote` takes its
`reigning is None` branch and installs immediately — no Wilson bound, no regression check — and
then overwrites the `beat_heuristic` baseline that every *later* candidate is compared against.
This fires exactly when `encoder.SIZE` has changed, because the reigning champion no longer
loads. So after any observation change the first promotion is ungated by construction: measure
against `HeuristicAgent(noise=0)` by hand first, and say in the record that the baseline was
reset.

---

## What has been tried and did not work

Recorded so it is not re-attempted.

- **1-ply lookahead with the value head.** Leak-safe, correct, 53.4% vs 52.2% over 800 games.
  Probably because the trunk is shared, so `V(s')` restates what the policy already encodes.
- **The opening bug fix alone.** Improved every structural measure of the opening and won no
  more games, because the road threshold prevented the agent from acting on it.
- **Gather-and-pool over embeddings** for neighbour aggregation: 3.8 ms at batch 512 for one
  relation, where the whole flat trunk cost 6.6 ms. Constant incidence matrices on raw
  features do the same job in 0.33 ms.
- **Behaviour cloning on a noisy teacher's actual moves.** Caps achievable agreement at the
  71.2% the noisy teacher shares with the noiseless one. Play noisy, label clean.

---

## Where to look

| | |
|---|---|
| Why something is the way it is | `docs/decisions/` — 23 records |
| What is done and what is next | `ROADMAP.md` |
| Whether a change helped | `training/evaluate.py`, and use enough games |
| How fast anything is | `python -m benchmark.benchmark`, and warm up first |
| How a training run went | `python -m training.alphazero.report` |
| Where the time goes | `python -m benchmark.profiler selfplay` |
| What the bot did in a real game | `python -m interfaces.web.recorder --margin 5` |

Run the full suite before committing: `python -m pytest tests -q` (~2 min). Two tests are
timing-based and flake under load — re-run them alone before believing a failure.

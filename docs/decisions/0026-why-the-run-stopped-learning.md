# 26. Why the run stopped learning, and what actually fixed it

Date: 2026-08-05
Status: accepted

## Context

Five AlphaZero runs in a row produced no measurable improvement. The symptom was reported as
"it is stagnating", and the loss curves agreed: `policy_loss` sat between 1.22 and 1.24 from
the first iteration to the last, in every run, for hours.

`CLAUDE.md` has a rule for exactly this — **when the policy will not learn, measure the
labels, not the loss curve** — and record 0023 has the measurement it refers to. Running it
again is what found the cause, and the cause was not where the loss curve pointed.

## What was measured

### The signature, visible in all five runs

`policy_loss` is the cross-entropy `H(target, pred)`; `entropy` is `H(pred)`. Across every
logged iteration of every run the two sat within 0.005 of each other. That is what happens
when a network's output has become its own training target.

### The label had stopped teaching

Decision 0023's table, re-measured on the champion, paired over 400 positions, against a
clean 400-simulation reference:

| setting | 0023 recorded | re-measured | vs prior | McNemar p |
|---|---:|---:|---:|---:|
| raw policy argmax | 58% | **76.2%** | — | — |
| 48 sims, noise 0.25 | 62% (+4) | 72.2% | −4.0 | 0.10 |
| 48 sims, noise 0.10 | 68% (+10) | 77.0% | +0.8 | 0.79 |
| **96 sims, noise 0.10** | **78% (+20)** | **80.5%** | **+4.3** | 0.036 |
| 160 sims, noise 0.10 | 80% (+22) | 78.0% | +1.8 | 0.47 |

The raw policy rose 58% → 76.2%: training had worked, and the network had absorbed its own
search. But the headroom that justified 96 simulations **collapsed from +20 points to +4.3**,
which does not survive correcting for the five settings compared. More simulations did not
restore it. Positions offer 9.6 legal moves on average, so 96 PUCT simulations spend most of
their budget confirming the prior and hand it back.

### The cause was the value head

Search's only source of discrimination is `Q`, and `Q` is built from the value head. Measured
on 176,342 held-out positions from 804 games the network never trained on:

| | value MSE | sign accuracy | variance explained |
|---|---:|---:|---:|
| champion, held out | 0.8315 | 72.8% | **14.4%** |
| champion, on the training buffer | 0.28–0.34 | — | — |
| predicting nothing (`V = 0`) | 0.9710 | — | 0% |

A three-fold gap between training and held-out error, and 14% of the variance explained. The
mechanism is specific and was hiding in plain sight: **the board is constant within a game and
is in the observation**, the buffer held 180,000 positions from only about 900 distinct games
(~195 searched decisions each), and 60 batches x 512 draws per iteration showed the network
each game about 33 times per iteration. It learned to recognise the board and recall the
result. Record 0021 warned about precisely this failure mode for the flat network's *policy*;
it had reappeared in the *value*.

So the chain was: too few distinct games → the value head memorises → `Q` carries no signal →
the visit distribution is the prior → the policy target teaches nothing → the loss curve is
flat.

## Decision

Six changes, all aimed at the value head and the data behind it. **None at capacity** — record
0025 had already measured that this game's data budget does not support more parameters.

| change | setting | why |
|---|---|---|
| Auxiliary heads | `owner_weight` 0.15, `margin_weight` 0.15 | Final ownership of every vertex and road, and the final VP margin. Dense, per-element, and **impossible to answer by recognising the board** — which is the point. Largest item in KataGo's ablation at 1.65x, and 551 parameters here. |
| Value target blend | `root_value_weight` 0.5 | The outcome is one bit shared by ~195 decisions. The search's root value is the same quantity per decision, at far lower variance. Not a bootstrap: recorded mid-game in the mover's frame, nowhere near the terminal-state invariants. |
| Playout cap randomization | 0.25 at 96 sims, 24 otherwise | KataGo's 1.37x, but adopted for its second-order effect: a game contributes a quarter as many rows, so the buffer holds ~4x as many distinct games. That is the direct fix for the memorisation. |
| Per-game batch cap | `max_per_game` 8 | No game may supply more than 8 of a 512-row batch. |
| Architecture | 128/64, depth 3, trunk 192 — 375,106 params | Record 0025: `depth` 2→3 halved value error for +37k parameters, and 57% of the old parameters sat in the pooled trunk. |
| Evaluation | vs `models/champion_az.pt`, snapshots ranked by the arena | The question a run must answer is whether it beat the player already on disk. |

The network keeps `aux=False` as its default so every existing checkpoint rebuilds
identically; the AlphaZero path turns it on.

## The run

`configs/train_v2.yaml`, warm-started by distilling the champion into the new shape
(`training/alphazero/distil.py`, since `graft` cannot cross a width change). The distilled
network measured **49.3% [42.8, 55.8]** against the champion — parity — so the run started
level and everything below is attributable to the training changes.

257 iterations, **15,764 games**, 672,956 positions, 180.7 minutes. The previous two-hour run
managed 4,403 games; this is about 3.6x the games per hour, which is the diversity fix.

### Result

Against `models/champion_az.pt`, 400 games, search on both sides, zero truncations:

| framing | win rate | 95% CI |
|---|---:|---:|
| equal simulations (64 v 64) | 68.50% | [63.8, 72.9] |
| **equal wall-clock (56 v 64)** | **67.25%** | **[62.5, 71.7]** |

Both framings are given because the candidate is the larger network: measured
single-threaded it costs 1.15x per move (58.1 ms against 50.7 at 64 simulations), so equal
simulations flatters it. **The Wilson lower bound is 62.5% even after handicapping it.**

Accumulated across the run rather than arriving at once — snapshots at 120 games each:
49.2 → 55.1 → 52.5 → 62.2 → 51.7 → 56.8 → 56.7 → 65.0 → 63.6 → 65.0 → 63.3 → 65.5.

Against the fixed heuristic at noise 0, 300 games, measured today on identical games for
both — because `CLAUDE.md` is right that a recorded `beat_heuristic` is only comparable
within one version of the rules:

| | vs heuristic |
|---|---:|
| champion | 80.13% [75.2, 84.3] |
| **az_v2** | **92.00% [88.4, 94.6]** |

And the quantity the whole diagnosis rested on, on the same held-out set as before:

| | value MSE | sign | variance explained |
|---|---:|---:|---:|
| champion | 0.8315 | 72.8% | 14.4% |
| **az_v2** | **0.6276** | **77.0%** | **35.4%** |

`policy_loss` fell 1.179 → 1.139 and was still falling at the end, against a flat 1.22–1.24
in every previous run.

## What did not work

**The Gumbel root — the change this record expected to matter most.** Gumbel-Top-k with
Sequential Halving and completed-Q targets (Danihelka et al.) is designed for exactly the
regime here: few simulations, a policy target that has stopped improving. It is implemented,
it is tested, and it is **off**, because it measured much worse. Paired, 250 positions,
agreement with a clean 400-simulation search:

| label | agrees | vs prior | p |
|---|---:|---:|---:|
| raw policy | 72.8% | — | — |
| PUCT 96, noise 0.10 | 73.2% | +0.4 | 1.00 |
| Gumbel, `c_visit` 0.5 | 59.2% | −13.6 | 0.0019 |
| Gumbel, `c_visit` 5 | 54.4% | −18.4 | <0.0001 |
| Gumbel, `c_visit` 50 (the paper's) | 49.2% | −23.6 | <0.0001 |

Monotone in the scale, and the small-scale limit is no escape: as `sigma` goes to zero the
target degenerates to the prior, which is 72.8%, still no better than PUCT.

**Why, and it is not an implementation bug.** Gumbel's improvement guarantee needs `Q` to
carry signal. Here every simulation resamples a *different determinized world* — a fresh guess
at the opponent's hand and the dice deck — and scores it with a value head that explained 14%
of the variance. `sigma(q)` multiplies `Q` differences by tens, so it amplifies
determinization variance into a near-one-hot target. Visit counts lean on `Q` far more weakly
and survive it.

Kept in the code, off, with tests. **Worth re-testing now**: it failed because `Q` was noise,
and `Q` is measurably less noisy than it was.

## The honest qualification

The causal story above is what motivated the changes, and it is only partly what happened.
Re-measuring the label on the new network, paired over 300 positions:

| setting | agrees | vs prior | p |
|---|---:|---:|---:|
| raw policy argmax | 81.3% | — | — |
| 96 sims, noise 0.10 | 82.0% | **+0.7** | 0.85 |
| 160 sims, noise 0.10 | 83.0% | +1.7 | 0.51 |

**The policy label still teaches almost nothing.** The headroom did not recover. So the 18
points did not come from better policy targets; they came from the auxiliary targets and the
root-value blend giving the network something else to learn, and from a better value head
making the search *play* better moves, which produced better games and better outcome labels.
The improvement channel was the critic and the data, not the label.

That is worth stating plainly because it changes what to do next. The 96-simulation visit
distribution remains near-information-free as a policy target, and that is now the largest
identified opportunity in the pipeline.

Also worth stating: `value_loss` in `checkpoints/az_v2/metrics.jsonl` is measured against the
**blended** target, so its ~0.10 is not comparable to the 0.28–0.34 of earlier runs. The
comparable number is the held-out MSE above.

## What to do next

1. **Re-test the Gumbel root.** It failed for a reason that has measurably weakened.
2. **Attack the policy label.** It is +0.7 points over the prior; everything else in the
   pipeline is now healthier than it is.
3. **Re-run the capacity sweep.** Record 0025 measured capacity against a value head that
   explained 14% of the variance. At 35% the answer may have moved.

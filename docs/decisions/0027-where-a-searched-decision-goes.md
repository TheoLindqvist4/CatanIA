# 27. Where a searched decision actually goes

Date: 2026-08-05
Status: accepted

## Context

Record 0025 measured the self-play loop as 80.9% engine and 19.1% network, and its action
point 5 asked somebody to profile the 80% — "a 20% engine win is worth more than the entire
network-side optimisation ceiling of 1.24x". This is that profile, and the changes it paid
for.

It also answers a question that was asked directly: *is generating the Catan board what takes
most of the time, and should boards be pre-generated?* The answer is no, by about four orders
of magnitude, and the measurement is recorded here so nobody has to wonder again.

## Where the time goes

`perf_counter` brackets accumulated *inside* a warmed, clock-boxed `Generator` at the shipped
settings — 96 simulations, `envs_per_worker=12`, one torch thread, the real champion — with
nesting flags so `clone` and `legal_actions` are not counted twice. Shares are the robust
quantity; the ms column normalises them onto record 0025's 41.7 ms per searched decision so
the two tables can be read together.

| component | share | ms / searched decision | calls / decision |
|---|---:|---:|---:|
| leaf `encoder.encode` | **23.2%** | 9.67 | 96.0 |
| network forward, batched | 19.2% | 8.00 | 8 batches / 96 rows |
| `rules.legal_actions` inside the search | **15.8%** | 6.59 | 172.9 |
| leaf `np.fromiter` — the list→float32 conversion | **8.9%** | 3.71 | 96.0 |
| `state.clone` inside the search | 6.6% | 2.74 | 306.6 |
| `rules.apply` | 5.4% | 2.39 | 124.6 |
| `rules.roll_dice` | 5.0% | 2.07 | 211.3 |
| `env.step`, everything it does | 0.7% | 0.29 | 1.31 |
| `determinize` | 0.1% | 0.05 | 1.31 |
| residual — MCTS bookkeeping, batching, Python | 15.1% | 6.29 | — |

Three cross-checks say the instrument is honest: the network share reproduced record 0025's
19.1% independently; `env.step` matched a separate probe's 0.714%; and `rules.legal_actions`
came out at 41.5 µs a call against the 42 µs already written down in `mcts.py`.

**The correction this makes to record 0025's picture.** "Engine" is not one thing. *The
observation is 32.1% of a searched decision — more than the network — and legal-move
generation is another 15.8%.* Two functions are half the loop.

## What was done

Six changes. Every one produces a bit-identical observation or a provably identical action
list; none of them changes what the agent plays.

**1. Chance nodes stop re-rolling a roll that is already determined — the largest single item.**
`Search._sample_roll` cloned the state and rolled the dice on *every* visit, then looked the
child up by total. Under the shipped ruleset that work has one possible answer:
`RANKED_1V1` uses the Balanced Dice deck, the deck is consumed rather than resampled,
`clone` copies it verbatim, and `draw_balanced` pops the *last* card. Measured over **9,002
real chance nodes, 24 resamples of each produced exactly one distinct total, every time**;
the same measurement under `BASE_GAME` gives 5–11, which is why the fast path is guarded on
`dice_deck` being a deck rather than `None`. 77% of chance-node visits were re-deriving
`dice_deck[-1]`.

⚠️ This is **distribution-preserving, not bit-preserving**, and the reason is worth knowing:
a discarded clone's deck could fall to `RESHUFFLE_AT` and call `new_deck(state.rng)` on the
*shared* generator, so those throw-away clones were advancing the real game's random stream.
A fixed seed now produces a different — equally valid — game.

**2. The observation is written as float32 instead of being converted to it.** `encoder.encode`
returns a list of Python floats and the search then spent 42 µs a leaf in `np.fromiter`
unboxing them. `encode_into` writes the same numbers into an `array('f')`, which numpy takes
as a buffer: the template reset drops from 8.3 µs to 0.26 µs and the conversion from 42 µs to
0.5 µs, against roughly 10 µs more for the narrower scalar stores. `encode` is untouched and
still returns a list of Python floats — 13 tests assert that, and every interface indexes it.

**3. One walk over the board instead of nine.** `_encode_players` called
`victory_points`, `public_victory_points`, `production_rates`, `trade_rates` and
`longest_road_length` per player; between them that is nine passes over the 54 vertices to
answer three questions about the same buildings. `encoder._survey` answers all three in one.

⚠️ This makes the encoder **a second implementation of scoring**, which the one-source-of-truth
rule otherwise forbids. It is legal only because of
`test_fused_player_scores_agree_with_the_rules`, which drives whole games at two, three and
four players in both rulesets and asserts every value equals the rules' own at *every*
position. `catan.rules` remains the authority.

**4. The vertices and roads blocks are encoded together.** They needed the same walk — which
vertices my roads touch decides both a vertex's "reachable" flag and a road's "I could build
here" — and were doing it twice, plus a third pass to turn one into the other.

**5. `board.expected_production` is computed once per board.** It is a pure function of a
frozen layout and was being recomputed on every call, once per owned vertex per encode.
`check_id` deliberately stays *outside* the cache: indexing a per-vertex table with an
unvalidated id makes `table[0 - 1]` vertex 54's row, which is the exact
`VERTEX_TILES`/`TILE_VERTICES` failure mode `CLAUDE.md` records.

**6. `legal_actions` stops re-checking payment 72 times.** The BUILD branch already gates the
road loop on pieces-left and affordability, and then asked `can_build_road`, which re-checks
both for all 72 roads. `can_build_road` remains the authority for every other caller and
`test_the_road_loop_agrees_with_can_build_road` pins the loop to it, action for action.

### Measured together

End to end, against a baseline that is *this* working tree with only these six changes
reverted — not against `HEAD`, which would have measured the concurrent record-0026 work at
the same time. Arms alternated, and the order alternated within each pair, per the warning
below. Both arms load the real champion and run the shipped settings.

| | pair ratios | median | won |
|---|---|---:|---:|
| quiet machine, 5 pairs | 1.352, 1.393, 1.352, 1.369, **0.609** | **1.35x** | 4/5 |
| loaded machine, 7 pairs | 1.733, 2.993, 1.084, 1.044, 1.320, 1.813, 1.429 | 1.43x | 7/7 |

**11 of 12 paired comparisons favour the change.** The one loss is kept rather than dropped:
its optimised arm delivered 1,231 leaf-evals/sec where identical code gave 2,940–2,952 in
three other pairs of the same run, which is a machine event, not a property of the code. The
second run's absolutes are a third of the first's — something else was using the box — so its
ratios are noisier in both directions and its median is not the number to quote.

**The number to quote is 1.35x**, from the quiet run: 41.7 ms → **~31 ms per searched
decision**.

It is larger than the sum of its parts predicted, so here is the arithmetic that says it is
real rather than a fluke. The encoder half was measured directly and separately, interleaved,
min-of-9-rounds, over 30 real positions, and verified bit-identical on every one:

```
baseline   encode + np.fromiter        166.71 us
optimised  encode_into + frombuffer     87.48 us
cut                                       47.5%   (79.2 us x 96 simulations = 7.6 ms)
```

47.5% of a block worth 32.1% of a decision is **15.2%**. Add the chance-node fast path (6.5–7.0%,
measured end-to-end by an independent paired A/B that won 14/14) and the road loop (~2.6%,
derived from a 0.653x ratio on 78 real BUILD states): 24.5% of the decision removed, i.e.
**1.32x predicted against 1.35x measured**. The two agree.

## The board-generation hypothesis, refuted

The question was whether generating the Catan board is what takes most of the time, and
whether pre-generating a pool of boards would help.

| | |
|---|---:|
| `Board(rng)` | 126 µs |
| `GameState` with a fresh board | 161 µs |
| `env.reset(seed)`, the whole per-game setup | 644 µs |
| one self-play game at 96 simulations | **17.3 s** |
| **board construction as a share of self-play** | **0.0018%** |
| the whole board-static path, including the template build | 0.0051% |

A board is built **once per game** — a 300-second instrumented slice built **20 of them,
5.40 ms in total** — because `GameState.clone` shares the board object by reference, so none
of the ~429,000 encodes, none of `determinize` and none of the ~21,000 clones per game ever
build one. A pool that also pre-warmed the static template was prototyped and measured
end-to-end at the most favourable workload a pool can ever see: **+0.64% ± 2.26%**. The noise
floor is 28× the effect.

It would also cost something real. 20,000 seeded boards produce 20,000 distinct layouts with
zero collisions, so today every game gets a board it has never seen. A pool small enough to be
free (N=64, one copy per spawned worker) would reuse each board ~31 times, duplicating
*opening positions* in the replay buffer — the one place duplication hurts most, because
opening placement is where `temperature=1.0` sampling exists to create variety. And a seed
would stop pinning the game.

**The instinct behind the question was right and aimed one level too high.** The thing
recomputed 429,000 times per 300 seconds is not the board, it is the observation — and the
board-static 40% of *that* is already cached by `encoder._static_template`, amortised over
21,462 encodes per build. Items 2–5 above are the same idea applied where the recomputation
actually is.

## The in-loop evaluation is off

`Trainer.evaluate` played 100–200 games against a yardstick every `evaluate_every`
iterations. Measured from five real runs' own `metrics.jsonl`, by differencing the unaccounted
residual on evaluating iterations against non-evaluating ones:

| run | evaluations | cost each | share of the run |
|---|---:|---:|---:|
| `az_run_1h` | 4 | 21.0 s | 2.33% |
| `az_run_1h_b` | 4 | 16.8 s | 1.87% |
| `az_run_2h` | 9 | 19.2 s | 2.40% |
| `az_stage1_run` | 4 | 19.9 s | 3.62% |
| `az_stage2_run` | 5 | 29.2 s | 4.38% |

All of it is idle time — the parent plays sequential games at batch 1 while all fourteen
workers wait. And the number it produces is one `CLAUDE.md` already says not to act on: it
scores the **raw policy**, which has twice been measured moving in the opposite direction to
search-ranked strength.

`evaluate_every: 0` now means off, and is the default. The only thing the evaluation was
load-bearing for was the `best.pt` write, so the run snapshots to `<run>/snapshots/` on the
existing checkpoint timer instead, and `training/alphazero/arena.py` ranks those afterwards
*with search* — which is the only ranking this repository treats as evidence. Set it back to
25 to get the smoke alarm during a long unattended run.

The dashboard was the other candidate and it is **free**: a separate process, two files opened
read-only, 13.3 ms. Deleting it would save nothing and lose the only artifact of a run you can
open tomorrow.

## A warm start builds the checkpoint's shape, not the configured one

Found while checking record 0025's sizing change, and it is the reason that change had not
taken effect.

`load_for_alphazero` builds from `checkpoint["config"]`. That is exactly right when the
*observation* has changed — `graft` widens the affected layers with zero columns and every
weight keeps its meaning. It cannot help when `width`, `depth` or `trunk` change: every tensor
then has a different shape and there is no column correspondence to preserve. `trunk` also
*shrinks* under 0025's recommendation, so no function-preserving widening exists either.

The consequence, reproduced rather than deduced: with `warm_start: models/champion_az.pt` set,
a run started after the defaults changed trained the reigning **200,379**-parameter shape, not
the configured 374,331 — and with `aux=False`, so the auxiliary targets self-play was
computing were dropped by the loss. Nothing raised.

`build_network` now compares the loaded network's geometry against a freshly built one and
prints every key that differs, the two parameter counts, and the two ways out (distil, or
`--cold`). Compared against `new_network()` rather than a written-down table, so it cannot go
stale when the signature changes — which is the event it exists to notice.

**Not made an error.** Carrying a trained policy forward is often worth more than the shape
you asked for. It has to be a decision somebody took, which is different from being silent.

## What was measured and is not worth doing

Recorded so it is not re-attempted.

| | the number that says so |
|---|---|
| Pre-generate a pool of boards | 0.0018% of wall clock; the pooled A/B measured +0.64% ± 2.26% |
| Delete the dashboard to save training time | 13.3 ms, separate process, 0.000% of the run |
| Optimise `catan/env.py` for throughput | the *whole* of `env.step` is 0.7% of a decision |
| Fix the `_advance_to_decision`/`legal_mask` duplicate `legal_actions` call | real, and 0.002% of a decision |
| Chase `rules.longest_road_length` | already memoised: 0.59 µs on a hit against 94.10 cold |
| Replace `np.fromiter` with a different converter | every alternative measured equal or worse — the pass has to go, not be swapped |
| Reuse one observation buffer across searches | no faster, **and it aliases**: `Generator.run` stacks all `width` pending observations at the end of a round |
| A full incremental/delta encoder | deltas are tiny (9.3 floats for one action) but 14.6% of transitions flip the mover and move 85.1 floats; it is a second complete encoder whose defining property is that hidden information is impossible |
| A per-search cache of the ownership span | the one idea here that fails *silently* — a stale key returns wrong floats and the leak tests cover `encode()`, not the cache. Measured only in an untrustworthy window. Not built. |

And one methodological result, which cost more time than any of the above:

⚠️ **A sequential A/B cannot be run on this machine.** Throughput decays monotonically under
sustained load — 594.9 → 422.5 → 425.6 → 351.2 → 353.2 decisions/sec across five minutes — so
whichever arm runs second loses. A round-per-variant design returned median 1.0013x with range
[0.843, 1.170] on a change that is really worth 5.4%. **Alternate the arms and compare medians
of adjacent pairs**, or do not write the number down.

And the corollary, which cost an hour of confusion here:
**`test_sharing_the_stream_makes_cloning_much_cheaper` asserts an absolute
`shared < 10 us`, so it fails whenever anything else is using the box.** During this work it
read 15.8 µs against the 4.4 µs the same unchanged `GameState.clone` measured on a quiet
machine — a live 14-worker training run was in progress. `CLAUDE.md` already says to re-run
the timing tests alone before believing them; that is not enough, because *alone* does not
mean *unloaded*. Check `tasklist | grep python` before concluding anything from it, and do not
loosen the threshold: the number is right, the machine was busy.

## What is left, and where it is

`rules.legal_actions` is still **15.8% of self-play wall clock at 173 calls per searched
decision**, and item 6 takes only ~2.6 points off it. The rest would need incremental
legality — deriving a child's legal set from its parent's plus the action applied — which is a
project, not a plan item. That block is the answer to record 0025's action point 5: it is where
the remainder of the 80% is.

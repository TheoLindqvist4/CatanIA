# 25. How big should the network be?

Date: 2026-08-05
Status: accepted

## Context

The question was asked directly: what is the recommended size of the network, and what is the
biggest one this project can run so that it plays best? AlphaZero's chess network is the
reference everybody reaches for, so the honest answer needs two halves that are usually
conflated — what the literature establishes about network size, and what *this* machine
affords. They point the same way, and it is not the way the reference suggests.

The network today is `width=64, road_width=32, context=128, hops=1, depth=2, rounds=0,
trunk=256` — **200,379 parameters**. AlphaZero's chess tower is ~22.8M, trained on 5,000 TPUs.
The gap is 114x, and the instinct is that it is a deficiency.

It is not. Everything below was measured on this machine, on the 20-thread Alder Lake CPU with
no GPU, against the literature rather than from it.

## What was measured

### The self-play loop is engine-bound, not network-bound

Instrumented over a warmed, clock-boxed slice of the real `Generator` at the shipped settings
(96 simulations, `envs_per_worker=12`, one thread), split by where the time goes:

| | per searched decision | share |
|---|---:|---:|
| engine, determinize, MCTS bookkeeping | 33.7 ms | **80.9%** |
| network forward passes (96.1 rows) | 8.0 ms | **19.1%** |

The 96.1 rows per decision is a check on the instrument, not a result: it should equal
`mcts_simulations`, and it does. The 19–20% network share reproduced across two independent
measurement sessions, one of them on a fully loaded machine where every absolute number was
~4x inflated. **The ratio is the robust quantity; the absolutes are not.**

Consequence: the ceiling on *all* network-side optimisation combined is 1/0.809 = **1.24x**.
The engine has 5.2x of headroom. `python -m benchmark.profiler selfplay` is a better afternoon
than any width change.

### The exchange rate, measured in the loop rather than extrapolated

A tight-loop microbenchmark reads ~5x faster than the same network inside self-play — in the
loop the weights are evicted between calls by twelve games' worth of env state and MCTS trees,
and the evaluator pays numpy↔torch conversion. That gap is *not* a constant factor, so scaling
a microbenchmark ratio flatters the large configs. Each row below ran the actual `Generator`:

| config | params | ms/decision | throughput | vs now |
|---|---:|---:|---:|---:|
| 64/32 d2 t256 — **current** | 200,379 | 41.7 | 1.00x | — |
| 128/64 d3 t192 | 374,331 | 46.9 | **0.89x** | −11% |
| 128/64 d3 t512 | 726,651 | 57.7 | **0.72x** | −28% |
| 192/96 d3 t768 | 1,550,747 | 65.8 | **0.63x** | −37% |

**3.6x the parameters costs 28% of the data rate.** That is far cheaper than a GPU-trained
project would see, and it is entirely because of the 80% above.

It also runs the other way, and this is the more surprising half: dropping to 64,123 parameters
buys **~2%** more positions. At these widths the network is bound by PyTorch dispatch across
~370 `aten` ops, not by arithmetic. **There is nothing to gain by going smaller.**

⚠️ Measured single-process. Fourteen workers contending for memory bandwidth would penalise
the larger configs more than this table shows. Treat the ratios as optimistic bounds.

### The capacity sweep

Rather than argue from Go-derived sizes, capacity was measured directly. 176,342 heuristic
decisions were generated — deliberately matched to the 180,000-position replay buffer, because
the question is what capacity *this project's data volume* supports — and nine configurations
trained on an identical split with an identical schedule (8 epochs, `lr` 1e-3, batch 512).
Held-out agreement with the noiseless heuristic, best over epochs:

| config | params | held-out | train/test gap | value MAE |
|---|---:|---:|---:|---:|
| 32/16 d2 t128 | 64,123 | 76.95% | +0.33 | 0.2527 |
| 64/32 d2 t256 — **current** | 200,379 | 79.36% | +1.19 | 0.1219 |
| 96/48 d2 t384 | 408,827 | 79.30% | +0.88 | 0.1418 |
| 128/64 d2 t512 | 689,467 | 80.19% | +1.75 | 0.1121 |
| **128/64 d3 t512** | 726,651 | **80.28%** | +1.77 | **0.0554** |
| 128/64 d3 t512 **hops2** | 736,187 | 79.82% | +1.37 | 0.1651 |
| **192/96 d3 t768** | 1,550,747 | **81.27%** | +1.96 | **0.0495** |
| 256/128 d3 t1024 | 2,682,043 | 79.00% | +0.71 | 0.2738 |
| **128/64 d3 t192** | 374,331 | 79.74% | +1.75 | 0.0735 |

Five things fall out, and the third is the one worth acting on.

**1. Policy agreement peaks around 0.7–1.5M and falls after.** 64k → 200k buys +2.4 points;
200k → 1.55M buys another +1.9; 2.68M is a 2.3-point *regression*. Sampling error on 17,635
held-out rows is about ±0.6 points, so the peak and the fall are both real.

**2. Overfitting is not the limit.** Every train/test gap is ≤2.0 points, including at 2.68M.
Record 0021 retired the flat network on a 13.9-point gap; nothing here is close. Whatever
bounds capacity at this data volume, it is not memorisation.

**3. `depth` 2 → 3 halves the value error, for +37k parameters.** At width 128, holding
everything else fixed: policy agreement 80.19% → 80.28% (nothing), value MAE
**0.1121 → 0.0554**. This is the single largest effect in the table and it lands exactly on
the weakness `ROADMAP.md` already names — "the structured network's value head is worse than
the flat one's (MAE 0.210 against 0.074) … the likely reason PPO's one-ply lookahead bought
nothing" — and on what `CLAUDE.md` says search leans on hardest.

**4. The trunk is where the parameters are, and it is not where they belong.** Of the current
200,379:

| module | params | share |
|---|---:|---:|
| `head` — the pooled trunk, `Linear(448, 256)` | 114,944 | **57.4%** |
| `context_mlp` | 32,384 | 16.2% |
| `context_bias` | 20,640 | 10.3% |
| `policy_head` | 12,850 | 6.4% |
| **all six per-entity encoders** | **18,816** | **9.4%** |

The shared tile/vertex/road MLPs — the entire point of record 0021 — hold under a tenth of the
weights. So the sweep included a rebalanced config: width doubled, trunk cut 512 → 192. At
**374,331 parameters it reaches value MAE 0.0735 and 79.74% agreement — most of what the
726,651-parameter config gets, for barely half the parameters and a third of its throughput
cost.** `width` barely moves the parameter count; `trunk` dominates it.

**5. `hops` 1 → 2 measured *worse*.** 79.82% against 80.28%, and value MAE 0.1651 against
0.0554, at the same width and depth. This contradicts the recommendation the literature review
produced (KataGo and lc0 both favour depth-like structure over width), so it is recorded rather
than quietly dropped. One run, one schedule — suggestive, not established.

### The data budget

The 86-iteration run in `checkpoints/alphazero` generated **382,286 positions from 1,946
games**. AlphaGo Zero's 20-block network consumed 4.9 million *games*. The value target is one
bit shared across ~195 decisions, so the value head's independent label count is the game
count — about 2,000.

Published AZ runs land at 10–25 unique samples per parameter (KataGo b20c256: 241M samples at
23.4M params; b6c96: 23M at 1.0M). An hours-long run here yields 2–7M positions, supporting
**160k–700k parameters**. The current 200,379 sits mid-band. It is correctly sized for the data
this machine produces, not starved.

## What the literature actually establishes

Seven parallel research strands, each with its load-bearing numeric claims independently
fact-checked against primary sources. The working range spans four orders of magnitude and
tracks game complexity and compute budget, not orthodoxy:

| system | params |
|---|---:|
| TD-Gammon 2.1 — near-world-class backgammon | 16,244 |
| GNU Backgammon contact net — world-class *today* | 32,773 |
| **this repository** | **200,379** |
| OpenSpiel AZ default MLP (w256, d2) — DeepMind's own non-grid reference | ~287k |
| Jones's swept optimum, 9×9 Hex — *perfect play* | ~500k |
| KataGo's published floor, b6c96 | 1,004,333 |
| AlphaZero / AGZ 20-block | ~22.8M |

Four results carry weight, and the first corrects the reference everyone reaches for:

**The AlphaGo Zero "20 vs 40 block" comparison is not size-scaling evidence.** The 20-block
net trained 3 days on 4.9M games and 700k steps; the 40-block trained 40 days on 29M games and
3.1M steps. On the one matched metric the *bigger* net is worse — game-outcome MSE 0.180
against 0.177 (Extended Data Table 2). It confounds size with an order of magnitude of
training. Do not cite it for sizing.

**AGZ's only genuine fixed-thinking-time ablation found the cheaper net won.** Four variants
trained on one fixed dataset, evaluated at 5 s/move: `dual-res` (one tower, both heads) beat
`sep-res` (two towers, ~2x the per-evaluation cost) by ~600 Elo.

**Search dominates weights.** Identical AGZ weights: raw network argmax 3,055 Elo; the same
weights with MCTS 5,185. Search was worth ~2,130 Elo — more than any size change in the
literature. This repository already knows the local version: `CLAUDE.md` records the raw-policy
column and the search-ranked order moving in opposite directions, twice.

**In Catan specifically, three independent studies measured larger = worse.** Whelan's
DQN/DDQN collapsed to ~0% win rate ("larger models are not necessarily better", attributed to
data starvation). Asher's 2,896,001-parameter 10×500 residual MLP reached only "beginner" after
a week on 24 cores. Gendre & Kaneko's smaller 15-channel network beat their larger 40-channel
CNN+ResNet *at every search depth*, and the shallower one learned faster.

The one fitted scaling law for AZ agents (Neumann & Gros, Connect Four / Pentago) gives
strength ∝ N^0.88 and N_opt ∝ C^0.62: **doubling parameters demands 3.0x more total training
compute** to stay compute-optimal.

### Why the chess and Go numbers do not transfer

Catan's branching factor is ~65 — and that figure is for the *four-player game with player
trading over an 1,882-action space*. This is two-player, no trading, 325 actions, so effective
branching is materially below it. Go is 250. The "1.2×10¹⁵ Catan state space" often quoted is
the count of *initial board setups*; Go's real state space is 2.08×10¹⁷⁰ legal positions.

More decisive: **AGZ's input is 19×19×17 binary planes — eight of own stones, eight of
opponent stones, one colour. Zero derived features. No liberties, no eyes, no ladders.** The
entire job of its 22.8M-parameter tower is computing features from raw stones. This
observation ships pip potential, per-resource expected production, harbour nearness,
buildability, affordability priced through own trade rates, and cumulative production and
spending history — and hands the network the board's adjacency as constant incidence matrices
instead of making it learn one.

The best-quantified price of that substitution is Stockfish's: **SFNNv10 added `FullThreats`
input features and cut L1 from 3072 to 1024 in the same revision** — a 3x width reduction
bought back by better inputs, after four years of *growing* it. KataGo's ablation prices it
independently: removing game-specific input features costs 1.55x training time. Record 0024
made exactly the Stockfish move. **Feature work outranks width work**, and that is the single
strongest reason not to size this network from Go.

## Decision

**The recommended configuration is `width=128, road_width=64, context=192, hops=1, depth=3,
rounds=0, trunk=192` — 374,331 parameters, 1.9x the current count, costing 11% of the data
rate.** It captures the one large effect in the sweep (depth 3 halving value error) and pays
for it by cutting the over-weighted trunk rather than by buying throughput.

**The upper option, if a run can afford 28% fewer positions, is `trunk=512` at the same width
and depth — 726,651 parameters.**

**The maximum worth running is ~1.5M** (`192/96 d3 t768`). Above it three things break at once:
the dispatch tax is fully amortised so cost becomes near-linear in width; the data budget stops
supporting it (10–25 samples/parameter needs ~15M positions, more than any run this project
does); and N_opt ∝ C^0.62 says the compute requirement has risen ~26x. The 2.68M row measured
worse on every axis.

**Do not shrink.** A third of the parameters buys ~2% more positions.

**Do not raise `hops` or `rounds`.** `hops=2` measured worse here; `rounds=1` is already
recorded in `CLAUDE.md` as costing more than the rest of the network for nothing.

**Do not buy more simulations.** Record 0023's label-quality curve is flat above 96: 96 → 160
buys 2 points of agreement for 1.67x the cost.

**Do not pursue quantization.** Measured on this silicon: bf16 7.4–17x *slower*, fp16
12.6–22.9x slower, int8 dynamic quantization 0.43–0.91x, and `torch.compile`'s CPU backend
needs `cl.exe`, which is absent. Alder Lake consumer parts have no AVX-512, no AVX512-BF16 and
no AMX. It cannot buy simulations back here.

### A rule this establishes

KataGo's own numbers contradict each other in a way that matters: b40c256 is +158 Elo over
b20c256 *per playout*, yet lightvector states it is "not as strong yet per-equal-compute-time"
at low-thousands of playouts, and warns that ratings "based on equal search nodes rather than
GPU cost" unfairly favour the larger net.

> **Every network-size comparison in this repository must run through
> `training/alphazero/arena.py` at fixed *seconds per move*, never fixed simulations.**

At fixed simulations a bigger network will essentially always win, and the gate will promote
one this machine cannot afford. This compounds the existing warning that `best.pt` is
best-*policy*, not best-*player*.

## Action points, ranked by measured leverage

1. **Gumbel root — Gumbel-Top-k, Sequential Halving, completed-Q policy targets.** The highest
   leverage available and it costs nothing at inference. MuZero *fails to learn* at ≤16
   simulations on 9×9 Go; Gumbel MuZero learns reliably at 2. MiniZero: at equal wall-clock on
   8×8 Othello, Gumbel-AZ at n=2 matched AZ at n=200. At 96 simulations over 325 actions this
   project sits exactly where the visit-count target stops being a guaranteed policy
   improvement. Either better labels at 96, or the same labels at 32–48 — i.e. 2–3x the
   positions per hour.
2. **Auxiliary targets.** KataGo's largest single ablation item: removing owner and score
   targets costs **1.65x** training time, ahead of global pooling (1.60x), game-specific input
   features (1.55x) and playout cap randomization (1.37x). The Catan analogues cost a few
   hundred parameters because the positional heads already exist — a per-vertex and per-road
   "who owns this at game end" head off the existing `Linear(d, 2)` / `Linear(d, 1)` features,
   and a VP-margin head beside win/loss. This also attacks the value head directly.
3. **`depth` 2 → 3 and the rebalanced trunk** — the sizing change above. Value MAE
   0.1219 → 0.0735 for 11% of throughput.
4. **Playout cap randomization**, p=0.25 full search at 96 and ~20 otherwise. 1.37x measured,
   and unlike everything else here it cuts *engine* time too — which is the 80%.
5. **Profile the engine.** The 80% has never been profiled at this granularity. A 20%
   engine win is worth more than the entire network-side optimisation ceiling of 1.24x.
6. **Grow width only by KataGo's concurrent-loss-parity rule** — train the next width on the
   same replay buffer and switch when its average loss catches the smaller's.
   `network.graft` already widens observation-width layers with zero columns, so this
   repository can execute it. That answers the sizing question empirically, mid-run, rather
   than by argument.

## Caveats, stated rather than buried

**The capacity sweep is imitation, not reinforcement.** Agreement with a fixed heuristic
saturates at the teacher's ceiling, and the AlphaZero policy is meant to *exceed* the
heuristic. The sweep is evidence that 200k is not starved and that 2.68M is not usable; it is
not proof that a larger network could not help under an outcome objective. The value-MAE
column is the less ceiling-bound half and is where the depth effect shows.

**One learning rate for every configuration.** All nine ran at `lr` 1e-3. The 2.68M collapse
has the signature of underfitting, not overfitting — its train/test gap was the *smallest* in
the table at +0.71 — so it is probably an optimisation artifact of too high a rate for that
size, not a capacity wall. The throughput and data-budget arguments against 2.68M stand on
their own; that row is weak evidence and is not what the ceiling rests on.

**Value MAE is noisy epoch to epoch.** The 192/96 config read 0.0495 at epoch 6 and 0.0716 at
epoch 7. The table reports the best of 8. Differences under ~0.02 should not be read.

**None of this has been run through the arena.** These are supervised proxies and throughput
measurements. Per the rule above, the sizing change is a *candidate*, and the promotion gate
over 400 games at fixed seconds per move is what decides it.

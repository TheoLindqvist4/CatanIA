# 0021 — A network that knows the board has a shape

**Status:** accepted · **Date:** 2026-07-30 · **Phase:** 8

## The problem

`PolicyValueNet` treats the 1808-float observation as an unordered bag of numbers. It is not
one. `catan.encoder` lays it out as regular blocks — 19 tiles × 19, 54 vertices × 16, 72
roads × 6 — and `catan.action_space` lays the 324 actions out the same way: 275 of them
(84.9%) name a board element.

Two costs follow.

**Everything is learned 54 times.** What "a vertex with three high-pip tiles and nobody built
next door" means is the same fact at vertex 3 and at vertex 41. A flat first layer learns it
once per column group, from 54 times less data each.

**Board identity is free capacity to waste.** The tiles block is 361 floats — 20% of the
input — and is *constant for a whole game*. A dense first layer can spend that on
memorising which board it is on rather than on how to play.

Both showed up in behaviour cloning: 87% training agreement against 71% held-out.

## The decision

`training/structured_net.py`. Three ideas, each measured rather than assumed.

**Weight sharing.** One small MLP is applied to all 54 vertex rows, another to all 72 road
rows, another to all 19 tile rows. Every vertex of every game becomes a training example for
the same weights.

**Positional heads.** The logit for "build a settlement at vertex 23" is `Linear(d, 2)` applied
to vertex 23's *own* embedding, not a column of a 512×324 dense layer. Two weights per output
instead of 512, shared across all 54 vertices — so a vertex never built on in training still
gets a sensibly ranked logit.

**Neighbourhood information, bought cheaply.** The encoder tells a vertex its pip potential but
never which *resources* it touches, nor whether the robber sits on one of its tiles. Those are
one hop away in `catan.topology`.

The obvious implementation — gather each entity's neighbours' *embeddings* and pool them — was
built and measured first, and is far too slow: one gather-and-sum of the vertex-from-road
relation at width 64 costs 3.8 ms at batch 512, where the entire flat trunk costs 6.6 ms. It
materialises a `(512, 162, 64)` intermediate to add up 162 numbers. So the aggregation runs on
the **raw** features instead, through constant row-normalised incidence matrices: the same
relation as one dense matmul at width 6–19 costs 0.33 ms. Because the graph is fixed and the
aggregation linear, *k* hops compose into a single precomputed matrix.

## Measured

Same data, same epochs, same seed — 200 demonstration games, 67,351 decisions:

| | flat MLP | structured |
|---|---|---|
| parameters | 1,355,589 | **184,730** |
| held-out agreement | 69.6% | **80.3%** |
| train agreement | 83.5% | 82.4% |
| **train/test gap** | 13.9 pts | **2.2 pts** |
| value MAE | **0.074** | 0.210 |
| forward @ batch 512 | **8.6 ms** | 21.3 ms |

+10.7 points held-out from 7.3× fewer parameters, and the overfitting gap essentially
disappears — which is the diagnosis confirming itself, not just a better score.

Straight out of cloning, before any self-play (200 games each):

| | flat clone | structured clone |
|---|---|---|
| vs heuristic | 30.8% | **39.5%** |
| vs greedy | 92.8% | 92.7% |
| vs random | 94.4% | 95.8% |

## The two honest caveats

**The value head is worse** — MAE 0.210 against 0.074. The pooled trunk is 256 wide against
the flat network's 512, and the value target is a whole-position judgement rather than a
per-element one, so it has less to work with. It is the one place the flat network still wins,
and it matters because PPO is limited by its critic early on.

**It is 2.5× slower per forward pass.** Less bad than it sounds in context: at the rollout's
batch of ~64 it is 0.11 ms/row against a ~0.28 ms environment step, so the network is not the
bottleneck. It costs about 2 seconds per PPO update.

## Not accepted on the author's say-so

This module was drafted by a subagent during a design workflow. Its central claim was
re-measured independently — same script, same data, both networks, from scratch — before it
was kept. It was also committed before that review happened, which was a mistake; the
verification is what makes it defensible, not the fact that it reads well.

## See also

- [0019 — caching the board-static observation](0019-cache-the-board-static-observation.md) —
  the same observation from the other side
- [0018 — cloning](0018-clone-before-self-play.md) — the benchmark used to compare them
- [0020 — parallel rollouts and lookahead](0020-parallel-rollouts-and-lookahead.md)

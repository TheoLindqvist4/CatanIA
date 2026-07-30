# 0018 — Clone the heuristic, then self-play

**Status:** accepted · **Date:** 2026-07-30 · **Phase:** 8

## The problem

From-scratch self-play works and is slow. Measured: 0.5% against the heuristic at iteration 0,
~70 minutes and 2.7M transitions to reach 37%. Watching the games, most of that time is spent
rediscovering things `catan/heuristics.py` already states outright — that settlements belong
on high-pip vertices, that a city beats a road, that four ore for a sheep is a bad trade.

There is no reason to pay for that twice.

## The decision

Train the policy to *imitate* the heuristic first — supervised learning on
`(observation, mask) -> action` — and start PPO from there. `python -m training.clone`.

The value head is trained at the same time on the eventual outcome of the game each position
came from, so the critic starts knowing that a two-city lead is winning. That makes every
advantage in the first PPO iteration less noisy, which matters more than the policy head:
early PPO is limited by the critic, not the actor.

Cloning sets the starting point, not the objective. Once PPO takes over, the only thing being
maximised is winning, and the heuristic's mistakes are free to be unlearned. What it cannot do
is *invent* a strategy the heuristic never demonstrates — which is what the self-play half is
for.

## Noisy play, clean labels

The first attempt got this wrong, and the way it was wrong is worth recording.

A deterministic demonstrator visits one narrow band of positions, and a policy cloned from it
has never seen the states it reaches the first time it plays differently. That is the standard
failure of behaviour cloning, so the games are played by a **noisy** teacher
(`noise = 0.35`) for coverage.

The mistake was then *training on the noisy teacher's actual move*. That teaches the student
to imitate the mistakes along with the judgement, and it caps what is achievable: measured, a
`noise=0.35` teacher agrees with the noiseless heuristic only **71.2%** of the time, so no
learner imitating it could score better than that.

The fix is to keep the noisy rollouts and record what the **noiseless** heuristic would do in
that position. One extra evaluation per decision, and nothing else:

| labels | held-out agreement | vs heuristic |
|---|---|---|
| noisy teacher's move | 57.4% | 11.7% |
| noiseless heuristic's move | **70.9%** | **30.8%** |

This also corrected a conclusion that had been drawn too early. The 57% plateau, next to 80%
training accuracy, looked like evidence that a flat MLP cannot express the heuristic's decision
rule — a real concern, since the tiles section is 27% of the observation and constant within a
game, so a flat first layer can spend its capacity on board identity. Most of the gap turned
out to be the noisy teacher. The architecture concern is still live (the train/test gap is
genuine, and the ceiling is not 100% — the noiseless heuristic self-agrees only 91.7%, because
`_best` breaks ties at random), but it was a smaller effect than the labelling bug in front of
it.

## Measured

300 demonstration games — 86,551 decisions, 47 seconds to generate — then 20 epochs:

| | vs heuristic | vs greedy | vs random |
|---|---|---|---|
| cloned policy | 30.8% | 92.8% | 94.4% |
| *(2.7M steps of from-scratch self-play, for comparison)* | 37.2% | — | — |

About four minutes of work reaches what took seventy minutes of self-play. That is the whole
argument.

## Fine-tuning is not training

PPO resumed from a cloned checkpoint runs at **lr 1e-4 and entropy 5e-3**, not the from-scratch
3e-4 and 1e-2. A competent policy has low entropy by construction (measured 0.68 after cloning
versus 1.99 from random init); the default settings push hard enough to destroy what cloning
learned before the outcome signal can replace it.

## What was rejected

- **Cloning with a deterministic teacher.** Better labels, far worse coverage: the student
  never sees the states its own mistakes lead to.
- **Cloning only the policy head.** The critic is what limits early PPO. It is also the
  cheaper thing to learn — value MAE reached 0.08 while agreement was still climbing.
- **Skipping cloning and buying more compute.** It is not that self-play cannot get there;
  it is that it spends its first two million steps learning what is already written down.

## See also

- [0017 — PPO self-play](0017-ppo-self-play.md)
- [0016 — the heuristic being cloned](0016-heuristic-opponent-and-difficulty.md)

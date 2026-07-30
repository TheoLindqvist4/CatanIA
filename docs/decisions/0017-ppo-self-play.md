# 0017 — PPO self-play, not AlphaZero

**Status:** accepted · **Date:** 2026-07-30 · **Phase:** 8

## Why not the chess answer

The obvious model is AlphaZero: self-play, MCTS, a policy-value network, superhuman in a day.
It does not transfer, for three reasons that are properties of Catan rather than of effort.

**Search needs a state you can roll forward.** `CatanEnv.clone` copies `dice_deck`,
`dev_deck` and opponents' `dev_cards` *verbatim* — correct, because these are hidden rather
than random, but it means every rollout replays the same future instead of sampling one. To
search honestly you would first have to reshuffle the unseen parts consistently with what has
been observed, which is belief sampling, which is a project. This is documented on `clone`
itself and in [0014](0014-ai-surface.md).

**The dice.** Even with perfect information, identical policies in identical positions win or
lose on a 7. The signal is buried in variance, so telling a good move from a lucky one takes
far more games than chess.

**Throughput.** Measured 3,327 env steps/sec on one worker and 22,377 across 16 of 20 cores.
That is ample for PPO, which needs many cheap steps, and nowhere near enough for MCTS, which
needs many expensive ones.

So: PPO on the observation the agent actually gets, which already hides what it should.

## Three engine invariants this depends on

None of these is a documented guarantee. All three are load-bearing, and each fails
*silently* — a run trains happily and learns nothing. Measured, then pinned in
`tests/test_training.py`.

**The winner is always the player who just acted** (21/21 decided games). `env.py:117`
captures the actor before applying, and only a player's own action can complete their victory
condition. So `step()`'s reward is *always* `+1` and `LOSS_REWARD` is unreachable on the
normal path. A learner that consumed the returned reward would see a constant `+1`, its critic
would converge to `V ≡ 1`, every advantage would collapse to zero, and `explained_variance`
would sit at 0 forever with nothing in the logs to say why.

→ The rollout discards the returned reward, reads `info["winner"]`, and writes `±1` onto the
last stored transition of **both** seats.

**The terminal observation belongs to the winner** (21/21). `state.current_player` returns
`self.winner` once the phase is `GAME_OVER`, and the observation is built from
`current_player`. Bootstrapping the loser's trajectory from it is an exact sign flip on half
the data. Nothing is bootstrapped at a terminal here — both returns are known to be exactly
`±1`.

**The terminal mask is all-zero** (21/21). A masked softmax over nothing. The collector resets
an environment the moment its game ends, so a dead position never reaches the policy head.

## The rest of the design

**Per-seat trajectories, GAE over a seat's own decisions.** The game is not alternating —
during a discard the decision belongs to whoever is over the hand limit — so `info["player"]`
is the authority. The timeline for GAE is a seat's *consecutive decisions*: between two of its
moves the opponent has acted, dice have rolled and cards have been drawn, all of which are
environment dynamics from that seat's point of view. Discounting a player's own future by the
opponent's activity is not a quantity anyone wants.

**γ = 1.0.** The game pays out once, at the end. Discounting a terminal reward only biases it
toward whoever moved last.

**Forced decisions are skipped.** 12.3% of all decisions have exactly one legal action
(measured over 40 heuristic games). The PPO ratio is 1 by construction, so they carry no
gradient; they are played without querying the network and never enter the buffer. With
γ = 1.0 skipping them is exact rather than approximate — one of the reasons for that default.
A forced action can still *end* the game, so its outcome is folded onto the seat's previous
stored transition.

**Potential-based shaping.** Φ(s) is the normalised victory-point lead; the reward added is
`Φ(s') − Φ(s)`, which telescopes over an episode and so **cannot change which policy is
optimal** (Ng, Harada & Russell 1999). It only densifies credit across ~150 decisions per
seat. Per-step rewards for building things would silently rewrite the objective instead.
Public points only — a shaping term that read victory-point development cards would leak them.

**Truncations are adjudicated, not discarded.** Scored on the victory-point difference at
weight 0.5, the way an adjourned game is adjudicated on material. Early in training almost
every game truncates, so discarding them would throw away most of the data and impose
survivorship bias on exactly the quantity being learned. The 0.5 weight keeps stalling from
ever beating winning.

**The opponent is drawn per game, not per iteration** — 60% the live policy, 15% the
heuristic, the rest frozen past selves sampled toward the recent. Pure self-play improves
against its *present* self and forgets what it used to do; the frozen pool is the pressure not
to, and the heuristic is an external anchor. Self-play win rate is 50% whether the agent is
improving or cycling, so the only honest number is against something that does not move.

**The collector outlives an iteration.** Rebuilding it per iteration discards every in-flight
game: measured 7.5x the necessary work at 128 environments, which made *more* environments
slower rather than faster (420 → 167 → 76 transitions/sec at 16/48/128). With persistent
environments, iteration time went from 88s to ~10s.

## Measured

**From scratch**, 8,192 transitions/iteration, 200-game evaluations against
`HeuristicAgent(noise=0)`: 0.5% → 11.0% → 19.5% → 29.4% → 31.4% → **37.2%** over 2.7M
transitions (~70 minutes). Real, steady learning that plateaus below the heuristic.

**Cloned first, then fine-tuned** ([0018](0018-clone-before-self-play.md)) — 3.33M
transitions, 68 minutes, `lr 1e-4`, `entropy 5e-3`:

| iteration | vs heuristic |
|---|---|
| 0 (the clone) | 27.8% |
| 19 | 21.9% |
| 39 | 31.6% |
| 79 | 38.9% |
| 99 | 45.0% |
| 179 | 49.0% |
| 199 | 54.4% |
| 299 | 55.8% |

The final policy, on **1,000 games** (±3.1 points — 200 games gives ±7, which cannot tell
parity from an edge):

| opponent | result | rate | 95% CI |
|---|---|---|---|
| heuristic (`hard`) | 507 – 467 | **52.1%** | [48.9, 55.2] |
| heuristic (`medium`) | 299 – 97 | 75.5% | [71.0, 79.5] |
| heuristic (`easy`) | 341 – 57 | 85.7% | [81.9, 88.8] |
| greedy | 399 – 1 | 99.8% | [98.6, 100.0] |
| random | 399 – 1 | 99.8% | [98.6, 100.0] |

**Read this honestly: the interval includes 50%, so it is parity, not a win.** The 55.8%
seen during training was noise on top of ~52%. Against everything *below* the heuristic it is
clearly stronger — 99.8% versus greedy where the heuristic itself scores 96.7% — but it has
not demonstrated an edge over the thing it was cloned from.

Diagnostics stayed healthy throughout: explained variance 0.62–0.79, KL 0.014–0.028 against a
0.03 guard, and truncations falling from 20 per 200 games to 1 — it learned to *finish*, which
is most of what separates 30% from 50%.

**The first hundred iterations looked like a failure.** 27.8% → 21.9% → 31.6% → 31.1% reads
as PPO destroying what cloning learned, and the temptation to intervene was real. The
confidence intervals overlapped the whole time. Fine-tuning needed ~100 iterations to work
through the disruption before it passed the from-scratch run; stopping early would have thrown
away the entire result.

## What it does not learn

**The observation has no history.** It is a pure snapshot: phase, last roll, turn number,
bank, deck size. Nothing records what the opponent discarded, bought or built over time. A
person tracks "they have hoarded ore for three turns, cities are coming"; this agent cannot
represent the thought. That is the hardest ceiling in the current design, and it is an
observation problem, not a training one.

**No lookahead.** A pure policy with zero search.

**Board geometry is rediscovered from correlations.** The network is an MLP over a flat
1808-vector; that vertex 23 neighbours vertex 24 is never given to it, though `topology.py`
knows. A graph network over the vertex/road structure is the obvious next architecture.

## See also

- [0018 — cloning the heuristic first](0018-clone-before-self-play.md)
- [0014 — the AI surface](0014-ai-surface.md) · [0016 — the heuristic](0016-heuristic-opponent-and-difficulty.md)

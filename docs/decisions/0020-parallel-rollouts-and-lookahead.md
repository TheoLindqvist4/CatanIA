# 0020 — Parallel rollouts, and a lookahead that does not cheat

**Status:** accepted · **Date:** 2026-07-30 · **Phase:** 8

Two of the four things [0017](0017-ppo-self-play.md) listed as being in the way. One worked,
one did not, and both are recorded because a negative result that is not written down gets
re-attempted.

## Parallel rollouts — 12.6x

The engine is pure Python, so a rollout is bound by the GIL and uses one core of twenty.

| workers | transitions/sec |
|---|---|
| 1 | 1,879 |
| 2 | 3,731 |
| 4 | 5,411 |
| 8 | 9,525 |
| 12 | 17,289 |
| 16 | **23,599** |

Steady state — 8,192 transitions per call, warm-up calls discarded. The warm-up matters: with
persistent collectors, transitions are banked in bursts when games finish, so a short
measurement measures luck. The first version of this benchmark reported 4 workers as *faster
than* 8 for exactly that reason.

Three things made it less mechanical than it looks.

**Windows spawns rather than forks.** Children re-import the module, so the worker function is
module-level and nothing is inherited — each worker builds its own environments and its own
copy of the network. A script run from a heredoc cannot be a parent at all, because the child
cannot re-import `__main__`.

**Thread oversubscription is the usual way to make this slower.** Torch takes one OpenMP
thread per core by default — measured `get_num_threads() == 14` here — so eight workers would
request 112 threads on 20 cores. Workers are pinned to one thread each. The variables have to
be set *before* the child imports torch; since a child inherits `os.environ` at spawn, they
are set in the parent immediately before the pool is created. The parent has already
initialised its own pool by then, so it keeps the wide setting for the PPO update, which is a
genuinely parallel matmul.

**Each worker keeps its own opponent pool.** Broadcasting it would cost 54 MB per worker per
iteration at ten frozen networks. Instead each worker snapshots the weights it receives on the
same fixed schedule; since they all receive the same weights, the pools are identical in
content without a byte crossing the boundary. Only the learner's 5.4 MB is sent.

## Lookahead — leak-safe, and no measurable gain

`LookaheadAgent` scores each candidate by the position it leads to:
`log π(a) + weight · (V(s') − V(s))`.

**The interesting part is what it refuses to search.** `GameState.clone` copies the dev deck,
the dice deck and opponents' hands *verbatim* — correct for a point-in-time copy, and exactly
why a naive lookahead cheats:

| action | what applying it to a clone reveals |
|---|---|
| `BUY_DEV_CARD` | the real next card off the deck |
| `MOVE_ROBBER`, `PLAY_KNIGHT` | the real steal — a card from the victim's hand |
| `END_TURN` | the real next roll from the 36-card balanced deck |
| `PLAY_MONOPOLY` | what opponents actually hold |

So `DETERMINISTIC_TYPES` is a **leak boundary**, not an optimisation, and a test asserts none
of those five is in it. A second test replays a whole game and demands the identical move at
every decision once the opponent's hidden cards are rewritten at constant public counts —
the same scramble the heuristic's leak test uses, now shared in `tests/helpers.py` so the two
cannot drift apart.

What remains searchable is where to build and what to trade, which is most of what positional
judgement is *for*.

### It does not help

300 games each against `HeuristicAgent(noise=0)`:

| weight | rate | 95% CI |
|---|---|---|
| 0.0 (no search) | 53.7% | [48.0, 59.4] |
| 0.5 | 51.4% | [45.7, 57.0] |
| 1.0 | 56.9% | [51.2, 62.5] |
| 2.0 | 53.1% | [47.4, 58.7] |

Every interval overlaps every other. The best-looking number, 56.9%, is not distinguishable
from doing nothing — so it was re-run at **800 games** (±3.5) to be fair to it:

| weight | rate | 95% CI |
|---|---|---|
| 0.0 | 52.2% | [48.7, 55.7] |
| 1.0 | 53.4% | [49.9, 56.9] |

1.2 points apart, with intervals overlapping almost entirely. That is a settled negative
result rather than an underpowered one.

The likely reason is the **shared trunk**: the policy and value heads read the same features,
trained together, so `V(s')` mostly re-states what `π` already encodes. A lookahead is worth
something when the critic knows something the actor does not, and here it does not.

Kept, off by default, because the plumbing is what a genuinely better critic would need and it
costs nothing at `weight=0`. Deeper search is a different matter entirely — it needs the
opponent's reply, which needs their hand, which is hidden. That is belief sampling.

## See also

- [0017 — PPO self-play](0017-ppo-self-play.md) — where these were listed as next steps
- [0021 — the structured network](0021-structure-aware-network.md)

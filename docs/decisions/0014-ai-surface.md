# 0014 — The AI surface: action space, observations, environment

**Status:** accepted · Phase 3

## Context

Everything before this phase served one goal: give an agent a complete, machine-readable view
of a Catan game. This record covers the four modules that expose it, and the decisions inside
them that are not obvious.

    catan/action_space.py   324 flat indices, plus legal_mask(state)
    catan/encoder.py        1808-float observation, perspective-rotated, hidden-info masked
    catan/env.py            reset(seed) / step(index)
    catan/agents.py         random and greedy baselines, plus a match harness

## The action space

**324 indices, in contiguous blocks by action type.** A slice of the mask is therefore a
slice of one action kind, which helps debugging and allows per-type policy heads.

**The size does not depend on the player count.** Robber actions naming a player who is not in
the game simply never come back legal. A network trained on 1v1 has the same output shape as
one trained on four players, so weights transfer and evaluation code does not branch.

**The mask is translated from `rules.legal_actions`, never re-derived.** This is the single
most important line in the module. Re-deriving legality here would recreate exactly the bug
the engine was rebuilt to remove — two authorities that can disagree. Translation costs +2%.

**`encode` raises on an inexpressible action** rather than returning a sentinel. A silent drop
would make that action permanently unreachable for an agent, and the failure would look like a
policy that never learns a particular move. The load-bearing test plays whole games under both
rulesets at 2 and 4 players, encoding everything the rules offer, so adding an `ActionType`
without extending the space fails loudly.

**Year of Plenty pairs are sorted.** Taking ore-then-wheat is the same move as the reverse, so
allowing both would put two indices on one outcome. A test caught `apply` accepting a
non-canonical form that `legal_actions` never offered.

## The observation

**1808 floats, fixed length**, in named blocks (`LAYOUT`) with shapes (`SHAPES`) so a consumer
can reshape the per-tile / per-vertex / per-road runs for a graph or convolutional model rather
than being forced through an MLP.

**Perspective rotation.** `encode(state, me)` puts *me* in player slot 0 and the others in turn
order after. One network plays every seat, and a position encodes identically whichever player
number holds it — `test_the_same_position_encodes_identically_whichever_number_holds_it` builds
the same position twice under different numbers and compares.

**Hidden information is masked per observer.** An observation never contains:

| hidden | what is public instead |
|---|---|
| another player's hand *composition* | its size — cards are countable |
| another player's dev-card *composition* | how many they hold, and Knights played |
| the dev deck's order or contents | how many are left |
| the Balanced Dice deck | nothing at all |

This is enforced by **leak detectors**: mutate the hidden thing and assert the observation does
not move. Swapping an opponent's three Knights for three Victory Points must change nothing;
reshuffling the dev deck must change nothing; replacing the dice deck must change nothing.

`public_victory_points` exists for this reason and is used in three places — the observation, the
`info` dict, and Friendly Robber's threshold.

**A few derived features are included on purpose.** Pip potential per vertex (summed production
odds of its adjacent tiles), buildability flags, and trade rates are all derivable from other
parts of the vector, but not by a plain MLP without learning the incidence structure first.
They are cheap and they are what a human reads off the board.

**Everything is scaled into `[0, 1]`**, using exact maxima where one exists — a resource count
cannot exceed the bank's 19 — and a documented soft cap otherwise. A test asserts every value
is a `float` in range at every point of real games; that caught `sum()` of an empty generator
returning int `0` for a vertex touching only the desert.

## The environment

**The dice are rolled for you.** `roll_dice` is stochasticity, not a move, so `step` rolls
whenever nobody has a choice — in a loop, since one roll can lead straight to the next player's
roll. An agent never sees a state whose only option is "roll". *Except*: a Knight played before
the roll is a real choice — it decides which tile pays out this turn — so if one is available the
environment stops and offers it.

**Whoever must act is the observer.** Catan is not strictly alternating — during a discard the
decision belongs to whoever is over the hand limit, usually an opponent. So `info["player"]`
reports who is being asked, and the observation is built from *their* view. A self-play loop
must read it rather than assume turn order; assuming is the classic multi-agent environment bug.

**An illegal index raises.** Substituting a legal move would teach an agent that its choice does
not matter, which is worse than a crash.

**Reward is terminal and zero-sum**: `+1` winner, `-1` otherwise, `0` during. Attributed to the
player who *acted*. Shaping is a training decision, not an environment one, so it is left to the
caller.

**Truncation is separate from termination.** A game stopped at `max_turns` reports
`truncated=True`, no winner, and reward `0`. A learner that reads a time-out as a loss is
learning from noise.

## Performance

Memoising longest road was the single biggest win. It is an exponential search, and both
`update_awards` (after *every* build) and the encoder (per player) want it.

The memo is keyed on **the ownership arrays themselves**, not invalidated by hand. Hand
invalidation would be a rule every future mutation site had to remember — including test helpers
that write straight into the arrays — and a missed one is a silently wrong award. A derived key
simply misses instead. Building and hashing it costs a few microseconds against tens per search.

`legal_actions` also gained cheap affordability gates: scanning 72 roads to discover the player
cannot afford one is the common case, since an empty hand is far more frequent than a full one.

| | before | after |
|---|---|---|
| `update_awards` | 83 µs | **1 µs** |
| `encoder.encode` | 458 µs | **250 µs** |
| `legal_mask`, poor hand | 245 µs | **16 µs** |
| `legal_mask`, rich hand | 245 µs | 236 µs |
| **per env step**, typical | ~700 µs | **~270 µs** (~3,700/s) |

`encode` is now the floor. Further gains want numpy or incremental updates, which is worth doing
only once a training loop shows it matters.

## The baselines

`RandomAgent` is the floor — anything that cannot beat it is broken. `GreedyAgent` takes the
highest-priority *action type* available, with no idea *where* to build, and still wins about
70% against random over 40 games.

`play_match` **swaps seats every other game**. Catan's first-player advantage is real and large,
so a fixed-seat match measures the seat as much as the agent. A mirror-match test asserts
identical agents split near evenly, which is a check on the harness rather than on the agents.

## Known limitation: sampling hidden state

`clone(rng=state.rng)` makes rollouts diverge for plain dice, but **three pieces of hidden state
are copied verbatim** and so replay identically:

* `dice_deck` (Balanced Dice)
* `dev_deck`
* opponents' `dev_cards`

For search to sample genuine futures it must reshuffle the unseen parts itself. That is
belief-sampling and it is **not implemented** — it belongs with whatever search algorithm is
built on top, since the right approach depends on the algorithm. Recorded here so it is not
discovered as a surprise:
`test_with_balanced_dice_a_clone_replays_the_same_rolls_even_sharing_the_rng` pins the behaviour.

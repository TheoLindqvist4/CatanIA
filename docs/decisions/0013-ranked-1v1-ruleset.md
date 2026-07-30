# 0013 — Ranked 1v1 is the target ruleset

**Status:** accepted · Phase 2 (revised)

## Context

The engine was built to the printed base-game rules. But the training target is **Colonist.io
ranked 1v1**, which is not base Catan with two players — it changes four things that alter how
the game is actually played.

Published settings for ranked 1v1:

> there are only **2 players**, **Friendly Robber is On**, **Dice are Balanced**, game-speed is
> set to **Very Fast**, win condition is set at **15 Victory Points**, and you can safely hold
> up to **9 cards** in your hand without 7ing out

("Very Fast" is a move timer, so it has no engine meaning.)

## Decision

Make the differences **configuration**, not special cases threaded through the rules.
`catan.rulesets.RuleSet` is a frozen NamedTuple; `GameState(ruleset=...)` takes one and
defaults to `RANKED_1V1`.

| | `BASE_GAME` | `RANKED_1V1` |
|---|---|---|
| victory points to win | 10 | **15** |
| hand limit before discarding | 7 | **9** |
| Friendly Robber | off | **on**, threshold 2 |
| Balanced Dice | off | **on** |

Keeping the base game runnable is not sentimentality — it is how you tell a *variant* from a
*bug*. Every ranked-1v1 behaviour has a base-game counterpart test, so a change that breaks
both is a bug and a change that breaks one is the variant working.

Player-to-player trading stays out, per
[decision 0011](0011-no-player-to-player-trading.md) — and the guide agrees that 1v1 "lacks
social interaction and trading".

## Friendly Robber

> You can only be blocked and stolen from after openly having more than 2 Points (Settlement,
> City, Longest Road, or Largest Army; Victory Point cards don't count).

Two consequences, both implemented:

* a protected player **cannot be robbed** — `victims_at` excludes them
* a protected player's tiles **cannot even be blocked** — `robber_destinations` excludes any
  tile where a protected *opponent* has a building

"Openly" is why `public_victory_points` exists: hidden Victory Point cards do not count toward
the threshold, so a player sitting on VP cards keeps their protection. Awards do count.

Your own buildings never protect a tile from you — blocking yourself is legal, if silly.

`robber_destinations` falls back to ignoring the restriction if it would leave nowhere to go.
That is unreachable — a protected player has at most 2 public points, so at most 2 buildings,
so at most 6 of the 18 candidate tiles — but an empty list would deadlock the game rather than
fail loudly, so it is guarded. `test_the_robber_always_has_somewhere_to_go` asserts at least 12
destinations remain.

## Balanced Dice

> Dice Deck is a card deck with all 36 combinations found in a 2 dice system. Instead of
> rolling the dice you draw a card. [...] a single Dice Deck with reshuffling at 12 cards
> remaining and a 30% probability reduction of rolling the same number 2 times in a row.

**Implemented:** the 36-card deck, drawing instead of rolling, and replacement once 12 cards
remain. That is the part that tightens the distribution, and it is verifiable — no combination
repeats within a deck, and the long-run distribution matches the triangular one.

**Deliberately not implemented**, because the published description is not precise enough to
reproduce and inventing numbers would be worse than a stated gap:

* the **30% same-number-twice-in-a-row reduction**. The article gives the *effect* (doubles per
  game fall from 5.43 to 3.75) but not the weighting, and links to code that is no longer
  reachable.
* the **7-ownership balancing** described separately, which nudges 7 probability toward an even
  split between the two players over a game.

`test_the_undocumented_parts_of_balanced_dice_are_not_faked` asserts the repeat rate is at its
natural level, so the gap is recorded in the suite rather than only in prose.

The article does not say what becomes of the 12 undrawn cards at a reshuffle. Replacing the
deck with a fresh 36 is the reading taken — it is what stops the tail of a deck being
deducible, which is presumably why the reshuffle exists at all.

### ⚠️ A consequence for search

The dice deck is copied by `clone()`, so **a clone replays the same rolls even when it shares
the RNG**. That is correct — the deck is hidden information, already determined, not a fresh
die roll. But it means search cannot get divergent rollouts from the RNG alone: to sample
futures it must reshuffle the unseen remainder of `dice_deck` itself.

That is belief-sampling over hidden state, and it now applies to three things: `dice_deck`,
`dev_deck`, and opponents' `dev_cards`. Phase 3 has to handle all three together.
`test_with_balanced_dice_a_clone_replays_the_same_rolls_even_sharing_the_rng` pins the
behaviour.

## Consequences

**Good**

- The engine models the format it will be trained and evaluated on.
- One ruleset object rather than scattered conditionals, so a third variant is cheap.
- The base game remains a control.

**Measured** — 30 random games each, to a 40,000-action cap:

| | finished | median turns | best VP reached |
|---|---|---|---|
| base game (10) | 30/30 | 326 | 10 |
| ranked 1v1 (15) | 30/30 | 435 | 15 |

15 points is comfortably reachable: 4 cities + 5 settlements is 13 on their own, and both
awards add 4.

**Cost**

- Longer games mean more steps per episode, so training throughput matters more. Phase 3's
  performance work is now more valuable, not less.
- Two rulesets is two things to keep tested. Mitigated by expressing tests in terms of
  `state.ruleset.*` wherever the mechanism, not the number, is the point.

## Sources

- [Ranked 1v1 — a comprehensive strategy guide (Colonist.io)](https://blog.colonist.io/ranked-1v1-comprehensive-strategy-guide-colonist-io/)
- [Colonist rules — base game](https://colonist.io/catan-rules)
- [Designing balanced dice](https://blog.colonist.io/designing-balanced-dice/)
- [Balancing 7s in 1v1](https://blog.colonist.io/balancing-7s-on-1v1/)
- [1v1 friendly robber variation](https://colonist.featureupvote.com/suggestions/511431/1v1-friendly-robber-variation)

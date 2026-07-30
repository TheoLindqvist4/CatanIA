# 0011 — No player-to-player trading in this version

**Status:** accepted · deferred to a future version

## Context

Catan has two kinds of trade: with the bank (4:1, or 3:1 / 2:1 at a harbour) and directly
with another player, on any terms both accept.

Bank trading is implemented, and it is what made games finishable — 4 of 40 random games
reached 10 points before it, 39 of 40 after. Player-to-player trading is a separate feature,
and it is the hardest thing in Catan to expose to an agent.

**Why it is hard.** An unrestricted offer is a *multiset for a multiset*: give any
combination of your cards for any combination of theirs. That does not flatten into a
discrete action space. Bounding it (give *n* of X for *m* of Y, small *n* and *m*) still costs
5×5×n×m offer actions plus accept/reject, and it makes every turn a negotiation sub-game with
its own state — who has offered what to whom, and whose turn it is to respond.

**Why it matters less here.** The immediate target is **1v1**. In two-player Catan, trading
with your only opponent hands resources to the single person who can beat you, so strong play
rarely does it. The feature carries most of its strategic weight at 3–4 players, where you can
trade with whoever is not currently leading.

## Decision

**Leave player-to-player trading out.** Bank and harbour trading stay; direct offers do not
exist, and no action type is reserved for them.

The engine still supports 2–4 players, and that support is tested. Only the 1v1 case is being
targeted for training, which is what makes the omission acceptable rather than merely
convenient.

## Consequences

**Good**

- The action space stays flat and small: every action is `(type, position, extra)` with fixed
  arity, so Phase 3's discrete codec is a lookup rather than a parse.
- No negotiation sub-game, so no extra phases, no pending-offer state, and no need to decide
  how an agent values a counter-offer.
- Phase 2 finishes sooner, and the remaining items (robber, dev cards, awards) are all
  unambiguous.

**Bad**

- **Not complete Catan.** An agent trained here has never negotiated, so it would be weak at
  3–4 players against opponents who trade, and could not play a human game where trading is
  expected.
- Resource conversion is only ever at 4:1 / 3:1 / 2:1 with the bank, so the economy is
  tighter than the real game and positions that a trade would rescue stay stuck.
- Whoever adds it later inherits the action-space design problem, and the encoder will need
  opponent-hand *estimates* to value offers — which interacts with hidden-information
  masking.

## When to revisit

Adding 3–4 player training, or evaluating against humans. At that point it needs its own
decision record covering: the bounded offer form, whether counter-offers exist, how a pending
offer sits in `GameState`, and how the action space grows. It should not be bolted on.

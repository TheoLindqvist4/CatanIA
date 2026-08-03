# 0022 — The agent can see how far its hand is from a purchase

**Status:** accepted · **Date:** 2026-08-04 · **Phase:** 8

## The problem

Nothing in the observation said what anything cost. The agent learned affordability from the
legality mask alone: "build a settlement here" was either offered or it was not.

A binary mask cannot represent **"I am one brick short of a settlement"** — and that is the
most common mid-turn judgement in the game, the one that decides whether to trade 4:1 at the
bank, hold, or spend the cards elsewhere. One card short and three cards short are the same
observation today: an action that is not legal.

`CLAUDE.md` had called this the largest known gap for some time.

## Encoding the cost table would have been worthless

This is the part worth writing down, because it is the obvious fix and it does nothing.

The four costs are **identical in every state a network will ever see**. A constant input
carries no information: the first layer folds it into a bias in one gradient step and nothing
downstream can condition on it, because it never varies to condition on. Adding
`ROAD_COST` to the observation would have grown `encoder.SIZE`, invalidated every checkpoint,
and changed no decision.

What varies — and what a turn actually turns on — is the **difference** between the hand and
the price, and, since a deficit gets closed at the bank, that difference **priced through my
own harbours**. One brick short holding six wood is:

| | cards to give |
|---|---|
| 2:1 wood harbour | 2 |
| 3:1 generic harbour | 3 |
| no harbour | 4 |
| holding only three wood | out of reach |

Four situations the old observation could not distinguish from each other, and could not
distinguish from being three cards short either.

## What was added

A new `affordability` block: `len(PURCHASES)` rows × `AFFORDABILITY_FEATURES` columns, 16
floats, sitting between `players` and `history`. `encoder.SIZE` 1868 → 1884.

Per purchase — road, settlement, city, development card:

| column | meaning |
|---|---|
| `affordable` | my hand covers it now (a Road Building credit counts as paid) |
| `cards_short` | cards missing, over what this purchase costs — so the rows are commensurable |
| `trade_price` | cheapest cards handed to the bank to close the gap; 1.0 means out of reach |
| `coverable_now` | the gap *can* be closed, by trades the bank can actually honour |

Two of the four are conjunctions the first layer cannot form, which is why they are worth
their floats rather than being left implicit. A per-resource shortfall is a hinge on one input
coordinate at an integer threshold, and the first `Linear` gets those free from the hand
composition. But *"every one of four resources is covered"* is not a threshold on any sum —
four wood passes every sum test and affords nothing — and neither is
`sum(surplus[g] // rates[g]) >= need`, which divides by a rate that moves with harbour
ownership.

The other two are the graded distances underneath, and they are aimed at the **critic**.
Masking is applied to the policy logits only; `forward` returns an unmasked value. So the
value head has no access to legality at all and was inferring progress from five raw card
counts.

### Why four columns and not twenty

A per-resource shortfall grid — 4 purchases × 5 resources — was designed and rejected. Nine
of its twenty cells are structurally zero in *every* state, because no purchase costs all five
resources, so nine floats would have been exactly the constant input this record opens by
ruling out. The eleven live cells are the single-input hinges the first layer already computes.

### The arithmetic, and three ways to get it wrong

Each is a silent bug rather than a crash:

1. **Surplus is floored per resource, never pooled.** A bank trade is paid with `rate` cards of
   *one* resource held at once, so three wood and three sheep at 4:1 fund nothing at all.
2. **Surplus is measured above the cost**, not from the raw hand. This protects the cards the
   purchase itself needs — and because surplus in a resource implies no shortfall in it, the
   engine's `give != take` rule becomes impossible to violate rather than separately enforced.
3. **Cheapest rate first**, which is the true minimum bill because every trade yields exactly
   one card whatever it cost. `test_the_cheapest_plan_is_the_cheapest_one` proves this against
   an exhaustive search over trade-count vectors rather than taking the argument on trust.

The bank term is `bank[r] >= deficit[r]`, not `bank[r] >= 1`: each trade takes one card, and
paying refills the pile you gave from, never the one you took. One ore in the bank cannot
supply three.

### Affordability, not legality

An empty development deck, a spent city piece and a vertex with nowhere legal to build **do
not move these rows**. Those facts are already encoded — deck remainder in `global`, piece
counts in `players`, placement in the per-vertex and per-road buildability flags — and folding
any of them in would make one float mean three things, so a zero would have three
indistinguishable causes. `_encode_roads` already reports reachability "cost aside" for the
same reason; this block is the mirror, and the two compose.

`test_affordability_is_not_legality` pins it, so that a later well-meaning fix does not
"correct" the divergence.

## The scaling deviates from the stated contract, on evidence

`encoder`'s docstring says to use an exact maximum where one exists. For `trade_price` one
does: five trades at `BANK_RATE` is 20 cards, and synthetic hands reach it.

It was still rejected, measured rather than asserted. Over 63,372 `(player, purchase)` samples
from random games, coverable gaps priced at `{0: 3545, 2: 175, 3: 524, 4: 991, 5: 17, 6: 20,
8: 26, 12: 1}` — 0.02% above 8, and none anywhere near 20. Dividing by 20 would leave every
value that actually occurs inside the bottom fifth of the range, with an empty gap up to the
out-of-reach sentinel. So `TRADE_PRICE_SCALE = 2 * BANK_RATE`, clipped, and out of reach
saturates at the same 1.0 — which keeps the column monotone "cards this would cost me"
throughout, with the collision at the top separated by `coverable_now` in the same row.

⚠️ That measurement came from random play. A trained policy that hoards deliberately could
push mass into the clip; re-measure the histogram on champion rollouts before assuming it
holds.

## Consequences

**Every checkpoint was invalidated**, which is the price of the change. The fallback is
graceful and was verified rather than assumed: `models/champion.pt` fails to build,
`champion.load()` returns `None`, and the web game falls back to the noiseless heuristic.

**The opponent dropdown was a hardcoded list in `index.html`**, so it went on offering
"Learned (strongest)" after the champion stopped loading, and picking it answered with an HTTP
400. The list is now served from `/api/geometry` and built by the client — the same rule the
rest of this interface already follows.

**The first promotion after this change is ungated by construction.** With no loadable
reigning champion, `champion.promote` takes its `reigning is None` branch and installs
immediately: no Wilson lower bound, no regression check. So the gating for that one promotion
has to be done by hand, against `HeuristicAgent(noise=0)` over enough games for the interval
to mean something.

**Nothing in the network needed changing.** The block sits at `LAYOUT["players"].stop`, and
`structured_net.CONTEXT_START` is `LAYOUT["players"].start`, so `CONTEXT_FEATURES` grew on its
own and the 16 floats reach both the trunk and — through `context_bias`, added before the
first ReLU — all 145 per-position embeddings.

That last part was luck, and a guard was added for it: a block placed anywhere *before* the
players block would have fallen into a gap that no slice covers and been silently read by
nothing at all. `structured_net._validate` now asserts `ROAD_SPAN.stop == CONTEXT_START`, and
`encoder._validate` asserts `tuple(LAYOUT) == order` so the next block cannot be added to the
layout and forgotten in the check.

## Alternatives considered

- **The cost table as features.** Constant, therefore free of information. The reason this
  record exists.
- **A per-resource shortfall grid (20 floats).** Nine structurally-zero cells; the live ones
  are hinges the first layer already forms.
- **A single scalar "distance to the nearest purchase".** Hides which purchase, and hides
  which resource is missing — which is exactly what decides what to trade for.
- **Folding in the hand limit.** A plan costing 12 surplus cards is a plan a discard can
  halve. It enters as the *rationale* for the soft cap and never as arithmetic, so the block
  means the same thing under both rulesets.
- **Modelling Year of Plenty, Monopoly or a robber steal as ways to close a gap.** Monopoly's
  yield depends on hidden hands: computing it from mine is fiction, from theirs a leak. My own
  development cards are already visible to me, so the network can condition on the two
  channels separately.

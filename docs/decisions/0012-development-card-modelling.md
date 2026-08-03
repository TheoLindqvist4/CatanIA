# 0012 — How development cards are modelled

**Status:** accepted · Phase 2

## Context

Development cards are the last of Catan's rules, and several of them do not map cleanly onto
a fixed-arity discrete action. Each choice below was made to keep every action a
`(type, position, extra)` triple, because Phase 3 has to flatten the whole action space into
one discrete index.

## Decisions

### The deck is a shuffled list, drawn from the end

`state.dev_deck` holds the 25 cards in a fixed shuffled order, and buying pops from the end.

The alternative — keeping counts and drawing at random per purchase — would make a clone
replay differently, because each draw would consume the RNG. A shuffled list makes the deck
**hidden information rather than a fresh die roll**, which is what it actually is: the order
is already determined, the players just cannot see it.

⚠️ **Phase 3's encoder must mask `dev_deck`.** A search that can read it knows every future
purchase and is cheating. Same for `dev_cards` of opponents.

### Victory Point cards are never played

They are not an action at all. They count toward `victory_points` while *held*, and
`PLAYABLE` excludes them so `dev_card_actions` never offers one.

`public_victory_points` exists alongside `victory_points` for the same reason: a VP card is
hidden until it wins, so what opponents can see differs from the truth. The encoder needs
both.

Buying a VP card can win the game immediately, so `BUY_DEV_CARD` checks for a winner.

### A Knight may be played before rolling

The rules allow playing a development card before the dice, which matters — a Knight can
block a tile *before* it produces.

So `legal_actions` returns the available card plays during `Phase.ROLL`, and `apply` accepts
only those there. **`roll_dice` is still how you advance**; the pre-roll list is what may be
done first, and is usually empty.

> **Amended.** Originally all four playable cards were offered pre-roll, since the printed
> rules permit it. Only the Knight is now, via `catan.actions.PRE_ROLL_PLAYS` — a deliberate
> narrowing of the printed rule, applied to both `legal_actions` and `apply`.
>
> The Knight is the only one whose value depends on going first: it moves the robber, so it
> changes what this roll pays out. Road Building, Year of Plenty and Monopoly have identical
> effects either side of the dice, so pre-roll they were a choice between two orderings of
> the same turn — noise in the action mask, and in the web interface something that read as a
> bug to a player who knows the game. Nothing is taken away: the card is still playable that
> same turn, just after the roll.
>
> The action space is unchanged, so checkpoints stay valid; only the legality mask narrowed.

This keeps the dice as environment stochasticity rather than an action. The cost is that
"`legal_actions` is empty" no longer means "time to roll" — a driver must branch on
`phase is Phase.ROLL` instead. Existing drivers already did, so nothing broke.

Because a Knight can reach `Phase.MOVE_ROBBER` from either side of the roll,
`state.rolled_this_turn` records whether the roll has happened, and the robber move returns
to `ROLL` or `BUILD` accordingly.

### Road Building grants credit, not forced placements

Playing it sets `state.free_roads = 2`, and `can_build_road` waives the cost while that is
positive. The player then builds roads as normal actions.

The alternative — a phase that demands two road placements immediately — would be marginally
more faithful but needs another phase and has to handle "fewer than two legal spots". Credit
reuses the existing road action and needs nothing new. Unused credit lapses at `END_TURN`.

Slightly permissive: the player may interleave other actions between the two free roads.
Harmless, and it removes an ordering constraint an agent would otherwise have to learn.

`can_play_road_building` probes legality by temporarily incrementing `free_roads` — the card
is what pays, so affordability must be checked with the cost waived. It restores the value in
a `finally`, and a test asserts no trace is left.

### Year of Plenty takes a sorted pair

`Action(PLAY_YEAR_OF_PLENTY, first, second)` requires `first <= second`, and the
`play_year_of_plenty` helper sorts for you.

Taking ore-then-wheat is the same move as wheat-then-ore. Allowing both would put two indices
in the flat action space for one outcome, and — as a test caught — would mean `apply`
accepting something `legal_actions` never offers, breaking the single-authority guarantee.
15 pairs, not 25.

### Two timing rules, two pieces of bookkeeping

* **One card per turn** — `state.dev_card_played_this_turn`, cleared at `END_TURN`.
* **Not the turn you bought it** — `state.dev_cards_new` records this turn's purchases and is
  cleared at `END_TURN`, so `playable_dev_cards` is `dev_cards - dev_cards_new`.

Two arrays rather than one because a player can hold two Knights, play one, and buy another
in the same turn; a single flag per card type could not express that.

## Awards

Largest Army and Longest Road share one function, `_update_award`, because they behave
identically: a minimum to qualify (3 knights / 5 road segments), sole leadership needed to
take it from someone, and the holder keeps it on a tie. If the holder falls behind and the new
best is tied, nobody holds it until someone is clearly ahead.

`update_awards` runs after **every build**, not just after a road — building a settlement or
city can *break* an opponent's road and take Longest Road off them.

## Consequences

Measured over 40 random 2-player games: **40 of 40** reach 10 points, median 349 turns (down
from 393 before dev cards — more ways to score). Largest Army is held in 39 games and Longest
Road in 37, so both are genuinely contested rather than decorative. Winners' points came from
buildings 203, Victory Point cards 92, Largest Army 58, Longest Road 50.

## Enforced by

`tests/test_dev_cards.py` — 53 tests covering the deck, buying, both timing rules, each card's
effect, and both awards including the break-a-road case.

`test_apply_accepts_exactly_what_legal_actions_offers` now enumerates the **entire** action
space at every step rather than a sample, in both directions. That is what caught the Year of
Plenty ordering mismatch.

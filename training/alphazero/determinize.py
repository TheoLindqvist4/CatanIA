"""Turning a game state into one the searching player is *entitled* to reason about.

This is the correctness boundary of the whole package, and it exists because of a fact
recorded in ``CLAUDE.md``: :meth:`catan.state.GameState.clone` copies the development deck,
the dice deck and every opponent's hand **verbatim**. A tree search over a clone is
therefore a search that has read the opponent's cards. Nothing crashes; the agent simply
plays like someone who can see through the table, and the policy it learns cannot be
reproduced at play time from the observation alone — so the network is trained on targets
it has no way to predict.

:func:`determinize` builds a *particle*: one full game state drawn from the set of states
consistent with everything the observer can see. Search that particle and every belief the
tree forms is a belief about a world the observer could actually be in.

**What is public, and how each hidden quantity is resampled**

``resources``
    Cards are conserved — every card of a resource is in the bank or in a hand
    (``tests/test_rules.py`` pins this). So the number of resource *r* held by everyone
    other than me is ``BANK_PER_RESOURCE - bank[r] - my_hand[r]``, and both terms are
    public. That unseen pool is dealt out to the opponents by hand *size*, which is public
    too — you watch cards enter and leave a hand even when you cannot read them.

``development cards``
    Each opponent keeps the *number* of cards they hold and loses the identities. The
    replacement identities are drawn from ``DECK_COUNTS`` minus my own holding minus the
    knights everybody has visibly played. Whatever is left over becomes the remaining deck,
    reshuffled.

``dice deck``
    Under the ranked ruleset the 36-card deck is consumed rather than resampled, so its
    *length* carries information — and the length is public, because the reshuffle rule is
    printed. The contents are not, so a deck of the same length is shuffled fresh.

**Why this is checkable rather than merely plausible.** Every quantity above is read from
``bank``, ``knights_played``, my own hand and holding, and opponents' hand *sizes*. None of
it is read from what the opponents actually hold. So scrambling the hidden state at constant
public counts — ``tests/helpers.py::scramble_hidden_state``, the same scrambler the encoder
and the heuristic are held to — must leave the determinized state's *distribution*
unchanged. ``tests/test_alphazero.py`` asserts exactly that, seeded, as an equality.

The observer's own cards are never touched: determinizing is about the rest of the table.
"""

import random

from catan.dev_cards import DECK_COUNTS, DevCard, NUM_DEV_CARDS
from catan.resources import BANK_PER_RESOURCE, NUM_RESOURCES, total
from catan import dice


def unseen_resources(state, observer):
    """How many of each resource are held by somebody other than ``observer``.

    Read from the bank and from the observer's own hand, both of which they may see.
    Never from an opponent's hand — that is the point.
    """
    return [
        max(0, BANK_PER_RESOURCE - state.bank[r] - state.hands[observer][r])
        for r in range(NUM_RESOURCES)
    ]


def unseen_dev_cards(state, observer):
    """The development cards that are neither in ``observer``'s hand nor visibly spent.

    A played Knight is face up and counted in ``knights_played``. Road Building, Year of
    Plenty and Monopoly are also played openly, but the engine keeps no per-player tally of
    them, so they are *not* subtracted here — the pool is a little larger than a perfect
    card counter's would be. That errs toward the searcher knowing less, which is the safe
    direction, and it costs at most four cards out of twenty-five.
    """
    pool = [DECK_COUNTS[card] for card in DevCard]
    for card in range(NUM_DEV_CARDS):
        pool[card] -= state.dev_cards[observer][card]
    pool[DevCard.KNIGHT] -= sum(state.knights_played)
    return [max(0, count) for count in pool]


def _deal(pool, count, rng):
    """Draw ``count`` items from a multiset of counts, without replacement.

    Returns the drawn counts and mutates ``pool``. If the pool runs dry — which the
    approximations above make possible rather than impossible — the shortfall is filled
    with index 0, so the caller always gets a hand of exactly the size it asked for.
    """
    drawn = [0] * len(pool)
    for _ in range(count):
        remaining = sum(pool)
        if remaining <= 0:
            drawn[0] += 1
            continue
        pick = rng.randrange(remaining)
        for index, held in enumerate(pool):
            if pick < held:
                pool[index] -= 1
                drawn[index] += 1
                break
            pick -= held
    return drawn


def determinize(state, observer, rng=None, inplace=False):
    """A state consistent with what ``observer`` can see, randomised in what they cannot.

    Args:
        state: the true state. Not modified unless ``inplace``.
        observer: the player whose information set this is. Their own hand, holding,
            buildings and every public fact survive untouched.
        rng: the generator to sample the hidden parts with. Also becomes the returned
            state's ``rng``, so rolling dice inside the search draws from this stream.
        inplace: overwrite ``state`` instead of cloning. For a search root that the caller
            already owns a private copy of; saves a clone per search.

    Returns:
        A :class:`~catan.state.GameState`. Public counts — hand sizes, card counts, the
        bank, the board, every score — are exactly those of ``state``.
    """
    rng = random.Random() if rng is None else rng
    world = state if inplace else state.clone(rng=rng)
    world.rng = rng

    # --- resources ---------------------------------------------------------------- #
    pool = unseen_resources(state, observer)
    for player in world.players:
        if player == observer:
            continue
        world.hands[player] = _deal(pool, total(state.hands[player]), rng)

    # --- development cards: opponents first, then whatever is left is the deck ------ #
    dev_pool = unseen_dev_cards(state, observer)
    for player in world.players:
        if player == observer:
            continue
        held = sum(state.dev_cards[player])
        world.dev_cards[player] = _deal(dev_pool, held, rng)
        # Cards bought this turn are a *subset* of the holding and may not be played yet.
        # The count is public (the purchase was watched); which of the resampled cards they
        # are is not, so take them off the front of the new holding.
        world.dev_cards_new[player] = _take_from(
            world.dev_cards[player], sum(state.dev_cards_new[player])
        )

    deck = [card for card in DevCard for _ in range(dev_pool[card])]
    # The deck's length is public — 25 minus everything bought. Reconcile: the pool may be
    # larger than the true deck (see unseen_dev_cards) or smaller (if the approximations
    # bit the other way), and the *length* is the part an agent may legitimately know.
    wanted = len(state.dev_deck)
    while len(deck) < wanted:
        deck.append(DevCard.KNIGHT)
    rng.shuffle(deck)
    world.dev_deck = deck[:wanted]

    # --- the dice deck: length public, order not ----------------------------------- #
    if state.dice_deck is not None:
        fresh = dice.new_deck(rng)
        world.dice_deck = fresh[:len(state.dice_deck)]

    # A determinized world has no story of its own, and events describe transitions.
    world.events = []
    return world


def _take_from(holding, count):
    """Split ``count`` cards off ``holding`` as a separate holding of the same shape.

    Used for "bought this turn". Takes from the lowest index up, which is arbitrary and
    fine: the identities were just resampled, so there is no ordering to respect.
    """
    taken = [0] * len(holding)
    for index in range(len(holding)):
        while taken[index] < holding[index] and sum(taken) < count:
            taken[index] += 1
        if sum(taken) >= count:
            break
    return taken

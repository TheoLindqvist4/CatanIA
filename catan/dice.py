"""Dice: plain 2d6, or Colonist's Balanced Dice.

**Plain** rolls two independent dice. The distribution is triangular in expectation, but a
short game can still see wild streaks.

**Balanced** draws from a deck of all 36 two-dice combinations, so across a deck the
distribution is exact rather than merely expected. Colonist's ranked 1v1 uses this, and its
published description is:

    "Dice Deck is a card deck with all 36 combinations found in a 2 dice system. Instead of
    rolling the dice you draw a card. [...] a single Dice Deck with reshuffling at 12 cards
    remaining and a 30% probability reduction of rolling the same number 2 times in a row."

Two parts of that are **not** implemented, because they are not documented precisely enough
to reproduce and guessing would be worse than a stated gap:

* the **30% same-number-twice-in-a-row reduction**. The article gives the effect (doubles per
  game fall from 5.43 to 3.75) but not the weighting, and points at code that is no longer
  reachable.
* the **7-ownership balancing** described separately, which nudges 7 probability toward an
  even split between players over a game.

What is implemented is the deck and the reshuffle point, which is the part that actually
tightens the distribution. See ``docs/decisions/0013-ranked-1v1-ruleset.md``.

Reshuffling replaces the deck with a fresh shuffled 36 once 12 cards remain, discarding
those 12 — that is what stops the tail of a deck from being deducible. The article does not
say what becomes of drawn cards, so this is the reading taken.
"""

FACES = 6

#: Every ordered two-dice outcome. 36 of them.
COMBINATIONS = tuple((a, b) for a in range(1, FACES + 1) for b in range(1, FACES + 1))

DECK_SIZE = len(COMBINATIONS)

#: Cards left when the deck is replaced.
RESHUFFLE_AT = 12


def new_deck(rng):
    """A shuffled deck of all 36 combinations, drawn from the end."""
    deck = list(COMBINATIONS)
    rng.shuffle(deck)
    return deck


def roll_plain(rng):
    """Two independent dice."""
    return rng.randint(1, FACES), rng.randint(1, FACES)


def draw_balanced(state):
    """Draw the next dice card, reshuffling when the deck runs low.

    Mutates ``state.dice_deck``. Returns the two die faces.
    """
    if state.dice_deck is None:
        state.dice_deck = new_deck(state.rng)

    first, second = state.dice_deck.pop()

    if len(state.dice_deck) <= RESHUFFLE_AT:
        state.dice_deck = new_deck(state.rng)

    return first, second

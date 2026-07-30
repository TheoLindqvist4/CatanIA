"""Rule variants.

The training target is Colonist.io's **ranked 1v1** format, which differs from base-game
Catan in four ways that matter mechanically. Those differences are configuration, not
special cases sprinkled through the rules, so the engine can also run the base game — which
matters for checking that a change is a *variant* and not a bug.

    from catan.rulesets import BASE_GAME, RANKED_1V1
    state = GameState(ruleset=RANKED_1V1)     # the default

See ``docs/decisions/0013-ranked-1v1-ruleset.md``.
"""

from typing import NamedTuple


class RuleSet(NamedTuple):
    """Everything that varies between formats."""

    name: str

    #: Victory points needed to win.
    victory_points_to_win: int = 10

    #: Cards you may hold without discarding when a 7 is rolled. Above this, you lose
    #: half, rounded down.
    hand_limit: int = 7

    #: Friendly Robber: a player at or below :attr:`friendly_robber_threshold` *public*
    #: victory points cannot be robbed, and the robber may not be placed on a tile where
    #: they have a building. Public points exclude hidden Victory Point cards.
    friendly_robber: bool = False
    friendly_robber_threshold: int = 2

    #: Balanced Dice: draw from a 36-card deck of every two-dice combination instead of
    #: rolling, so the distribution is exact over a deck rather than only in expectation.
    balanced_dice: bool = False


#: Standard Catan, as printed.
BASE_GAME = RuleSet(name="base game")

#: Colonist.io ranked 1v1. The format this project targets.
RANKED_1V1 = RuleSet(
    name="ranked 1v1",
    victory_points_to_win=15,
    hand_limit=9,
    friendly_robber=True,
    balanced_dice=True,
)

#: The default for a new game.
DEFAULT = RANKED_1V1

ALL = (BASE_GAME, RANKED_1V1)

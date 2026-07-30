"""A single die.

``dice_value`` used to be a *class* attribute, so its default was shared by every
instance and a fresh die reported a value of 6 before it had ever been rolled. It is
now per-instance and starts as ``None``.
"""

import random


class Dice:
    SIDES = 6

    def __init__(self, rng=None, sides=SIDES):
        """
        Args:
            rng: a ``random.Random`` instance. Injected so rolls are reproducible;
                several dice may share one generator.
            sides: faces on the die.
        """
        self.rng = rng if rng is not None else random.Random()
        self.sides = sides
        self.dice_value = None  # None until first rolled

    def roll_dice(self):
        self.dice_value = self.rng.randint(1, self.sides)
        return self.dice_value

    def __repr__(self):
        return f"Dice(value={self.dice_value})"

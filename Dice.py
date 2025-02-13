import random

class Dice:
    dice_value = 6

    def roll_dice(self):
        self.dice_value = random.randint(1,6)
        return self.dice_value

    def print_value(self):
        return print(self.dice_value)
    

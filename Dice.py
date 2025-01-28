import random

class Dice:
    dice_value = 6

    def roll_dice(self):
        self.dice_value = random.randint(1,6)
        return self.dice_value

    def print_value(self):
        return print(self.dice_value)
    
# dice_1 = Dice()
# dice_2 = Dice()
# dice_1.roll_dice()
# dice_2.roll_dice()
# dice_1.print_value()
# dice_2.print_value()
# total_dice = dice_1.dice_value + dice_2.dice_value
# print(total_dice)
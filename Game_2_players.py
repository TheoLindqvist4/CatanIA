from Board import Board
from Deck import Deck
from Dice import Dice
from Player import Player
import random

class Game_2_players:
    player_order = [1, 2]
    
    def __init__(self):
        # Create player instances
        self.players = {
            1: Player(),
            2: Player(),
        }
        self.board = Board()
        self.randomize_order()
        self.print_order() 
        self.placing_first_settlement()  
        self.placing_second_settlement()
        

        
    def randomize_order(self):
        random.shuffle(self.player_order)


    def place_settlement(self, player_number):
        position = self.get_user_number()
        player = self.players[player_number]
        if player.player_settlement > 0:
            if position in self.board.settlement_positions:
                player.player_settlement -= 1
                self.board.delete_settlement_position(position)
                player.player_numbers.add(position)
                return f"Settlement placed at position {position}."
            else:
                return "You cannot put a settlement in this position."
        else:
            return "You do not have any more settlements."


    def placing_first_settlement(self):
        for player_num in self.player_order:
            while self.players[player_num].player_settlement > 0:
                print(f"Player {player_num}, please choose a position on the board:")
                settlement_message = self.place_settlement(player_num)
                print(settlement_message)

                if "Settlement placed" in settlement_message:
                    self.players[player_num].player_roads -= 1
                    print(f"Player {player_num} placed a road. Remaining roads: {self.players[player_num].player_roads}")
                    break  # Exit the loop once the settlement is placed
                else:
                    print("Please place a settlement before proceeding.")

            print("")



    def placing_second_settlement(self):
        for player_num in reversed(self.player_order):  # Reverse the player order
            while self.players[player_num].player_settlement > 0:
                print(f"Player {player_num} please choose a position on the board:")
                settlement_message = self.place_settlement(player_num)
                print(settlement_message)

                if "Settlement placed" in settlement_message:
                    self.players[player_num].player_roads -= 1
                    print(f"Player {player_num} placed a road. Remaining roads: {self.players[player_num].player_roads}")
                    break  # Exit the loop once the settlement is placed
                else:
                    print("Please place a settlement before proceeding.")

            print("")



    def print_order(self):
        print("Player turn order:", self.player_order)
        return
    
    @staticmethod
    def get_user_number():
        while True:
            try:
                number = int(input("Enter a number: "))
                return number
            except ValueError:
                print("Invalid input. Please enter a valid number.")




game = Game_2_players()
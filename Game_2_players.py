from Board import Board
from Deck import Deck
from Dice import Dice
from Player import Player

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

    def placing_first_settlement(self):
        for player_num in self.player_order:
            player = self.players[player_num]
            
            print(f"Player {player_num} please choose a position on the board:")
            position = self.get_user_number()
            player.player_settlement -= 1
            print(f"Player {player_num} placed a settlement in the position {position}. Remaining settlements: {player.player_settlement}")
            
            player.player_roads -= 1
            print(f"Player {player_num} placed a road. Remaining roads: {player.player_roads}")
            print("")

    def placing_second_settlement(self):
        for player_num in reversed(self.player_order):  # Reverse the player order
            player = self.players[player_num]
            
            print(f"Player {player_num} please choose a position on the board:")
            position = self.get_user_number()
            player.player_settlement -= 1
            print(f"Player {player_num} placed a settlement in the position {position}. Remaining settlements: {player.player_settlement}")
            
            player.player_roads -= 1
            print(f"Player {player_num} placed a road. Remaining roads: {player.player_roads}")
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
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

    # PLACING SETTLEMENTS AND ROADS

    def place_settlement(self, player_number, position):
        player = self.players[player_number]

        if player.player_settlement <= 0:
            return "You do not have any more settlements."

        is_valid, message = self.check_if_position_settlement_is_valid(position)
        if not is_valid:
            return message

        player.player_settlement -= 1
        self.board.delete_settlement_position(position)
        player.player_settlement_position.add(position)

        return f"Settlement placed at position {position}."
    
    def place_road(self, player_number, position):
        player = self.players[player_number]

        if player.player_roads <= 0:
            return "You do not have any more roads."

        # Validate road placement
        is_valid, message = self.check_if_position_road_is_valid(position)
        if not is_valid:
            return message

        # Place the road
        player.player_roads -= 1
        self.board.delete_road_position(position)
        player.player_road_position.add(position)

        return f"Road placed at position {position}."

    # CHECKING IF THE POSITION IS VALID
        
    def check_if_position_settlement_is_valid(self, position):
        if position not in self.board.settlement_positions:
            return False, "You cannot put a settlement in this position."

        return True, "Valid position for settlement."


    def check_if_position_road_is_valid(self,position):
        if position not in self.board.road_positions:
            return False, "You cannot put a road in this position."

        return True, "Valid position for road."


    # FIRST TURN OF THE GAME, PLACING THE SETTLEMENTS

    def placing_first_settlement(self):
        for player_num in self.player_order:  # Normal player order
            placed = False  # Ensure only one settlement is placed

            while not placed:
                print(f"PLAYER {player_num}, PLEASE CHOOSE A SETTLEMENT POSITION ON THE BOARD:")
                available_settlements = self.board.get_available_settlements()
                print("Available settlements:", available_settlements)

                position = self.get_user_number()
                settlement_message = self.place_settlement(player_num, position)
                print(settlement_message)
                print(f"you have now a total of= {self.players[player_num].player_settlement} settlements")


                if "Settlement placed" in settlement_message:
                    placed = True  # Stop the settlement loop after one successful placement
                    
                    print(f"PLAYER {player_num}, PLEASE CHOOSE A ROAD POSITION ON THE BOARD:")
                    available_roads = self.board.get_available_road_from_settlement(position)
                    print("Available roads:", available_roads)

                    while True:
                        road_position = self.get_user_number()
                        if road_position in available_roads:
                            road_message = self.place_road(player_num, road_position)
                            print(road_message)
                            print("")
                            break
                        else:
                            print("Invalid choice. Please choose a valid road position from the list:", available_roads)

            print("")  # Space between players for readability



    def placing_second_settlement(self):
        for player_num in reversed(self.player_order):  # Reverse the player order
            placed = False  # Ensure only one settlement is placed

            while not placed:
                print(f"PLAYER {player_num}, PLEASE CHOOSE A SETTLEMENT POSITION ON THE BOARD:")
                available_settlements = self.board.get_available_settlements()
                print("Available settlements:", available_settlements)
                position = self.get_user_number()
                settlement_message = self.place_settlement(player_num, position)
                print(settlement_message)

                if "Settlement placed" in settlement_message:
                    placed = True  # Stop the settlement loop after one successful placement
                    
                    print(f"PLAYER {player_num}, PLEASE CHOOSE A ROAD POSITION ON THE BOARD:")
                    available_roads = self.board.get_available_road_from_settlement(position)
                    print("Available roads:", available_roads)

                    while True:
                        road_position = self.get_user_number()
                        if road_position in available_roads:
                            road_message = self.place_road(player_num, road_position)
                            print(road_message)
                            print("")
                            break
                        else:
                            print("Invalid choice. Please choose a valid road position from the list:", available_roads)
            print("")



    def print_order(self):
        print("Player turn order:", self.player_order)
        return
    
    # ASK IN THE TERMINAL FOR THE NUMBER OF THE POSITION
    
    @staticmethod
    def get_user_number():
        while True:
            try:
                number = int(input("Enter a number: "))
                return number
            except ValueError:
                print("Invalid input. Please enter a valid number.")




game = Game_2_players()
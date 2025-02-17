from Board import Board
from Deck import Deck
from Dice import Dice
from Player import Player
import random

class Game_2_players:
    player_order = [1, 2]
    turn_number = 0
    
    def __init__(self):
        self.game()
        
    
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

    
    def check_valid_settlement_once_game_has_begun(self, player):
        all_road_positions = self.players[player].player_road_position
        valid_settlement_positions = []

        for road_position in all_road_positions:
            adjacent_settlements = self.board.get_adjacent_settlement_from_road(road_position)
            if adjacent_settlements is None:
                continue

            for settlement_position in adjacent_settlements:
                is_valid, _ = self.check_if_position_settlement_is_valid(settlement_position)
                if is_valid:
                    valid_settlement_positions.append(settlement_position)

        valid_settlement_positions = list(set(valid_settlement_positions))
        return valid_settlement_positions or []


    def check_valid_road_once_game_has_begun(self, player):
        all_road_positions = self.players[player].player_road_position
        valid_road_positions = []

        for road_position in all_road_positions:
            adjacent_roads = self.board.get_adjacent_roads_from_road(road_position)
            if adjacent_roads is None:
                continue

            for road_position in adjacent_roads:
                is_valid, _ = self.check_if_position_road_is_valid(road_position)
                if is_valid:
                    valid_road_positions.append(road_position)

        valid_road_positions = list(set(valid_road_positions))
        return valid_road_positions or []

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
    

    def game(self):
        # Create player instances
        self.players = {
            1: Player(),
            2: Player(),
        }
        self.board = Board()
        self.dice_1 = Dice()
        self.dice_2 = Dice()
        # self.randomize_order()
        # self.print_order() 
        # self.placing_first_settlement()  
        # self.placing_second_settlement()
        # self.turn(self.turn_number)
        # self.turn(self.turn_number)
        # self.turn(self.turn_number)
        # self.turn(self.turn_number)
        # self.turn(self.turn_number)
        # self.turn(self.turn_number)
        # self.turn(self.turn_number)
        # self.turn(self.turn_number)
        # self.turn(self.turn_number)
        # self.turn(self.turn_number)
        # self.turn(self.turn_number)
        # self.turn(self.turn_number)
        # self.turn(self.turn_number)
        # self.turn(self.turn_number)
        # self.turn(self.turn_number)
        # self.turn(self.turn_number)


        self.place_road(1,11)
        self.place_road(1,12)
        self.place_road(1,20)
        self.place_road(1,26)
        self.place_road(1,25)
        self.place_road(1,19)
        self.place_road(1,27)
        self.place_road(1,7)

        


        self.place_road(2,3)
        self.place_road(2,16)

        valid_positions = self.check_valid_settlement_once_game_has_begun(1)    
        print(f"Player 1 has {valid_positions} valid positions")
        valid_positions = self.check_valid_settlement_once_game_has_begun(2)
        print(f"Player 2 has {valid_positions} valid positions")
    
        valid_road_positions = self.check_valid_road_once_game_has_begun(1)    
        print(f"Player 1 has {valid_road_positions} valid road positions")
        valid_road_positions = self.check_valid_road_once_game_has_begun(2)
        print(f"Player 2 has {valid_road_positions} valid road positions")



        print("")
        print(f"Player 1 roads: {self.players[1].get_player_road_positions()}")
        longest_road = self.find_longest_path(1)
        print(f"Player 1 has a max length road of: {longest_road}")

        print(f"Player 2 roads: {self.players[2].get_player_road_positions()}")
        longest_road = self.find_longest_path(2)
        print(f"Player 2 has a max length road of: {longest_road}")

        return    



    def turn(self, turn_number):
        players_turn = self.whos_turn_is_it(turn_number)
        print("")
        print(f"Player {players_turn}'s turn")
        self.dice_1.roll_dice()
        self.dice_2.roll_dice()
        total_dice = self.dice_1.dice_value + self.dice_2.dice_value
        self.give_cards_to_players(total_dice)
        print(f"The total of the dice is: {total_dice}")

        can_build_road, can_build_settlement, can_build_city, can_buy_dev_cards = self.players[players_turn].check_player_actions()
        
        if can_build_road:
            print("You can build a road")
        else:
            print("You cannot build a road")
        
        if can_build_settlement:
            print("You can build a settlement")
        else:
            print("You cannot build a settlement")
        
        if can_build_city:
            print("You can build a city")
        else:
            print("You cannot build a city")
        
        if can_buy_dev_cards:
            print("You can buy development cards")
        else:
            print("You cannot buy development cards")
        
        self.turn_number += 1
        return

    def whos_turn_is_it(self, turn_number):
        if turn_number % 2 == 0:
            return self.player_order[0]
        else:
            return self.player_order[1]
        
    def give_cards_to_players(self, dice_number):
        # Get the positions where the dice_number is present
        positions = self.board.get_positions_by_dice(dice_number)
        
        # Iterate through all the players
        for player_number, player in self.players.items():
            # Check if the player has a settlement in any of the matching positions
            for position, tiles in positions.items():
                if position in player.player_settlement_position:
                    # Iterate through the tiles in the position
                    for tile in tiles:
                        for resource_value, resource in tile.items():
                            # Only give the resource if the dice number matches the resource value
                            if resource_value == dice_number:
                                # Skip Desert resources
                                if resource == 'Desert':
                                    continue
                                
                                # Add the appropriate resource to the player's resources
                                player.player_ressources[resource] += 1
                                print(f"Player {player_number} receives 1 {resource} for settlement at position {position}")

        return


    ###############################################################################
     ###############################################################################
      ###############################################################################
       ###############################################################################


    def find_longest_path(self, player_number):
        """
        Finds the longest road (number of road segments) for the given player,
        taking branching into account (only one branch counts at an intersection).
        
        The method does the following:
        1. For each road owned by the player, “infer” its two endpoints (intersections)
            by grouping its adjacent roads into clusters that are mutually adjacent.
        2. Build an undirected graph where each unique intersection is a node and 
            each player road (edge) connects its two intersections.
        3. Run a DFS on this graph (avoiding reusing roads) to find the longest path.
        
        Args:
            player_number (int): The player's number whose longest road is calculated.
            
        Returns:
            int: The length (number of road segments) of the longest road.
        """
        player_roads = self.players[player_number].get_player_road_positions()
        
        def get_intersections_for_road(road):
            """
            Given a road, use its list of adjacent roads (from the board’s mapping)
            to “cluster” them into groups that are all mutually adjacent. Each cluster
            represents the other roads that share one intersection with this road.
            
            For example, if road 7 has adjacent roads [1, 11, 12] and 11 and 12 are adjacent,
            then the clusters are [{1}, {11, 12}]. We then define the two endpoints (intersections)
            for road 7 as:
                - frozenset({7} ∪ {1})       i.e. frozenset({1, 7})
                - frozenset({7} ∪ {11, 12})   i.e. frozenset({7, 11, 12})
            """
            nbrs = self.board.get_adjacent_roads_from_road(road)
            clusters = []
            for n in nbrs:
                placed = False
                for cluster in clusters:
                    # If n is adjacent to every road already in the cluster,
                    # then it belongs in that cluster.
                    if all(n in self.board.get_adjacent_roads_from_road(other) for other in cluster):
                        cluster.add(n)
                        placed = True
                        break
                if not placed:
                    clusters.append({n})
            # Every road has two endpoints. If we found only one cluster, use an empty cluster
            # for the other endpoint.
            if len(clusters) == 1:
                clusters.append(set())
            # Define each endpoint as the union of the road itself with one cluster.
            ep1 = frozenset({road} | clusters[0])
            ep2 = frozenset({road} | clusters[1])
            return (ep1, ep2)
        
        # Compute endpoints for each road.
        road_endpoints = {}
        for road in player_roads:
            road_endpoints[road] = get_intersections_for_road(road)
        
        # Build a graph: keys are intersections (nodes), values are lists of tuples
        # (neighbor_intersection, road) representing an edge.
        graph = {}
        for road, (ep1, ep2) in road_endpoints.items():
            for ep in (ep1, ep2):
                if ep not in graph:
                    graph[ep] = []
            # Add the bidirectional edge (annotated with the road number so we avoid reusing it)
            graph[ep1].append((ep2, road))
            graph[ep2].append((ep1, road))
        
        # Now perform DFS to find the longest simple path (by counting edges)
        longest = 0
        def dfs(intersection, used_roads):
            max_length = 0
            for neighbor, road in graph.get(intersection, []):
                if road in used_roads:
                    continue
                used_roads.add(road)
                path_length = 1 + dfs(neighbor, used_roads)
                max_length = max(max_length, path_length)
                used_roads.remove(road)
            return max_length
        
        for intersection in graph:
            longest = max(longest, dfs(intersection, set()))
        return longest



     ###############################################################################
      ###############################################################################
       ###############################################################################
        ###############################################################################


        


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

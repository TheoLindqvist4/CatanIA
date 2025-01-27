import random

class Player:
    player_ressources = {
        'Ore': 0,  
        'Weat': 0, 
        'Sheep': 0,  
        'Brick': 0,
        'Wood': 0 
    }
    player_dev_cards = {
        'Knight': 0,  
        'Year of plenty': 0, 
        'Monopoly': 0,  
        'Road builder': 0,
        'Victory Point': 0 
    }
    player_numbers = {}
    player_points = 0
    player_settlement = 5
    player_cities = 4
    player_roads = 15

    player_longest_road = False
    player_army = False

    player_ore_port = False
    player_weat_port = False
    player_sheep_port = False
    player_brick_port = False
    player_wood_port = False
    player_3_to_1_port = False

class Board:
    def __init__(self):
        self.numbers = [2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12]
        self.rows = [3, 4, 5, 4, 3]  # 3-4-5-4-3 pattern for row lengths
        self.grid = []  # Stores numbers on each tile
        self.tile_grid = []  # Stores tile types on each tile
        self.positions_grid =[] #Stores numbers, tiles based on city position
        self.tiles = {
            'Ore': 3,  
            'Weat': 4, 
            'Sheep': 4,  
            'Brick': 3,
            'Wood': 4,
            'Desert': 1
        }
        self.generate_board()
        self.assign_tiles()
        self.display_board()
        self.assign_intersection_resources()

    def is_adjacent_valid(self, row, col, num):
        # Retrieve the numbers of adjacent tiles
        adjacent_numbers = self.get_adjacent(row, col)

        # Check if the same number exists in adjacent tiles
        if num in adjacent_numbers:
            return False

        # Specific number adjacency restrictions
        if (num == 2 and 12 in adjacent_numbers) or (num == 12 and 2 in adjacent_numbers):
            return False
        if (num == 8 and 6 in adjacent_numbers) or (num == 6 and 8 in adjacent_numbers):
            return False

        return True


    def get_flat_index(self, row, col):
        # Convert a row-column pair to a flat tile index
        row_start_indices = [0, 3, 7, 12, 16]  # Starting indices of tiles in each row
        return row_start_indices[row] + col + 1

    def get_coordinates_from_index(self, index):
        # Convert a flat tile index to a row-column pair
        row_start_indices = [0, 3, 7, 12, 16]  # Starting indices of tiles in each row
        for row, start_index in enumerate(row_start_indices):
            if start_index <= index - 1 < start_index + len(self.grid[row]):
                return row, (index - 1) - start_index
        return None, None  # Return invalid coordinates if the index is out of bounds

    def get_adjacent(self, row, col):
        # Define the adjacency map for the tiles
        adjacency_map = {
            1: [2, 4, 5],
            2: [1, 3, 5, 6],
            3: [2, 6, 7],
            4: [1, 5, 8, 9],
            5: [1, 2, 4, 6, 9, 10],
            6: [2, 3, 5, 7, 10, 11],
            7: [3, 6, 11, 12],
            8: [4, 9, 13],
            9: [4, 5, 8, 10, 13, 14],
            10: [5, 6, 9, 11, 14, 15],
            11: [6, 7, 10, 12, 15, 16],
            12: [7, 11, 16],
            13: [8, 9, 14, 17],
            14: [9, 10, 13, 15, 17, 18],
            15: [10, 11, 14, 16, 18, 19],
            16: [11, 12, 15, 19],
            17: [13, 14, 18],
            18: [14, 15, 17, 19],
            19: [15, 16, 18],
        }

        # Get the flat index of the current tile
        current_tile = self.get_flat_index(row, col)

        # Collect numbers of adjacent tiles
        adjacent_numbers = []
        for adj_tile in adjacency_map.get(current_tile, []):
            adj_row, adj_col = self.get_coordinates_from_index(adj_tile)
            # Only process valid coordinates
            if adj_row is not None and adj_col is not None:
                if 0 <= adj_row < len(self.grid) and 0 <= adj_col < len(self.grid[adj_row]):
                    adjacent_numbers.append(self.grid[adj_row][adj_col])

        return adjacent_numbers

    def get_adjacents_for_positions(self):

        adjacent_to_each_position ={
            1: [4,5],
            2: [5,6],
            3: [6,7],
            4: [1,8],
            5: [1,2,9],
            6: [2,3,10],
            7: [3,11],
            8: [4,12,13],
            9: [5,13,14],
            10: [6,14,15],
            11: [7,15,16],
            12: [8,17],
            13: 
        }
        return adjacents

    def generate_board(self):
        while True:  # Retry until a valid board is generated
            # Initialize the grid with empty slots
            self.grid = [[] for _ in range(5)]
            numbers = self.numbers[:]  # Copy the list of numbers
            random.shuffle(numbers)  # Shuffle numbers for random placement

            # Flag to detect if the generation fails
            generation_failed = False

            for row, row_length in enumerate(self.rows):
                for col in range(row_length):
                    valid_number_found = False  # Track if a valid number is placed
                    for num in numbers:
                        if self.is_adjacent_valid(row, col, num):
                            self.grid[row].append(num)
                            numbers.remove(num)
                            valid_number_found = True
                            break
                    
                    if not valid_number_found:
                        generation_failed = True
                        break  # Break inner loop if placement fails
                
                if generation_failed:
                    break  # Break outer loop if generation fails

            # Validate if board generation is successful
            all_placed = all(len(self.grid[row]) == row_length for row, row_length in enumerate(self.rows))
            if all_placed and not numbers:  # Ensure all numbers are placed
                break  # Exit the loop if successful



    def assign_tiles(self):
        # Initialize an empty grid for tiles following the 3-4-5-4-3 structure
        self.tile_grid = [[] for _ in range(5)]
        
        # Find the location of the number 7 and assign the Desert tile there
        tile_counts = self.tiles.copy()
        for row in range(5):
            for col in range(len(self.grid[row])):
                if self.grid[row][col] == 7:
                    self.tile_grid[row].append('Desert')
                    tile_counts['Desert'] -= 1
                else:
                    self.tile_grid[row].append(None)  # Placeholder for other tiles

        # Flatten available tile positions for easier assignment
        tile_positions = [(r, c) for r in range(5) for c in range(len(self.grid[r])) if self.tile_grid[r][c] is None]
        tile_types = [tile for tile, count in tile_counts.items() for _ in range(count)]
        random.shuffle(tile_types)

        # Assign remaining tiles randomly
        for (row, col), tile_type in zip(tile_positions, tile_types):
            self.tile_grid[row][col] = tile_type

    def display_board(self):
        for num_row, tile_row in zip(self.grid, self.tile_grid):
            row_display = " ".join(f"{num}({tile})" for num, tile in zip(num_row, tile_row))
            print(row_display)   

    def assign_intersection_resources(self):
        # Associer chaque tuile aux IDs correspondants
        tile_to_ids = {
            1: [1, 4, 5, 8, 9, 13],
            2: [2, 5, 6, 9, 10, 14],
            3: [3, 6, 7, 10, 11, 15],
            4: [8, 12, 13, 17, 18, 23],
            5: [9, 13, 14, 18, 19, 24],
            6: [10, 14, 15, 19, 20, 25],
            7: [11, 15, 16, 20, 21, 26],
            8: [17, 22, 23, 28, 29, 34],
            9: [18, 23, 24, 29, 30, 35],
            10: [19, 24, 25, 30, 31, 36],
            11: [20, 25, 26, 31, 32, 37],
            12: [21, 26, 27, 32, 33, 38],
            13: [29, 34, 35, 39, 40, 44],
            14: [30, 35, 36, 40, 41, 45],
            15: [31, 36, 37, 41, 42, 46],
            16: [32, 37, 38, 42, 43, 47],
            17: [40, 44, 45, 48, 49, 52],
            18: [41, 45, 46, 49, 50, 53],
            19: [42, 46, 47, 50, 51, 54]
        }


        # Dictionnaire final des intersections avec les ressources
        intersection_resources = {}

        # Fonction pour obtenir l'indice (row, col) basé sur le numéro de tuile
        def get_row_col(tile):
            counter = 0
            for row, row_length in enumerate(self.rows):
                if counter <= tile - 1 < counter + row_length:
                    col = tile - 1 - counter
                    return row, col
                counter += row_length
            raise ValueError(f"Tile {tile} not found in grid structure.")

        # Parcourir les IDs et les tuiles correspondantes
        for tile, ids in tile_to_ids.items():
            for id_ in ids:
                try:
                    # Récupérer la ligne et la colonne de la tuile
                    row, col = get_row_col(tile)
                    resource = self.tile_grid[row][col]
                    number = self.grid[row][col]

                    # Ajouter la ressource et le numéro au dictionnaire
                    if id_ not in intersection_resources:
                        intersection_resources[id_] = []
                    intersection_resources[id_].append({number: resource})
                except (IndexError, ValueError) as e:
                    print(f"Error processing tile {tile} for ID {id_}: {e}")

        # Trier les ID par ordre croissant
        sorted_intersections = dict(sorted(intersection_resources.items()))
        self.positions_grid = sorted_intersections

        print(sorted_intersections)
        return 





 

class Deck:
    resource_ore = 21
    resource_weat = 21
    resource_sheep = 21
    resource_brick = 21
    resource_wood = 21

    dev_knight = 14
    dev_victory_points = 5
    dev_road_builder = 2
    dev_year_of_plenty = 2
    dev_monopoly = 2



    
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




# Example usage:
game = Game_2_players()

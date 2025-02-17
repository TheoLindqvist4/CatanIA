import random

class Board:
    def __init__(self):
        self.numbers = [2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12]
        self.rows = [3, 4, 5, 4, 3]  # 3-4-5-4-3 pattern for row lengths
        self.grid = []  # Stores numbers on each tile
        self.tile_grid = []  # Stores tile types on each tile
        self.positions_grid =[] #Stores numbers, tiles based on city position
        self.settlement_positions = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 
            39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54
            ]
        self.road_positions = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 
            39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 
            57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72
        ]

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

    # CREATION OF THE BOARD

    def get_flat_index(self, row, col):
        row_start_indices = [0, 3, 7, 12, 16] 
        return row_start_indices[row] + col + 1

    def get_coordinates_from_index(self, index):
        row_start_indices = [0, 3, 7, 12, 16]  
        for row, start_index in enumerate(row_start_indices):
            if start_index <= index - 1 < start_index + len(self.grid[row]):
                return row, (index - 1) - start_index
        return None, None 


    def generate_board(self):
        while True: 
            self.grid = [[] for _ in range(5)]
            numbers = self.numbers[:] 
            random.shuffle(numbers)  

            generation_failed = False

            for row, row_length in enumerate(self.rows):
                for col in range(row_length):
                    valid_number_found = False  
                    for num in numbers:
                        if self.is_adjacent_valid(row, col, num):
                            self.grid[row].append(num)
                            numbers.remove(num)
                            valid_number_found = True
                            break
                    
                    if not valid_number_found:
                        generation_failed = True
                        break  
                
                if generation_failed:
                    break  

            all_placed = all(len(self.grid[row]) == row_length for row, row_length in enumerate(self.rows))
            if all_placed and not numbers:  
                break  



    def assign_tiles(self):
        self.tile_grid = [[] for _ in range(5)]
        
        tile_counts = self.tiles.copy()
        for row in range(5):
            for col in range(len(self.grid[row])):
                if self.grid[row][col] == 7:
                    self.tile_grid[row].append('Desert')
                    tile_counts['Desert'] -= 1
                else:
                    self.tile_grid[row].append(None) 

        tile_positions = [(r, c) for r in range(5) for c in range(len(self.grid[r])) if self.tile_grid[r][c] is None]
        tile_types = [tile for tile, count in tile_counts.items() for _ in range(count)]
        random.shuffle(tile_types)

        for (row, col), tile_type in zip(tile_positions, tile_types):
            self.tile_grid[row][col] = tile_type

    def assign_intersection_resources(self):
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

        intersection_resources = {}

        def get_row_col(tile):
            counter = 0
            for row, row_length in enumerate(self.rows):
                if counter <= tile - 1 < counter + row_length:
                    col = tile - 1 - counter
                    return row, col
                counter += row_length
            raise ValueError(f"Tile {tile} not found in grid structure.")

        for tile, ids in tile_to_ids.items():
            for id_ in ids:
                try:
                    row, col = get_row_col(tile)
                    resource = self.tile_grid[row][col]
                    number = self.grid[row][col]

                    if id_ not in intersection_resources:
                        intersection_resources[id_] = []
                    intersection_resources[id_].append({number: resource})
                except (IndexError, ValueError) as e:
                    print(f"Error processing tile {tile} for ID {id_}: {e}")

        sorted_intersections = dict(sorted(intersection_resources.items()))
        self.positions_grid = sorted_intersections

        print(sorted_intersections)
        return 
    
    # DISPLAY OF THE BOARD
    
    def display_board(self):
        for num_row, tile_row in zip(self.grid, self.tile_grid):
            row_display = " ".join(f"{num}({tile})" for num, tile in zip(num_row, tile_row))
            print(row_display)
    
    # MAP FOR ADJACENT ROAD AND SETTLEMENT POSITION
    
    
    def get_adjacent(self, row, col):
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

        current_tile = self.get_flat_index(row, col)

        adjacent_numbers = []
        for adj_tile in adjacency_map.get(current_tile, []):
            adj_row, adj_col = self.get_coordinates_from_index(adj_tile)
            if adj_row is not None and adj_col is not None:
                if 0 <= adj_row < len(self.grid) and 0 <= adj_col < len(self.grid[adj_row]):
                    adjacent_numbers.append(self.grid[adj_row][adj_col])

        return adjacent_numbers
    
    def get_adjacents_for_positions(self, position):
        adjacent_to_each_position = {
            1: [4, 5],
            2: [5, 6],
            3: [6, 7],
            4: [1, 8],
            5: [1, 2, 9],
            6: [2, 3, 10],
            7: [3, 11],
            8: [4, 12, 13],
            9: [5, 13, 14],
            10: [6, 14, 15],
            11: [7, 15, 16],
            12: [8, 17],
            13: [8, 9, 18],
            14: [9, 10, 19],
            15: [10, 11, 20],
            16: [11, 21],
            17: [12, 22, 23],
            18: [13, 23, 24],
            19: [14, 24, 25],
            20: [15, 25, 26],
            21: [16, 26, 27],
            22: [17, 28],
            23: [17, 18, 29],
            24: [18, 19, 30],
            25: [19, 20, 31],
            26: [20, 21, 32],
            27: [21, 33],
            28: [22, 34],
            29: [23, 34, 35],
            30: [24, 35, 36],
            31: [25, 36, 37],
            32: [26, 37, 38],
            33: [27, 38],
            34: [28, 29, 39],
            35: [29, 30, 40],
            36: [30, 31, 41],
            37: [31, 32, 42],
            38: [32, 33, 43],
            39: [34, 44],
            40: [35, 44, 45],
            41: [36, 45, 46],
            42: [37, 46, 47],
            43: [38, 47],
            44: [39, 40, 48],
            45: [40, 41, 49],
            46: [41, 42, 50],
            47: [42, 43, 51],
            48: [44, 52],
            49: [45, 52, 53],
            50: [46, 53, 54],
            51: [47, 54],
            52: [48, 49],
            53: [49, 50],
            54: [50, 51]
        }

        if not isinstance(position, int) or not (1 <= position <= 54):
            return "The number must be between 1 and 54."

        return adjacent_to_each_position.get(position)

    def get_adjacent_roads_from_settlement(self,position):
        adjacent_to_each_position = {
            1: [1,2],
            2: [3,4],
            3: [5,6],
            4: [1,7],
            5: [2,3,8],
            6: [4,5,9],
            7: [6,10],
            8: [7,11,12],
            9: [8,13,14],
            10: [9,15,16],
            11: [10,17,18],
            12: [11,19],
            13: [12,13,20],
            14: [14,15,21],
            15: [16,17,22],
            16: [18,23],
            17: [19,24,25],
            18: [20,26,27],
            19: [21,28,29],
            20: [22,30,31],
            21: [23,32,33],
            22: [24,34],
            23: [25,26,35],
            24: [27,28,36],
            25: [29,30,37],
            26: [31,32,38],
            27: [33,39],
            28: [34,40],
            29: [35,41,42],
            30: [36,43,44],
            31: [37,45,46],
            32: [38,47,48],
            33: [39,49],
            34: [40,41,50],
            35: [42,43,51],
            36: [44,45,52],
            37: [46,47,53],
            38: [48,49,54],
            39: [50,55],
            40: [51,56,57],
            41: [52,58,59],
            42: [53,60,61],
            43: [54,62],
            44: [55,56,63],
            45: [57,58,64],
            46: [59,60,65],
            47: [61,62,66],
            48: [63,67],
            49: [64,68,69],
            50: [65,70,71],
            51: [66,72],
            52: [67,68],
            53: [69,70],
            54: [71,72]
        }
        if not isinstance(position, int) or not (1 <= position <= 54):
            return "The number must be between 1 and 54."

        return adjacent_to_each_position.get(position)
    
    def get_adjacent_roads_from_road(self,road_positiion):
        adjacent_road_to_road_position = {
            1: [2,7],
            2: [1,8],
            3: [2,4,8],
            4: [3,5,9],
            5: [4,6,9],
            6: [5,10],
            7: [1,11,12],
            8: [2,3,13,14],
            9: [4,5,15,16],
            10: [6,17,18],
            11: [7,12,19],
            12: [7,11,13,20],
            13: [8,12,14,20],
            14: [8,13,15,21],
            15: [9,14,16,21],
            16: [9,15,17,22],
            17: [10,16,18,22],
            18: [10,17,23],
            19: [11,24,25],
            20: [12,13,26,27],
            21: [14,15,28,29],
            22: [16,17,30,31],
            23: [18,32,33],
            24: [19,25,34],
            25: [19,24,26,35],
            26: [20,25,27,35],
            27: [20,26,28,36],
            28: [21,27,29,36],
            29: [21,28,30,37],
            30: [22,29,31,37],
            31: [22,30,32,38],
            32: [23,31,33,38],
            33: [23,32,39],
            34: [24,40],
            35: [25,26,41,42],
            36: [27,28,43,44],
            37: [29,30,45,46],
            38: [31,32,47,48],
            39: [33,49],
            40: [34,41,50],
            41: [35,40,42,50],
            42: [35,41,43,51],
            43: [36,42,44,51],
            44: [36,43,45,52],
            45: [37,44,46,52],
            46: [37,45,47,53],
            47: [38,46,48,53],
            48: [38,47,49,54],
            49: [39,48,54],
            50: [40,41,55],
            51: [42,56,57],
            52: [44,45,58,59],
            53: [46,47,60,61],
            54: [48,49,62],
            55: [50,56,63],
            56: [51,55,57,63],
            57: [51,56,58,64],
            58: [52,57,59,64],
            59: [52,58,60,65],
            60: [53,59,61,65],
            61: [53,60,62,66],
            62: [54,61,66],
            63: [55,56,67],
            64: [57,58,68,69],
            65: [59,60,70,71],
            66: [61,62,72],
            67: [63,68],
            68: [64,67,69],
            69: [64,68,70],
            70: [65,69,71],
            71: [65,70,72],
            72: [66,71]
        }
        if not isinstance(road_positiion, int) or not (1 <= road_positiion <= 72):
            return "The number must be between 1 and 72."

        return adjacent_road_to_road_position.get(road_positiion)

    def get_adjacent_settlement_from_road(self,road_positiion):
        adjacent_settlement_to_road_position = {
            1: [1,4],
            2: [1,5],
            3: [2,5],
            4: [2,6],
            5: [3,6],
            6: [3,7],
            7: [4,8],
            8: [5,9],
            9: [6,10],
            10: [7,11],
            11: [8,12],
            12: [8,13],
            13: [9,13],
            14: [9,14],
            15: [10,14],
            16: [10,15],
            17: [11,15],
            18: [11,16],
            19: [12,17],
            20: [13,18],
            21: [14,19],
            22: [15,20],
            23: [16,21],
            24: [17,22],
            25: [17,23],
            26: [18,23],
            27: [18,24],
            28: [19,24],
            29: [19,25],
            30: [20,25],
            31: [20,26],
            32: [21,26],
            33: [21,27],
            34: [22,28],
            35: [23,29],
            36: [24,30],
            37: [25,31],
            38: [26,32],
            39: [27,33],
            40: [28,34],
            41: [29,34],
            42: [29,35],
            43: [30,35],
            44: [30,36],
            45: [31,36],
            46: [31,37],
            47: [32,37],
            48: [32,38],
            49: [33,38],
            50: [34,39],
            51: [35,40],
            52: [36,41],
            53: [37,42],
            54: [38,43],
            55: [39,44],
            56: [40,44],
            57: [40,45],
            58: [41,45],
            59: [41,46],
            60: [42,46],
            61: [42,47],
            62: [43,47],
            63: [44,48],
            64: [45,49],
            65: [46,50],
            66: [47,51],
            67: [48,52],
            68: [49,52],
            69: [49,53],
            70: [50,53],
            71: [50,54],
            72: [51,54]
        }
        if not isinstance(road_positiion, int) or not (1 <= road_positiion <= 72):
            return "The number must be between 1 and 72."

        return adjacent_settlement_to_road_position.get(road_positiion)

    # IS THE POSITION VALID OR NOT
    
    def is_adjacent_valid(self, row, col, num):
        adjacent_numbers = self.get_adjacent(row, col)

        if num in adjacent_numbers:
            return False

        if (num == 2 and 12 in adjacent_numbers) or (num == 12 and 2 in adjacent_numbers):
            return False
        if (num == 8 and 6 in adjacent_numbers) or (num == 6 and 8 in adjacent_numbers):
            return False

        return True
    
    def is_settlement_position_available(self, number):
        """
        Checks if a settlement position is available.

        Args:
            number (int): The position number to check.

        Returns:
            bool: True if the position is available, False otherwise.
        """
        return number in self.settlement_positions
    
    def is_road_position_available(self, number):
        """
        Checks if a road position is available.

        Args:
            number (int): The position number to check.

        Returns:
            bool: True if the position is available, False otherwise.
        """
        return number in self.road_positions
    
    # DELETE THE SETTLEMENT OR ROAD POSITIONS FROM THE BOARD

    def delete_settlement_position(self, number):
        """
        Deletes a position and its adjacent positions from settlement_positions.

        Args:
            number (int): The position number to be deleted.
        """
        if self.is_settlement_position_available(number):
            self.settlement_positions.remove(number)

        adjacent_positions = self.get_adjacents_for_positions(number)
        for adjacent in adjacent_positions:
            if self.is_settlement_position_available(adjacent):
                self.settlement_positions.remove(adjacent)


    def delete_road_position(self, number):
        """
        Deletes a position from road_positions.

        Args:
            number (int): The position number to be deleted.
        """
        if self.is_road_position_available(number):
            self.road_positions.remove(number)


    # GETTERS

    def get_available_settlements(self):
        return self.settlement_positions
    
    def get_available_road(self):
        return self.road_positions
    
    def get_available_road_from_settlement(self,settlement_position):
        return self.get_adjacent_roads_from_settlement(settlement_position)
    
    def get_positions_by_dice(self, dice_number):
        matching_positions = {}

        # Iterate through all positions in the grid
        for position, tiles in self.positions_grid.items():
            for tile in tiles:
                # Check if the dice_number exists in the current tile
                if dice_number in tile:
                    matching_positions[position] = tiles
                    break  # No need to check further tiles for this position

        return matching_positions

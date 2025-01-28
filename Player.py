class Player:
    def __init__(self):
        self.player_ressources = {
            'Ore': 0,  
            'Weat': 0, 
            'Sheep': 0,  
            'Brick': 0,
            'Wood': 0 
        }
        self.player_dev_cards = {
            'Knight': 0,  
            'Year of plenty': 0, 
            'Monopoly': 0,  
            'Road builder': 0,
            'Victory Point': 0 
        }
        self.player_numbers = set()
        self.player_points = 0
        self.player_settlement = 5
        self.player_cities = 4
        self.player_roads = 15

        self.player_longest_road = False
        self.player_army = False

        self.player_ore_port = False
        self.player_weat_port = False
        self.player_sheep_port = False
        self.player_brick_port = False
        self.player_wood_port = False
        self.player_3_to_1_port = False

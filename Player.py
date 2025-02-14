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
        self.player_points = 0
        self.player_settlement = 5
        self.player_cities = 4
        self.player_roads = 15
        self.player_road_position = set()
        self.player_settlement_position = set()
        self.player_city_position = set()

        self.player_longest_road = False
        self.player_army = False

        self.player_ore_port = False
        self.player_weat_port = False
        self.player_sheep_port = False
        self.player_brick_port = False
        self.player_wood_port = False
        self.player_3_to_1_port = False


    def get_player_road_positions(self):
        return self.player_road_position
    
    def get_player_city_position(self):
        return self.player_city_position
    
    def get_player_settlement_position(self):
        return self.player_settlement_position

    def check_player_actions(self):
        can_build_road = False
        can_build_settlement = False
        can_build_city = False
        can_buy_dev_cards = False

        if self.player_ressources['Brick'] > 0 and self.player_ressources['Wood'] > 0:
            can_build_road = True

        if (self.player_ressources['Brick'] > 0 and 
            self.player_ressources['Wood'] > 0 and 
            self.player_ressources['Sheep'] > 0 and 
            self.player_ressources['Weat'] > 0):
            can_build_settlement = True

        if self.player_ressources['Weat'] > 1 and self.player_ressources['Ore'] > 2:
            can_build_city = True

        if (self.player_ressources['Sheep'] > 0 and 
            self.player_ressources['Weat'] > 0 and 
            self.player_ressources['Ore'] > 0):
            can_buy_dev_cards = True

        return can_build_road, can_build_settlement, can_build_city, can_buy_dev_cards

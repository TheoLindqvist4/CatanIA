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
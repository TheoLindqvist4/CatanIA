"""Game-level Phase 0 guarantees: instance state, seeding, longest road."""

import random

import pytest

import catan.topology as T
from Deck import Deck
from Dice import Dice
from Game_2_players import Game_2_players


def make_game(seed=0):
    return Game_2_players(seed=seed)


# --------------------------------------------------------------------------- #
# Instance state, not class state                                             #
# --------------------------------------------------------------------------- #

def test_player_order_is_not_shared_between_games():
    """As class attributes these were one list shared by every game in the process,
    so shuffling turn order in one rollout silently reordered all the others."""
    a, b = make_game(1), make_game(2)
    assert a.player_order is not b.player_order
    assert a.player_order is not Game_2_players.__dict__.get("player_order")

    a.player_order.reverse()
    assert b.player_order == [1, 2]


def test_randomize_order_does_not_leak_into_other_games():
    a, b = make_game(1), make_game(2)
    for _ in range(20):
        a.randomize_order()
    assert sorted(a.player_order) == [1, 2]
    assert b.player_order == [1, 2]


def test_turn_number_is_per_instance():
    a, b = make_game(1), make_game(2)
    a.turn_number = 5
    assert b.turn_number == 0


def test_players_and_board_are_not_shared():
    a, b = make_game(1), make_game(2)
    assert a.board is not b.board
    assert a.players[1] is not b.players[1]
    assert a.players[1].player_road_position is not b.players[1].player_road_position

    a.place_road(1, 30)
    assert b.players[1].player_road_position == set()
    assert b.board.is_road_position_available(30)


def test_class_attributes_no_longer_exist():
    for leaked in ("player_order", "turn_number"):
        assert leaked not in vars(Game_2_players), f"{leaked} is still class state"
    assert "dice_value" not in vars(Dice)


# --------------------------------------------------------------------------- #
# Determinism                                                                 #
# --------------------------------------------------------------------------- #

def test_same_seed_reproduces_the_whole_game_setup():
    a, b = make_game(42), make_game(42)
    assert a.board.grid == b.board.grid
    assert a.board.tile_grid == b.board.tile_grid
    assert a.randomize_order() == b.randomize_order()
    assert [a.dice_1.roll_dice() for _ in range(20)] == \
           [b.dice_1.roll_dice() for _ in range(20)]


def test_game_does_not_consume_the_global_random_stream():
    random.seed(5)
    expected = [random.random() for _ in range(3)]
    random.seed(5)
    g = make_game(1)
    g.randomize_order()
    g.dice_1.roll_dice()
    assert [random.random() for _ in range(3)] == expected


def test_legal_move_enumeration_is_sorted_for_reproducibility():
    g = make_game(0)
    for road in (11, 12, 20, 26, 25, 19, 27, 7):
        g.place_road(1, road)
    settlements = g.check_valid_settlement_once_game_has_begun(1)
    roads = g.check_valid_road_once_game_has_begun(1)
    assert settlements == sorted(settlements)
    assert roads == sorted(roads)
    assert len(set(settlements)) == len(settlements)


# --------------------------------------------------------------------------- #
# Dice and deck                                                               #
# --------------------------------------------------------------------------- #

def test_a_fresh_die_has_no_value_until_rolled():
    assert Dice(rng=random.Random(0)).dice_value is None


def test_rolls_stay_in_range_and_cover_every_face():
    die = Dice(rng=random.Random(0))
    seen = {die.roll_dice() for _ in range(400)}
    assert seen == {1, 2, 3, 4, 5, 6}


def test_deck_counts_are_standard_and_per_instance():
    a, b = Deck(), Deck()
    assert a.resources == {'Ore': 19, 'Weat': 19, 'Sheep': 19, 'Brick': 19, 'Wood': 19}
    assert sum(a.dev_cards.values()) == 25
    assert a.dev_cards['Knight'] == 14

    a.resources['Ore'] -= 5
    assert b.resources['Ore'] == 19


# --------------------------------------------------------------------------- #
# Longest road                                                                #
# --------------------------------------------------------------------------- #

def longest_for(roads, seed=0):
    g = make_game(seed)
    g.players[1].player_road_position = set(roads)
    return g.find_longest_path(1)


def test_no_roads_is_zero():
    assert longest_for([]) == 0


def test_a_single_road_is_one():
    assert longest_for([30]) == 1


def test_disconnected_roads_do_not_add_up():
    assert longest_for([3, 16]) == 1


def test_a_straight_chain_counts_every_segment():
    # 1=(1,4) 2=(1,5) 3=(2,5) 4=(2,6): the path 4-1-5-2-6
    assert longest_for([1, 2, 3, 4]) == 4


def test_a_branch_counts_only_one_arm():
    # roads 22=(15,20) 30=(20,25) 31=(20,26) form a Y centred on vertex 20
    assert longest_for([22, 30, 31]) == 2


def test_chains_crossing_vertex_5_are_measured_correctly():
    """Road 2's neighbour list used to omit road 3, breaking chains through vertex 5."""
    assert longest_for([2, 3]) == 2
    assert longest_for([2, 8]) == 2
    assert longest_for([2, 3, 4]) == 3


def test_chains_crossing_vertex_35_are_measured_correctly():
    """Road 51's neighbour list used to omit road 43, breaking chains through vertex 35."""
    assert longest_for([43, 51]) == 2
    assert longest_for([42, 43]) == 2
    assert longest_for([42, 51]) == 2


def test_a_route_may_not_reuse_an_intersection():
    """Strict simple path: 6, not the 7 the roads-only version gave.

    Vertex 8 carries all three of the player's roads. Counting 7 required passing
    through it twice. Settled in
    docs/decisions/0006-longest-road-intersection-reuse.md.
    """
    assert longest_for([7, 11, 12, 19, 20, 25, 26, 27]) == 6


def test_longest_road_never_exceeds_the_road_count():
    g = make_game(0)
    rng = random.Random(7)
    for _ in range(30):
        roads = set(rng.sample(range(1, T.NUM_ROADS + 1), 12))
        g.players[1].player_road_position = roads
        assert 0 <= g.find_longest_path(1) <= len(roads)


# --------------------------------------------------------------------------- #
# Placement                                                                   #
# --------------------------------------------------------------------------- #

def test_a_road_cannot_be_built_twice():
    g = make_game(0)
    assert "Road placed" in g.place_road(1, 30)
    assert "cannot" in g.place_road(2, 30)
    assert g.players[2].player_road_position == set()


def test_a_settlement_blocks_its_neighbours_for_everyone():
    g = make_game(0)
    assert "Settlement placed" in g.place_settlement(1, 20)
    for neighbour in T.VERTEX_NEIGHBOURS[20]:
        assert "cannot" in g.place_settlement(2, neighbour)


def test_piece_limits_are_enforced():
    g = make_game(0)
    placed = sum("Road placed" in g.place_road(1, r) for r in range(1, T.NUM_ROADS + 1))
    assert placed == 15
    assert g.players[1].player_roads == 0
    assert "do not have any more roads" in g.place_road(1, 60)


def test_setup_phase_never_costs_a_player_their_road(monkeypatch, capsys):
    """A rejected road choice used to break out of the loop, so the player kept the
    settlement but silently lost the road."""
    g = make_game(0)
    g.place_road(2, 30)  # occupy one of vertex 20's three roads

    answers = iter([
        999,  # not a buildable settlement
        20,   # valid settlement
        30,   # already taken by player 2
        22,   # valid
    ])
    monkeypatch.setattr(Game_2_players, "get_user_number",
                        staticmethod(lambda: next(answers)))

    g._place_starting_pair(1)
    capsys.readouterr()

    assert g.players[1].player_settlement_position == {20}
    assert g.players[1].player_road_position == {22}
    assert g.players[1].player_settlement == 4
    assert g.players[1].player_roads == 14


# --------------------------------------------------------------------------- #
# Turn order                                                                  #
# --------------------------------------------------------------------------- #

def test_turn_order_cycles_over_the_player_list():
    g = make_game(0)
    g.player_order = [2, 1]
    assert [g.whos_turn_is_it(t) for t in range(5)] == [2, 1, 2, 1, 2]


def test_turn_order_generalises_beyond_two_players():
    """`turn_number % 2` was hardcoded; Phase 2 needs 3-4 players."""
    g = make_game(0)
    g.player_order = [3, 1, 2]
    assert [g.whos_turn_is_it(t) for t in range(6)] == [3, 1, 2, 3, 1, 2]


def test_turn_advances_the_counter(capsys):
    g = make_game(0)
    g.turn(0)
    capsys.readouterr()
    assert g.turn_number == 1


def test_production_pays_only_settlement_owners(capsys):
    g = make_game(0)
    board = g.board
    # find a vertex and a dice number it actually produces on
    vertex, dice = next(
        (v, board.number_at(t))
        for v in range(1, T.NUM_VERTICES + 1)
        for t in T.VERTEX_TILES[v]
        if board.resource_at(t) != 'Desert'
    )
    g.players[1].player_settlement_position.add(vertex)

    g.give_cards_to_players(dice)
    capsys.readouterr()

    assert sum(g.players[1].player_ressources.values()) >= 1
    assert sum(g.players[2].player_ressources.values()) == 0


def test_rolling_a_seven_pays_nobody(capsys):
    """The robber is Phase 2; until then a 7 must simply be inert."""
    g = make_game(0)
    for v in range(1, T.NUM_VERTICES + 1):
        g.players[1].player_settlement_position.add(v)
    g.give_cards_to_players(7)
    capsys.readouterr()
    assert sum(g.players[1].player_ressources.values()) == 0

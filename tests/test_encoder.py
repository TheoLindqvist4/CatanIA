"""The observation encoder.

The important tests here are the **leak detectors**: mutate something a player is not
entitled to see and assert their observation does not move. Everything else is shape.
"""

import random

import pytest

import catan.topology as T
from catan import dice, encoder as E, rules
from catan.board import GENERIC_HARBOUR
from catan.dev_cards import DevCard
from catan.resources import NUM_RESOURCES, Resource
from catan.rulesets import ALL, BASE_GAME, RANKED_1V1
from catan.state import MAX_PLAYERS, GameState, NO_OWNER, Phase, Piece
from helpers import (
    complete_setup,
    enough_for_everything,
    fresh,
    give,
    in_build_phase,
    play_random_game,
    put_building,
    put_road,
)


def mid_game(seed=1, ruleset=None, num_players=2):
    """A state with buildings, roads, cards and an award in play."""
    state = fresh(seed=seed, ruleset=ruleset, num_players=num_players)
    complete_setup(state)
    in_build_phase(state, 1)
    enough_for_everything(state, 1)
    state.dev_cards[1][DevCard.KNIGHT] = 2
    state.dev_cards[2][DevCard.MONOPOLY] = 1
    state.knights_played[1] = 3
    rules.update_awards(state)
    return state


# =========================================================================== #
# SHAPE                                                                       #
# =========================================================================== #

def test_the_layout_covers_the_vector_exactly_once():
    covered = 0
    for name in ("tiles", "vertices", "roads", "players", "global"):
        span = E.LAYOUT[name]
        assert span.start == covered, f"a gap or overlap before {name}"
        covered = span.stop
    assert covered == E.SIZE


def test_repeated_blocks_have_consistent_shapes():
    for name, (rows, width) in E.SHAPES.items():
        span = E.LAYOUT[name]
        assert span.stop - span.start == rows * width
    assert E.SHAPES["tiles"][0] == T.NUM_TILES
    assert E.SHAPES["vertices"][0] == T.NUM_VERTICES
    assert E.SHAPES["roads"][0] == T.NUM_ROADS
    assert E.SHAPES["players"][0] == MAX_PLAYERS


def test_the_length_never_varies():
    """A network's input shape cannot depend on the situation."""
    lengths = set()
    for ruleset in ALL:
        for num_players in (2, 3, 4):
            state = fresh(seed=1, ruleset=ruleset, num_players=num_players)
            lengths.add(len(E.encode(state, 1)))
            complete_setup(state)
            lengths.add(len(E.encode(state, 1)))
            state.phase = Phase.GAME_OVER
            state.winner = 1
            lengths.add(len(E.encode(state, 1)))
    assert lengths == {E.SIZE}


def test_every_value_is_scaled_into_the_unit_range():
    for seed in range(8):
        state = play_random_game(seed=seed, max_actions=1200)
        for player in state.players:
            observation = E.encode(state, player)
            assert all(isinstance(v, float) for v in observation)
            out_of_range = [v for v in observation if not 0.0 <= v <= 1.0]
            assert not out_of_range, f"unscaled values: {out_of_range[:5]}"


def test_an_unknown_player_is_rejected():
    state = fresh(seed=1, num_players=2)
    for bad in (0, 3, -1):
        with pytest.raises(ValueError):
            E.encode(state, bad)


def test_the_default_view_is_the_current_player():
    state = mid_game()
    assert E.encode(state) == E.encode(state, state.current_player)


def test_block_pulls_out_reshaped_rows():
    state = mid_game()
    observation = E.encode(state, 1)
    tiles = E.block(observation, "tiles")
    assert len(tiles) == T.NUM_TILES
    assert all(len(row) == E.TILE_FEATURES for row in tiles)
    assert E.block(observation, "global") == observation[E.LAYOUT["global"]]


# =========================================================================== #
# HIDDEN INFORMATION — the leak detectors                                     #
# =========================================================================== #

def test_an_opponents_hand_composition_is_invisible():
    """Only the size is public, because cards are countable but not readable."""
    state = mid_game()
    give(state, 2, wood=5)
    before = E.encode(state, 1)

    give(state, 2, ore=5)          # same size, entirely different cards
    assert E.encode(state, 1) == before

    give(state, 2, wood=6)         # a different size does show
    assert E.encode(state, 1) != before


def test_my_own_hand_composition_is_visible():
    state = mid_game()
    give(state, 1, wood=5)
    before = E.encode(state, 1)
    give(state, 1, ore=5)
    assert E.encode(state, 1) != before


def test_an_opponents_development_card_composition_is_invisible():
    state = mid_game()
    state.dev_cards[2] = [3, 0, 0, 0, 0]      # three Knights
    before = E.encode(state, 1)

    state.dev_cards[2] = [0, 3, 0, 0, 0]      # three Victory Points instead
    assert E.encode(state, 1) == before, "the kind of card leaked"

    state.dev_cards[2] = [0, 4, 0, 0, 0]      # a different count does show
    assert E.encode(state, 1) != before


def test_an_opponents_hidden_victory_points_are_invisible():
    """A Victory Point card is worth a point but stays hidden until it wins."""
    state = mid_game()
    state.dev_cards[2] = [0, 0, 0, 0, 1]
    before = E.encode(state, 1)

    state.dev_cards[2] = [0, 1, 0, 0, 0]      # same count, now worth a point
    assert rules.victory_points(state, 2) != rules.public_victory_points(state, 2)
    assert E.encode(state, 1) == before, "an opponent's hidden points leaked"


def test_my_own_hidden_victory_points_are_visible_to_me():
    state = mid_game()
    before = E.encode(state, 1)
    state.dev_cards[1][DevCard.VICTORY_POINT] += 1
    assert E.encode(state, 1) != before


def test_the_development_deck_order_is_invisible():
    """Only how many are left is public. Seeing the order is seeing the future."""
    state = mid_game()
    before = E.encode(state, 1)

    state.dev_deck.reverse()
    assert E.encode(state, 1) == before

    state.rng.shuffle(state.dev_deck)
    assert E.encode(state, 1) == before

    state.dev_deck.pop()                       # a shorter deck does show
    assert E.encode(state, 1) != before


def test_which_development_cards_remain_is_invisible():
    state = mid_game()
    state.dev_deck = [DevCard.KNIGHT] * 10
    before = E.encode(state, 1)
    state.dev_deck = [DevCard.VICTORY_POINT] * 10
    assert E.encode(state, 1) == before


def test_the_balanced_dice_deck_is_invisible():
    """It is not encoded at all — it would be a preview of the next 24 rolls."""
    state = mid_game(ruleset=RANKED_1V1)
    assert state.dice_deck is not None
    before = E.encode(state, 1)

    state.dice_deck.reverse()
    assert E.encode(state, 1) == before
    state.dice_deck = dice.new_deck(random.Random(99))
    assert E.encode(state, 1) == before


def test_an_opponent_sees_the_mirror_of_what_i_hide():
    """The masking is per-observer, not global: what I hide from them, they hide from me."""
    state = mid_game()
    state.dev_cards[1] = [2, 1, 0, 0, 0]
    state.dev_cards[2] = [0, 0, 1, 1, 1]
    give(state, 1, wood=4)
    give(state, 2, ore=4)

    mine, theirs = E.encode(state, 1), E.encode(state, 2)
    assert mine != theirs

    # my composition changes my view but not theirs
    state.dev_cards[1] = [1, 2, 0, 0, 0]
    assert E.encode(state, 1) != mine
    assert E.encode(state, 2) == theirs


# =========================================================================== #
# PERSPECTIVE ROTATION                                                        #
# =========================================================================== #

def test_i_am_always_in_slot_zero():
    state = mid_game(num_players=4)
    for player in state.players:
        slots = E.player_slots(state, player)
        assert slots[player] == 0
        assert sorted(slots.values()) == [0, 1, 2, 3]


def test_opponents_follow_me_in_turn_order():
    state = fresh(num_players=4, player_order=[3, 1, 4, 2], seed=1)
    assert E.player_slots(state, 3) == {3: 0, 1: 1, 4: 2, 2: 3}
    assert E.player_slots(state, 4) == {4: 0, 2: 1, 3: 2, 1: 3}


def test_the_is_me_flag_marks_slot_zero_and_nothing_else():
    state = mid_game(num_players=4)
    rows = E.block(E.encode(state, 2), "players")
    # feature 0 is "in the game", feature 1 is "is me"
    assert [row[1] for row in rows] == [1.0, 0.0, 0.0, 0.0]
    assert [row[0] for row in rows] == [1.0, 1.0, 1.0, 1.0]


def test_absent_players_leave_their_slot_empty():
    state = mid_game(num_players=2)
    rows = E.block(E.encode(state, 1), "players")
    assert [row[0] for row in rows] == [1.0, 1.0, 0.0, 0.0]
    assert all(value == 0.0 for row in rows[2:] for value in row)


def test_the_same_position_encodes_identically_whichever_number_holds_it():
    """The point of rotation: one network plays every seat."""
    board = fresh(seed=7).board
    observations = []

    for me, them in ((1, 2), (2, 1)):
        state = GameState(num_players=2, seed=7, board=board, player_order=[me, them])
        put_building(state, me, 20, Piece.CITY)
        put_building(state, them, 31, Piece.SETTLEMENT)
        put_road(state, me, 30)
        put_road(state, them, 40)
        give(state, me, wood=3, ore=2)
        give(state, them, sheep=4)
        state.dev_cards[me][DevCard.KNIGHT] = 2
        state.dev_cards[them][DevCard.MONOPOLY] = 1
        state.knights_played[me] = 3
        rules.update_awards(state)
        in_build_phase(state, me)
        observations.append(E.encode(state, me))

    assert observations[0] == observations[1]


# =========================================================================== #
# THE BOARD BLOCK                                                             #
# =========================================================================== #

def test_each_tile_reports_one_resource_and_one_number():
    state = mid_game()
    for index, row in enumerate(E.block(E.encode(state, 1), "tiles")):
        tile = index + 1
        resource_hot = row[:NUM_RESOURCES + 1]
        assert sum(resource_hot) == 1.0
        expected = NUM_RESOURCES if state.board.resource_at(tile) is None \
            else int(state.board.resource_at(tile))
        assert resource_hot[expected] == 1.0

        number_hot = row[NUM_RESOURCES + 1:NUM_RESOURCES + 1 + len(E.ROLLS)]
        assert sum(number_hot) == 1.0
        assert number_hot[E.ROLLS.index(state.board.number_at(tile))] == 1.0


def test_the_desert_shows_no_production_odds_despite_carrying_a_seven():
    state = mid_game()
    rows = E.block(E.encode(state, 1), "tiles")
    odds_at = NUM_RESOURCES + 1 + len(E.ROLLS)
    desert = rows[state.board.desert_tile - 1]
    assert desert[odds_at] == 0.0

    other = next(t for t in range(1, T.NUM_TILES + 1)
                 if state.board.resource_at(t) is not None)
    assert rows[other - 1][odds_at] > 0.0


def test_exactly_one_tile_carries_the_robber():
    state = mid_game()
    rows = E.block(E.encode(state, 1), "tiles")
    robber_at = E.TILE_FEATURES - 1
    flags = [row[robber_at] for row in rows]
    assert sum(flags) == 1.0
    assert flags[state.robber_tile - 1] == 1.0


def test_a_vertex_reports_its_owner_relative_to_me():
    state = mid_game()
    put_building(state, 2, 47, Piece.SETTLEMENT)
    rows = E.block(E.encode(state, 1), "vertices")

    for vertex in range(1, T.NUM_VERTICES + 1):
        owner_hot = rows[vertex - 1][:MAX_PLAYERS + 1]
        assert sum(owner_hot) == 1.0
        owner = state.vertex_owner[vertex]
        expected = 0 if owner == NO_OWNER else 1 + E.player_slots(state, 1)[owner]
        assert owner_hot[expected] == 1.0


def test_a_city_is_flagged_and_a_settlement_is_not():
    state = mid_game()
    rows = E.block(E.encode(state, 1), "vertices")
    city_at = MAX_PLAYERS + 1
    for vertex in range(1, T.NUM_VERTICES + 1):
        expected = 1.0 if state.vertex_piece[vertex] is Piece.CITY else 0.0
        assert rows[vertex - 1][city_at] == expected


def test_a_vertex_reports_its_harbour():
    state = mid_game()
    rows = E.block(E.encode(state, 1), "vertices")
    harbour_at = MAX_PLAYERS + 2

    for vertex in range(1, T.NUM_VERTICES + 1):
        hot = rows[vertex - 1][harbour_at:harbour_at + E.HARBOUR_KINDS]
        harbours = state.board.harbours_at(vertex)
        if not harbours:
            assert hot[0] == 1.0 and sum(hot) == 1.0
        else:
            assert hot[0] == 0.0
            for harbour in harbours:
                slot = 1 if harbour is GENERIC_HARBOUR else 2 + int(harbour)
                assert hot[slot] == 1.0


def test_pip_potential_sums_the_odds_of_the_adjacent_tiles():
    state = mid_game()
    rows = E.block(E.encode(state, 1), "vertices")
    pips_at = MAX_PLAYERS + 2 + E.HARBOUR_KINDS

    for vertex in range(1, T.NUM_VERTICES + 1):
        expected = sum(
            (6 - abs(7 - state.board.number_at(tile))) / 36
            for tile in T.VERTEX_TILES[vertex]
            if state.board.resource_at(tile) is not None
        )
        assert rows[vertex - 1][pips_at] == pytest.approx(expected)


def test_buildability_flags_agree_with_the_rules():
    state = mid_game()
    vertex_rows = E.block(E.encode(state, 1), "vertices")
    distance_at = E.VERTEX_FEATURES - 2
    connected_at = E.VERTEX_FEATURES - 1
    for vertex in range(1, T.NUM_VERTICES + 1):
        row = vertex_rows[vertex - 1]
        assert row[distance_at] == float(rules.respects_distance_rule(state, vertex))
        assert row[connected_at] == float(rules.touches_own_road(state, 1, vertex))

    road_rows = E.block(E.encode(state, 1), "roads")
    reach_at = E.ROAD_FEATURES - 1
    for road in range(1, T.NUM_ROADS + 1):
        expected = (state.edge_owner[road] == NO_OWNER
                    and rules.is_road_connected(state, 1, road))
        assert road_rows[road - 1][reach_at] == float(expected)


# =========================================================================== #
# GLOBAL BLOCK                                                                #
# =========================================================================== #

def test_the_phase_is_one_hot():
    state = fresh(seed=1)
    for phase in Phase:
        state.phase = phase
        if phase is Phase.GAME_OVER:
            state.winner = 1
        if phase is Phase.DISCARD:
            state.pending_discards = [1]
            state.discards_owed[1] = 1
        values = E.encode(state, 1)[E.LAYOUT["global"]]
        hot = values[:len(Phase)]
        assert sum(hot) == 1.0
        assert hot[int(phase)] == 1.0


def test_no_roll_yet_has_its_own_slot():
    state = fresh(seed=1)
    values = E.encode(state, 1)[E.LAYOUT["global"]]
    roll_hot = values[len(Phase):len(Phase) + len(E.ROLLS) + 1]
    assert roll_hot[-1] == 1.0, "the 'not rolled yet' slot"

    state.last_roll = 8
    values = E.encode(state, 1)[E.LAYOUT["global"]]
    roll_hot = values[len(Phase):len(Phase) + len(E.ROLLS) + 1]
    assert roll_hot[E.ROLLS.index(8)] == 1.0
    assert roll_hot[-1] == 0.0


def test_the_ruleset_is_part_of_the_observation():
    """One network should be able to play either format, so it has to see which."""
    base = E.encode(mid_game(ruleset=BASE_GAME), 1)
    ranked = E.encode(mid_game(ruleset=RANKED_1V1), 1)
    assert base[E.LAYOUT["global"]] != ranked[E.LAYOUT["global"]]


def test_whose_decision_it_is_is_visible():
    """During a discard the decision can belong to an opponent, so an agent must be able
    to tell that it is not being asked to move."""
    state = mid_game()
    give(state, 1)                                            # player 1 holds nothing
    give(state, 2, wood=state.ruleset.hand_limit + 4)          # only player 2 is over
    state.rolled_this_turn = True
    rules.begin_robber(state)
    assert state.phase is Phase.DISCARD
    assert state.current_player == 2, "only player 2 should owe a discard"

    flag_at = (len(Phase) + len(E.ROLLS) + 1 + 1 + NUM_RESOURCES + 1 + 1 + 1 + 1)
    assert E.encode(state, 2)[E.LAYOUT["global"]][flag_at] == 1.0
    assert E.encode(state, 1)[E.LAYOUT["global"]][flag_at] == 0.0


# =========================================================================== #
# DETERMINISM AND COST                                                        #
# =========================================================================== #

def test_encoding_is_pure():
    state = mid_game()
    before = state.clone()
    first = E.encode(state, 1)
    assert state == before, "encoding mutated the state"
    assert E.encode(state, 1) == first


@pytest.mark.slow
def test_encoding_works_at_every_point_of_a_real_game():
    def check(state):
        for player in state.players:
            observation = E.encode(state, player)
            assert len(observation) == E.SIZE
            assert all(0.0 <= v <= 1.0 for v in observation)

    for seed in range(4):
        play_random_game(seed=seed, num_players=3, max_actions=900, on_step=check)


def test_encoding_is_fast_enough_to_sit_in_the_loop():
    import timeit
    state = mid_game()
    elapsed = timeit.timeit(lambda: E.encode(state, 1), number=2000) / 2000
    assert elapsed < 5e-3, f"{elapsed * 1e6:.0f} us per observation is too slow"

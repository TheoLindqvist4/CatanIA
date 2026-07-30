"""Development cards, Largest Army, and the Longest Road award."""

import collections
import random

import pytest

import catan.topology as T
from catan import rules
from catan.actions import (
    ActionType,
    build_road,
    buy_dev_card,
    end_turn,
    move_robber,
    play_knight,
    play_monopoly,
    play_road_building,
    play_year_of_plenty,
)
from catan.dev_cards import (
    AWARD_VICTORY_POINTS,
    DECK_COUNTS,
    DECK_SIZE,
    LARGEST_ARMY_MINIMUM,
    LONGEST_ROAD_MINIMUM,
    PLAYABLE,
    ROAD_BUILDING_ROADS,
    DevCard,
    build_deck,
)
from catan.resources import DEV_CARD_COST, Resource, total
from catan.rules import IllegalAction
from catan.rulesets import BASE_GAME
from catan.state import Phase, Piece
from helpers import (
    complete_setup,
    enough_for_everything,
    fresh,
    give,
    in_build_phase,
    put_building,
    put_road,
    put_roads,
    roll_to_build,
)


def hand_card(state, player, card, count=1, fresh_card=False):
    """Put a development card in a player's hand.

    ``fresh_card=True`` marks it as bought this turn, so it is not yet playable.
    """
    state.dev_cards[player][card] += count
    if fresh_card:
        state.dev_cards_new[player][card] += count


# =========================================================================== #
# THE DECK                                                                    #
# =========================================================================== #

def test_the_deck_is_the_standard_twenty_five_cards():
    assert DECK_SIZE == 25
    assert DECK_COUNTS == {
        DevCard.KNIGHT: 14,
        DevCard.VICTORY_POINT: 5,
        DevCard.ROAD_BUILDING: 2,
        DevCard.YEAR_OF_PLENTY: 2,
        DevCard.MONOPOLY: 2,
    }
    deck = build_deck(random.Random(0))
    assert collections.Counter(deck) == DECK_COUNTS


def test_a_victory_point_card_is_not_playable():
    assert DevCard.VICTORY_POINT not in PLAYABLE
    assert set(PLAYABLE) == set(DevCard) - {DevCard.VICTORY_POINT}


def test_the_deck_order_is_reproducible_and_varies_by_seed():
    assert build_deck(random.Random(4)) == build_deck(random.Random(4))
    orders = {tuple(build_deck(random.Random(s))) for s in range(20)}
    assert len(orders) > 1


def test_the_deck_is_shuffled_once_not_drawn_at_random():
    """A clone must replay the same purchases — the deck is hidden information, not a
    fresh die roll."""
    state = fresh(seed=3)
    a, b = state.clone(), state.clone()
    assert a.dev_deck == b.dev_deck
    assert [a.dev_deck.pop() for _ in range(5)] == [b.dev_deck.pop() for _ in range(5)]


# =========================================================================== #
# BUYING                                                                      #
# =========================================================================== #

def test_a_card_costs_sheep_wheat_and_ore():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    give(state, 1, sheep=1, wheat=1, ore=1, wood=2)

    rules.apply(state, buy_dev_card())
    assert state.hands[1] == [2, 0, 0, 0, 0]
    assert sum(state.dev_cards[1]) == 1


def test_buying_draws_from_the_top_of_the_deck():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    enough_for_everything(state, 1)
    expected = state.dev_deck[-1]
    before = len(state.dev_deck)

    rules.apply(state, buy_dev_card())
    assert len(state.dev_deck) == before - 1
    assert state.dev_cards[1][expected] == 1


def test_buying_returns_the_payment_to_the_bank():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    give(state, 1, sheep=1, wheat=1, ore=1)
    before = list(state.bank)
    rules.apply(state, buy_dev_card())
    for resource in (Resource.SHEEP, Resource.WHEAT, Resource.ORE):
        assert state.bank[resource] == before[resource] + 1


def test_you_cannot_buy_without_the_resources():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    give(state, 1, sheep=1, wheat=1)  # no ore
    assert buy_dev_card() not in rules.legal_actions(state)
    with pytest.raises(IllegalAction):
        rules.apply(state, buy_dev_card())


def test_you_cannot_buy_from_an_empty_deck():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    enough_for_everything(state, 1, times=50)
    state.dev_deck = []
    assert buy_dev_card() not in rules.legal_actions(state)
    with pytest.raises(IllegalAction):
        rules.apply(state, buy_dev_card())


def test_the_whole_deck_can_be_bought_out_and_then_no_more():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    bought = 0
    while state.dev_deck:
        enough_for_everything(state, 1, times=50)
        rules.apply(state, buy_dev_card())
        bought += 1
        if state.phase is Phase.GAME_OVER:  # a VP card may end it
            break
    assert bought <= DECK_SIZE
    if state.phase is not Phase.GAME_OVER:
        assert bought == DECK_SIZE
        assert buy_dev_card() not in rules.legal_actions(state)


# =========================================================================== #
# PLAY TIMING                                                                 #
# =========================================================================== #

def test_a_card_bought_this_turn_cannot_be_played():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    enough_for_everything(state, 1, times=50)

    while sum(state.dev_cards[1]) == 0 or not state.dev_cards_new[1][DevCard.KNIGHT]:
        if not state.dev_deck:
            pytest.skip("no knight drawn")
        rules.apply(state, buy_dev_card())

    assert state.dev_cards[1][DevCard.KNIGHT] >= 1
    assert play_knight() not in rules.legal_actions(state)
    with pytest.raises(IllegalAction):
        rules.apply(state, play_knight())


def test_a_card_becomes_playable_after_the_turn_ends():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    hand_card(state, 1, DevCard.KNIGHT, fresh_card=True)
    assert play_knight() not in rules.legal_actions(state)

    rules.apply(state, end_turn())
    assert state.dev_cards_new[1] == [0, 0, 0, 0, 0]
    in_build_phase(state, 1)
    assert play_knight() in rules.legal_actions(state)


def test_only_one_card_per_turn():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    hand_card(state, 1, DevCard.MONOPOLY, count=2)

    rules.apply(state, play_monopoly(Resource.WOOD))
    assert state.dev_card_played_this_turn is True
    assert state.dev_cards[1][DevCard.MONOPOLY] == 1

    assert not [a for a in rules.legal_actions(state)
                if a.type in {ActionType.PLAY_MONOPOLY, ActionType.PLAY_KNIGHT}]
    with pytest.raises(IllegalAction):
        rules.apply(state, play_monopoly(Resource.ORE))


def test_the_one_card_limit_resets_next_turn():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    hand_card(state, 1, DevCard.MONOPOLY, count=2)
    rules.apply(state, play_monopoly(Resource.WOOD))
    rules.apply(state, end_turn())
    assert state.dev_card_played_this_turn is False


def test_a_card_may_be_played_before_rolling():
    """Most usefully a Knight, to block a tile before it produces."""
    state = fresh(seed=1)
    complete_setup(state)
    assert state.phase is Phase.ROLL
    hand_card(state, state.current_player, DevCard.KNIGHT)

    assert play_knight() in rules.legal_actions(state)
    rules.apply(state, play_knight())
    assert state.phase is Phase.MOVE_ROBBER


def test_after_a_pre_roll_knight_the_player_still_has_a_roll_coming():
    state = fresh(seed=1)
    complete_setup(state)
    player = state.current_player
    hand_card(state, player, DevCard.KNIGHT)

    rules.apply(state, play_knight())
    target = rules.legal_actions(state)[0]
    rules.apply(state, target)

    assert state.phase is Phase.ROLL, "the roll has not happened yet"
    assert state.rolled_this_turn is False
    rules.roll_dice(state)
    assert state.phase in (Phase.BUILD, Phase.DISCARD, Phase.MOVE_ROBBER)


def test_a_knight_played_after_rolling_returns_to_build():
    state = fresh(seed=1)
    complete_setup(state)
    roll_to_build(state)
    player = state.current_player
    hand_card(state, player, DevCard.KNIGHT)

    rules.apply(state, play_knight())
    rules.apply(state, rules.legal_actions(state)[0])
    assert state.phase is Phase.BUILD


def test_only_a_dev_card_may_be_played_before_rolling():
    state = fresh(seed=1)
    complete_setup(state)
    enough_for_everything(state, state.current_player)
    with pytest.raises(IllegalAction):
        rules.apply(state, end_turn())


def test_you_cannot_play_a_card_you_do_not_hold():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    for action in (play_knight(), play_road_building(), play_monopoly(0),
                   play_year_of_plenty(0, 1)):
        assert action not in rules.legal_actions(state)
        with pytest.raises(IllegalAction):
            rules.apply(state, action)


# =========================================================================== #
# KNIGHT                                                                      #
# =========================================================================== #

def test_a_knight_moves_the_robber_and_counts_toward_the_army():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    hand_card(state, 1, DevCard.KNIGHT)
    before = state.robber_tile

    rules.apply(state, play_knight())
    assert state.knights_played[1] == 1
    assert state.dev_cards[1][DevCard.KNIGHT] == 0
    assert state.phase is Phase.MOVE_ROBBER

    rules.apply(state, rules.legal_actions(state)[0])
    assert state.robber_tile != before


def test_a_knight_can_steal():
    state = fresh(seed=1)
    complete_setup(state)
    in_build_phase(state, 1)
    tile = next(t for t in range(1, T.NUM_TILES + 1) if t != state.robber_tile)
    vertex = next(v for v in T.TILE_VERTICES[tile]
                  if rules.respects_distance_rule(state, v))
    put_building(state, 2, vertex)
    give(state, 2, wood=3)
    give(state, 1)
    hand_card(state, 1, DevCard.KNIGHT)

    rules.apply(state, play_knight())
    rules.apply(state, move_robber(tile, 2))
    assert total(state.hands[1]) == 1
    assert total(state.hands[2]) == 2


def test_a_knight_does_not_trigger_discards():
    """Only a 7 makes people discard."""
    state = fresh(seed=1)
    in_build_phase(state, 1)
    give(state, 2, wood=12)
    hand_card(state, 1, DevCard.KNIGHT)

    rules.apply(state, play_knight())
    assert state.phase is Phase.MOVE_ROBBER
    assert state.pending_discards == []
    assert total(state.hands[2]) == 12


# =========================================================================== #
# ROAD BUILDING                                                               #
# =========================================================================== #

def test_road_building_grants_two_free_roads():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    put_building(state, 1, 20)
    give(state, 1)  # no resources at all
    hand_card(state, 1, DevCard.ROAD_BUILDING)

    rules.apply(state, play_road_building())
    assert state.free_roads == ROAD_BUILDING_ROADS

    for _ in range(ROAD_BUILDING_ROADS):
        road = next(a for a in rules.legal_actions(state)
                    if a.type is ActionType.BUILD_ROAD)
        rules.apply(state, road)
    assert len(state.roads_of(1)) == 2
    assert state.hands[1] == [0, 0, 0, 0, 0], "the roads were free"
    assert state.free_roads == 0


def test_once_the_free_roads_are_used_roads_cost_again():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    put_building(state, 1, 20)
    give(state, 1)
    hand_card(state, 1, DevCard.ROAD_BUILDING)
    rules.apply(state, play_road_building())

    for _ in range(ROAD_BUILDING_ROADS):
        rules.apply(state, next(a for a in rules.legal_actions(state)
                                if a.type is ActionType.BUILD_ROAD))

    assert not [a for a in rules.legal_actions(state)
                if a.type is ActionType.BUILD_ROAD], "broke with an empty hand"


def test_unused_free_roads_lapse_at_the_end_of_the_turn():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    put_building(state, 1, 20)
    hand_card(state, 1, DevCard.ROAD_BUILDING)
    rules.apply(state, play_road_building())
    assert state.free_roads == 2

    rules.apply(state, end_turn())
    assert state.free_roads == 0


def test_road_building_needs_somewhere_to_build():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    hand_card(state, 1, DevCard.ROAD_BUILDING)
    # no buildings and no roads: nothing to connect to
    assert play_road_building() not in rules.legal_actions(state)
    with pytest.raises(IllegalAction):
        rules.apply(state, play_road_building())


def test_road_building_needs_a_road_piece_left():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    put_building(state, 1, 20)
    hand_card(state, 1, DevCard.ROAD_BUILDING)
    state.roads_left[1] = 0
    assert play_road_building() not in rules.legal_actions(state)


def test_probing_road_building_legality_leaves_no_trace():
    """It checks legality by temporarily waiving the cost; that must not leak."""
    state = fresh(seed=1)
    in_build_phase(state, 1)
    put_building(state, 1, 20)
    hand_card(state, 1, DevCard.ROAD_BUILDING)

    rules.can_play_road_building(state, 1)
    assert state.free_roads == 0


# =========================================================================== #
# YEAR OF PLENTY                                                              #
# =========================================================================== #

def test_year_of_plenty_takes_two_resources_from_the_bank():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    hand_card(state, 1, DevCard.YEAR_OF_PLENTY)
    before = list(state.bank)

    rules.apply(state, play_year_of_plenty(Resource.ORE, Resource.WHEAT))
    assert state.hands[1][Resource.ORE] == 1
    assert state.hands[1][Resource.WHEAT] == 1
    assert state.bank[Resource.ORE] == before[Resource.ORE] - 1
    assert state.bank[Resource.WHEAT] == before[Resource.WHEAT] - 1


def test_year_of_plenty_may_take_the_same_resource_twice():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    hand_card(state, 1, DevCard.YEAR_OF_PLENTY)
    rules.apply(state, play_year_of_plenty(Resource.ORE, Resource.ORE))
    assert state.hands[1][Resource.ORE] == 2


def test_taking_two_of_the_same_needs_two_in_the_bank():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    hand_card(state, 1, DevCard.YEAR_OF_PLENTY)
    state.bank[Resource.ORE] = 1

    assert play_year_of_plenty(Resource.ORE, Resource.ORE) not in \
        rules.legal_actions(state)
    with pytest.raises(IllegalAction):
        rules.apply(state, play_year_of_plenty(Resource.ORE, Resource.ORE))
    # one of each is still fine
    assert play_year_of_plenty(Resource.WHEAT, Resource.ORE) in rules.legal_actions(state)


def test_year_of_plenty_cannot_take_what_the_bank_lacks():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    hand_card(state, 1, DevCard.YEAR_OF_PLENTY)
    state.bank[Resource.ORE] = 0
    with pytest.raises(IllegalAction):
        rules.apply(state, play_year_of_plenty(Resource.ORE, Resource.WHEAT))


def test_year_of_plenty_offers_each_pair_once():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    hand_card(state, 1, DevCard.YEAR_OF_PLENTY)
    offered = [a for a in rules.legal_actions(state)
               if a.type is ActionType.PLAY_YEAR_OF_PLENTY]
    pairs = {frozenset((a.position, a.extra)) for a in offered}
    assert len(offered) == len(pairs) == 15, "5 doubles + 10 distinct pairs"


# =========================================================================== #
# MONOPOLY                                                                    #
# =========================================================================== #

def test_monopoly_takes_every_opponents_cards_of_one_resource():
    state = fresh(num_players=3, seed=1)
    in_build_phase(state, 1)
    hand_card(state, 1, DevCard.MONOPOLY)
    give(state, 1, ore=1)
    give(state, 2, ore=3, wood=2)
    give(state, 3, ore=4)

    rules.apply(state, play_monopoly(Resource.ORE))
    assert state.hands[1][Resource.ORE] == 8
    assert state.hands[2][Resource.ORE] == 0
    assert state.hands[3][Resource.ORE] == 0
    assert state.hands[2][Resource.WOOD] == 2, "other resources untouched"


def test_monopoly_on_a_resource_nobody_holds_is_legal_but_empty():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    hand_card(state, 1, DevCard.MONOPOLY)
    give(state, 2, wood=3)

    assert play_monopoly(Resource.ORE) in rules.legal_actions(state)
    rules.apply(state, play_monopoly(Resource.ORE))
    assert state.hands[1][Resource.ORE] == 0


def test_monopoly_does_not_touch_the_bank():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    hand_card(state, 1, DevCard.MONOPOLY)
    give(state, 2, ore=4)
    before = list(state.bank)
    rules.apply(state, play_monopoly(Resource.ORE))
    assert state.bank == before


# =========================================================================== #
# VICTORY POINT CARDS                                                         #
# =========================================================================== #

def test_a_victory_point_card_counts_while_held():
    state = fresh(seed=1)
    assert rules.victory_points(state, 1) == 0
    hand_card(state, 1, DevCard.VICTORY_POINT, count=2)
    assert rules.victory_points(state, 1) == 2


def test_a_victory_point_card_is_hidden_from_opponents():
    state = fresh(seed=1)
    put_building(state, 1, 20, Piece.CITY)
    hand_card(state, 1, DevCard.VICTORY_POINT, count=3)
    assert rules.victory_points(state, 1) == 5
    assert rules.public_victory_points(state, 1) == 2


def test_a_victory_point_card_is_never_offered_as_a_play():
    state = fresh(seed=1)
    in_build_phase(state, 1)
    hand_card(state, 1, DevCard.VICTORY_POINT, count=5)
    assert rules.playable_dev_cards(state, 1)[DevCard.VICTORY_POINT] == 0
    assert rules.dev_card_actions(state, 1) == []


def test_buying_a_victory_point_card_can_win_the_game():
    # base game: nine points already, so the tenth wins
    state = fresh(seed=1, ruleset=BASE_GAME)
    in_build_phase(state, 1)
    enough_for_everything(state, 1, times=50)
    hand_card(state, 1, DevCard.VICTORY_POINT, count=9)
    state.dev_deck = [DevCard.VICTORY_POINT]

    assert rules.victory_points(state, 1) == 9
    rules.apply(state, buy_dev_card())
    assert rules.victory_points(state, 1) == 10
    assert state.winner == 1
    assert state.phase is Phase.GAME_OVER


# =========================================================================== #
# LARGEST ARMY                                                                #
# =========================================================================== #

def test_largest_army_needs_three_knights():
    state = fresh(seed=1)
    for knights in (0, 1, 2):
        state.knights_played[1] = knights
        rules.update_awards(state)
        assert state.largest_army_holder is None, f"{knights} knights is not enough"

    state.knights_played[1] = LARGEST_ARMY_MINIMUM
    rules.update_awards(state)
    assert state.largest_army_holder == 1


def test_largest_army_is_worth_two_points():
    state = fresh(seed=1)
    state.knights_played[1] = 3
    rules.update_awards(state)
    assert rules.victory_points(state, 1) == AWARD_VICTORY_POINTS


def test_largest_army_is_kept_until_strictly_beaten():
    state = fresh(seed=1)
    state.knights_played[1] = 3
    rules.update_awards(state)
    assert state.largest_army_holder == 1

    state.knights_played[2] = 3  # a tie does not take it
    rules.update_awards(state)
    assert state.largest_army_holder == 1

    state.knights_played[2] = 4  # beating it does
    rules.update_awards(state)
    assert state.largest_army_holder == 2


def test_playing_knights_earns_the_army_through_the_normal_flow():
    state = fresh(seed=1)
    complete_setup(state)
    for _ in range(LARGEST_ARMY_MINIMUM):
        in_build_phase(state, 1)
        state.dev_card_played_this_turn = False
        hand_card(state, 1, DevCard.KNIGHT)
        rules.apply(state, play_knight())
        rules.apply(state, rules.legal_actions(state)[0])  # move the robber
    assert state.knights_played[1] == 3
    assert state.largest_army_holder == 1


# =========================================================================== #
# LONGEST ROAD AWARD                                                          #
# =========================================================================== #

def test_longest_road_needs_five_segments():
    state = fresh(seed=1)
    put_roads(state, 1, [1, 2, 3, 4])          # a chain of 4
    rules.update_awards(state)
    assert rules.longest_road_length(state, 1) == 4
    assert state.longest_road_holder is None

    put_road(state, 1, 7)                       # extends to 5
    rules.update_awards(state)
    assert rules.longest_road_length(state, 1) == LONGEST_ROAD_MINIMUM
    assert state.longest_road_holder == 1


def test_longest_road_is_worth_two_points():
    state = fresh(seed=1)
    put_roads(state, 1, [1, 2, 3, 4, 7])
    rules.update_awards(state)
    assert rules.victory_points(state, 1) == AWARD_VICTORY_POINTS


def test_longest_road_is_kept_on_a_tie():
    state = fresh(seed=1)
    put_roads(state, 1, [1, 2, 3, 4, 7])
    rules.update_awards(state)
    assert state.longest_road_holder == 1

    put_roads(state, 2, [19, 24, 34, 40, 50])
    rules.update_awards(state)
    assert rules.longest_road_length(state, 2) == 5
    assert state.longest_road_holder == 1, "a tie does not take the card"


def test_longest_road_changes_hands_when_beaten():
    state = fresh(seed=1)
    put_roads(state, 1, [1, 2, 3, 4, 7])
    rules.update_awards(state)
    put_roads(state, 2, [19, 24, 34, 40, 50, 55])
    rules.update_awards(state)
    assert rules.longest_road_length(state, 2) == 6
    assert state.longest_road_holder == 2


def test_a_settlement_that_breaks_a_road_can_take_the_award_away():
    """Longest Road has to be rechecked after building, not just after a road."""
    state = fresh(seed=1)
    # player 1: chain 4-1-5-2-6 plus 7, six vertices, length 5
    put_roads(state, 1, [1, 2, 3, 4, 7])
    rules.update_awards(state)
    assert state.longest_road_holder == 1
    assert rules.victory_points(state, 1) == AWARD_VICTORY_POINTS

    # player 2 settles in the middle of it, splitting the chain
    in_build_phase(state, 2)
    enough_for_everything(state, 2)
    put_road(state, 2, 8)  # gives player 2 a road into vertex 5
    assert rules.can_build_settlement(state, 2, 5)
    rules.apply(state, rules.build_settlement(5))

    assert rules.longest_road_length(state, 1) < LONGEST_ROAD_MINIMUM
    assert state.longest_road_holder is None, "the award is lost, not kept"
    assert rules.victory_points(state, 1) == 0


def test_when_the_holder_falls_behind_a_tie_leaves_nobody_holding_it():
    state = fresh(seed=1)
    state.longest_road_holder = 1
    scores = {1: 4, 2: 6, 3: 6}
    assert rules._update_award(1, scores, LONGEST_ROAD_MINIMUM) is None


def test_the_sole_leader_takes_it_when_the_holder_falls_behind():
    scores = {1: 4, 2: 7, 3: 6}
    assert rules._update_award(1, scores, LONGEST_ROAD_MINIMUM) == 2


# =========================================================================== #
# BOTH AWARDS TOGETHER                                                        #
# =========================================================================== #

def test_a_player_can_hold_both_awards():
    state = fresh(seed=1)
    state.knights_played[1] = 3
    put_roads(state, 1, [1, 2, 3, 4, 7])
    rules.update_awards(state)
    assert state.largest_army_holder == 1
    assert state.longest_road_holder == 1
    assert rules.victory_points(state, 1) == 2 * AWARD_VICTORY_POINTS


def test_awards_plus_buildings_plus_cards_add_up():
    state = fresh(seed=1)
    put_building(state, 1, 20, Piece.CITY)     # 2
    put_building(state, 1, 31, Piece.CITY)     # 2
    state.knights_played[1] = 3                # 2
    put_roads(state, 1, [1, 2, 3, 4, 7])       # 2
    hand_card(state, 1, DevCard.VICTORY_POINT, count=2)   # 2
    rules.update_awards(state)
    assert rules.victory_points(state, 1) == 10

"""The ranked 1v1 ruleset: 15 points, hand limit 9, Friendly Robber, Balanced Dice."""

import collections
import random

import pytest

import catan.topology as T
from catan import dice, rules
from catan.actions import move_robber
from catan.board import ROBBER_ROLL
from catan.dev_cards import DevCard
from catan.resources import Resource, total
from catan.rules import IllegalAction
from catan.rulesets import ALL, BASE_GAME, DEFAULT, RANKED_1V1, RuleSet
from catan.state import HAND_LIMIT, VICTORY_POINTS_TO_WIN, GameState, Phase, Piece
from helpers import complete_setup, fresh, give, in_build_phase, put_building, roll_sequence


def at_robber(state, player=1):
    """Put the state in MOVE_ROBBER with ``player`` about to move it."""
    in_build_phase(state, player)
    state.rolled_this_turn = True
    state.phase = Phase.MOVE_ROBBER
    return state


# =========================================================================== #
# THE RULESETS THEMSELVES                                                     #
# =========================================================================== #

def test_the_base_game_matches_the_printed_rules():
    assert BASE_GAME.victory_points_to_win == VICTORY_POINTS_TO_WIN == 10
    assert BASE_GAME.hand_limit == HAND_LIMIT == 7
    assert BASE_GAME.friendly_robber is False
    assert BASE_GAME.balanced_dice is False


def test_ranked_1v1_matches_the_published_settings():
    """From Colonist.io's ranked 1v1: 15 points, hand limit 9, Friendly Robber on,
    Balanced Dice on. See docs/decisions/0013-ranked-1v1-ruleset.md."""
    assert RANKED_1V1.victory_points_to_win == 15
    assert RANKED_1V1.hand_limit == 9
    assert RANKED_1V1.friendly_robber is True
    assert RANKED_1V1.friendly_robber_threshold == 2
    assert RANKED_1V1.balanced_dice is True


def test_ranked_1v1_is_the_default():
    """It is the format this project targets."""
    assert DEFAULT is RANKED_1V1
    assert GameState(seed=0).ruleset is RANKED_1V1


def test_a_ruleset_is_immutable():
    with pytest.raises(AttributeError):
        RANKED_1V1.hand_limit = 3


def test_the_ruleset_travels_with_a_clone_and_counts_for_equality():
    state = fresh(seed=1, ruleset=BASE_GAME)
    assert state.clone().ruleset is BASE_GAME

    other = GameState(seed=1, ruleset=RANKED_1V1, board=state.board)
    assert state != other, "the same position under different rules is not the same game"


@pytest.mark.parametrize("ruleset", ALL, ids=lambda r: r.name)
def test_every_ruleset_plays_a_whole_game(ruleset):
    from helpers import play_random_game
    state = play_random_game(seed=3, max_actions=40_000, ruleset=ruleset)
    assert state.winner is not None, f"{ruleset.name} did not finish"
    assert (rules.victory_points(state, state.winner)
            >= ruleset.victory_points_to_win)


# =========================================================================== #
# 15 VICTORY POINTS                                                           #
# =========================================================================== #

def test_ten_points_does_not_win_ranked_1v1():
    state = fresh(seed=1, ruleset=RANKED_1V1)
    in_build_phase(state, 1)
    state.dev_cards[1][DevCard.VICTORY_POINT] = 10
    assert rules.victory_points(state, 1) == 10
    rules._check_for_winner(state, 1)
    assert state.winner is None
    assert state.phase is Phase.BUILD


def test_fifteen_points_wins_ranked_1v1():
    state = fresh(seed=1, ruleset=RANKED_1V1)
    in_build_phase(state, 1)
    state.dev_cards[1][DevCard.VICTORY_POINT] = 15
    rules._check_for_winner(state, 1)
    assert state.winner == 1
    assert state.phase is Phase.GAME_OVER


def test_fifteen_is_reachable_from_the_pieces_available():
    """4 cities + 5 settlements is 13, plus both awards is 17 — so 15 is reachable
    without needing Victory Point cards at all."""
    from catan.dev_cards import AWARD_VICTORY_POINTS
    from catan.state import MAX_CITIES, MAX_SETTLEMENTS
    building_max = MAX_CITIES * 2 + MAX_SETTLEMENTS
    assert building_max == 13
    assert building_max + 2 * AWARD_VICTORY_POINTS >= RANKED_1V1.victory_points_to_win


# =========================================================================== #
# HAND LIMIT 9                                                                #
# =========================================================================== #

@pytest.mark.parametrize("held,over", [(7, False), (8, False), (9, False), (10, True)])
def test_nine_cards_is_safe_in_ranked_1v1(held, over):
    state = fresh(seed=1, ruleset=RANKED_1V1)
    give(state, 1, wood=held)
    assert rules.must_discard(state, 1) is over


@pytest.mark.parametrize("held,over", [(7, False), (8, True)])
def test_eight_cards_already_costs_you_in_the_base_game(held, over):
    state = fresh(seed=1, ruleset=BASE_GAME)
    give(state, 1, wood=held)
    assert rules.must_discard(state, 1) is over


def test_ten_cards_in_ranked_1v1_loses_five():
    state = fresh(seed=1, ruleset=RANKED_1V1)
    complete_setup(state)
    give(state, 1, wood=10)
    give(state, 2, wood=1)
    state.rolled_this_turn = True
    rules.begin_robber(state)

    assert state.discards_owed[1] == 5
    while state.phase is Phase.DISCARD:
        rules.apply(state, rules.legal_actions(state)[0])
    assert total(state.hands[1]) == 5


# =========================================================================== #
# FRIENDLY ROBBER                                                             #
# =========================================================================== #

def test_nobody_is_protected_in_the_base_game():
    state = fresh(seed=1, ruleset=BASE_GAME)
    assert rules.is_robber_protected(state, 1) is False
    put_building(state, 1, 20)
    assert rules.is_robber_protected(state, 1) is False


@pytest.mark.parametrize("buildings,protected", [(0, True), (1, True), (2, True), (3, False)])
def test_protection_lasts_until_more_than_two_public_points(buildings, protected):
    state = fresh(seed=1, ruleset=RANKED_1V1)
    spots = _spaced(state, buildings)
    for vertex in spots:
        put_building(state, 2, vertex, Piece.SETTLEMENT)
    assert rules.public_victory_points(state, 2) == buildings
    assert rules.is_robber_protected(state, 2) is protected


def test_a_city_takes_you_over_the_threshold():
    state = fresh(seed=1, ruleset=RANKED_1V1)
    spots = _spaced(state, 2)
    put_building(state, 2, spots[0], Piece.CITY)      # 2 points
    assert rules.is_robber_protected(state, 2) is True
    put_building(state, 2, spots[1], Piece.SETTLEMENT)  # 3 points
    assert rules.is_robber_protected(state, 2) is False


def test_hidden_victory_point_cards_do_not_cost_you_protection():
    """The rule counts what you *openly* have."""
    state = fresh(seed=1, ruleset=RANKED_1V1)
    put_building(state, 2, _spaced(state, 1)[0])
    state.dev_cards[2][DevCard.VICTORY_POINT] = 5

    assert rules.victory_points(state, 2) == 6
    assert rules.public_victory_points(state, 2) == 1
    assert rules.is_robber_protected(state, 2) is True


def test_an_award_does_cost_you_protection():
    state = fresh(seed=1, ruleset=RANKED_1V1)
    put_building(state, 2, _spaced(state, 1)[0])
    state.knights_played[2] = 3
    rules.update_awards(state)
    assert rules.public_victory_points(state, 2) == 3
    assert rules.is_robber_protected(state, 2) is False


def test_a_protected_player_cannot_be_robbed():
    state, tile, vertex = _opponent_on_a_tile(RANKED_1V1)
    give(state, 2, wood=5)
    assert rules.is_robber_protected(state, 2) is True

    assert rules.victims_at(state, tile, 1) == ()
    assert move_robber(tile, 2) not in rules.legal_actions(state)
    with pytest.raises(IllegalAction):
        rules.apply(state, move_robber(tile, 2))


def test_a_protected_players_tile_cannot_even_be_blocked():
    """Friendly Robber stops placement, not just stealing."""
    state, tile, vertex = _opponent_on_a_tile(RANKED_1V1)
    assert tile not in rules.robber_destinations(state, 1)
    assert not [a for a in rules.legal_actions(state) if a.position == tile]
    with pytest.raises(IllegalAction):
        rules.apply(state, move_robber(tile, 0))


def test_once_unprotected_the_tile_opens_up_again():
    state, tile, vertex = _opponent_on_a_tile(RANKED_1V1)
    give(state, 2, wood=5)

    for extra in _spaced(state, 2, avoid=[vertex]):
        put_building(state, 2, extra, Piece.SETTLEMENT)
    assert rules.public_victory_points(state, 2) == 3
    assert rules.is_robber_protected(state, 2) is False

    assert tile in rules.robber_destinations(state, 1)
    assert move_robber(tile, 2) in rules.legal_actions(state)


def test_the_base_game_lets_you_block_a_one_point_player():
    state, tile, vertex = _opponent_on_a_tile(BASE_GAME)
    give(state, 2, wood=5)
    assert rules.public_victory_points(state, 2) == 1
    assert tile in rules.robber_destinations(state, 1)
    assert move_robber(tile, 2) in rules.legal_actions(state)


def test_your_own_buildings_never_block_you():
    """Protection is about opponents; you may always block yourself."""
    state = fresh(seed=1, ruleset=RANKED_1V1)
    complete_setup(state)
    at_robber(state, 1)
    own_tile = next(
        t for t in range(1, T.NUM_TILES + 1)
        if t != state.robber_tile
        and any(state.vertex_owner[v] == 1 for v in T.TILE_VERTICES[t])
    )
    assert rules.is_robber_protected(state, 1) is True  # 2 points at setup
    assert own_tile in rules.robber_destinations(state, 1)


def test_the_robber_always_has_somewhere_to_go():
    """A protected player holds at most 2 points, so at most 2 buildings, so at most 6
    of the 18 candidate tiles can be blocked."""
    for seed in range(20):
        state = fresh(seed=seed, ruleset=RANKED_1V1)
        complete_setup(state)
        at_robber(state, 1)
        assert len(rules.robber_destinations(state, 1)) >= 12
        assert rules.legal_actions(state)


# =========================================================================== #
# BALANCED DICE                                                               #
# =========================================================================== #

def test_the_deck_holds_every_two_dice_combination():
    assert dice.DECK_SIZE == 36
    assert len(set(dice.COMBINATIONS)) == 36
    assert collections.Counter(a + b for a, b in dice.COMBINATIONS) == {
        2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1
    }


def test_only_the_balanced_ruleset_has_a_deck():
    assert fresh(seed=1, ruleset=RANKED_1V1).dice_deck is not None
    assert fresh(seed=1, ruleset=BASE_GAME).dice_deck is None


def test_draws_within_one_deck_never_repeat_a_combination():
    """The whole point of a deck: no combination comes up twice before a reshuffle."""
    state = fresh(seed=5, ruleset=RANKED_1V1)
    drawn = [dice.draw_balanced(state)
             for _ in range(dice.DECK_SIZE - dice.RESHUFFLE_AT)]
    assert len(drawn) == 24
    assert len(set(drawn)) == 24, "a combination repeated inside one deck"


def test_the_deck_is_replaced_when_twelve_cards_remain():
    state = fresh(seed=5, ruleset=RANKED_1V1)
    for _ in range(dice.DECK_SIZE - dice.RESHUFFLE_AT - 1):
        dice.draw_balanced(state)
    assert len(state.dice_deck) == dice.RESHUFFLE_AT + 1

    dice.draw_balanced(state)
    assert len(state.dice_deck) == dice.DECK_SIZE, "should be a fresh deck"


def test_balanced_rolls_follow_the_triangular_distribution():
    state = fresh(seed=5, ruleset=RANKED_1V1)
    counts = collections.Counter(roll_sequence(state, 36_000))
    assert set(counts) == set(range(2, 13))
    for roll in range(2, 13):
        expected = (6 - abs(7 - roll)) / 36
        assert abs(counts[roll] / 36_000 - expected) < 0.01, f"roll {roll}"


def test_balanced_dice_tighten_the_distribution():
    """Over a window the deck cannot stray as far from expectation as free dice can.

    Compared as the worst per-outcome deviation across many short windows, averaged over
    seeds, so it measures the mechanism rather than one lucky sample.
    """
    def worst_deviation(ruleset, seed, rolls=240):
        state = fresh(seed=seed, ruleset=ruleset)
        counts = collections.Counter(roll_sequence(state, rolls))
        return max(
            abs(counts[roll] / rolls - (6 - abs(7 - roll)) / 36)
            for roll in range(2, 13)
        )

    seeds = range(30)
    balanced = sum(worst_deviation(RANKED_1V1, s) for s in seeds) / len(seeds)
    plain = sum(worst_deviation(BASE_GAME, s) for s in seeds) / len(seeds)
    assert balanced < plain, f"balanced {balanced:.4f} should beat plain {plain:.4f}"


def test_rolling_still_reproduces_from_a_seed():
    a = fresh(seed=8, ruleset=RANKED_1V1)
    b = fresh(seed=8, ruleset=RANKED_1V1)
    assert roll_sequence(a, 50) == roll_sequence(b, 50)


def test_the_undocumented_parts_of_balanced_dice_are_not_faked():
    """Colonist's published description also mentions a 30% reduction in repeating the
    same number twice and a 7-ownership balance, neither specified precisely enough to
    reproduce. They are deliberately not implemented — see
    docs/decisions/0013-ranked-1v1-ruleset.md. This test states that so the gap is not
    mistaken for a bug.
    """
    state = fresh(seed=5, ruleset=RANKED_1V1)
    rolls = roll_sequence(state, 6_000)
    repeats = sum(1 for a, b in zip(rolls, rolls[1:]) if a == b)
    # with no repeat-suppression, consecutive equal rolls happen at the natural rate of
    # sum(p^2) = 0.1127 for two dice; suppression would push this well below
    assert repeats / len(rolls) > 0.08, "unexpected repeat suppression appeared"


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #

def _spaced(state, count, avoid=()):
    """``count`` mutually non-adjacent free vertices."""
    if count == 0:
        return []
    chosen, out = list(avoid), []
    for vertex in range(1, T.NUM_VERTICES + 1):
        if state.vertex_owner[vertex] != 0:
            continue
        if any(v == vertex or v in T.VERTEX_NEIGHBOURS[vertex] for v in chosen):
            continue
        chosen.append(vertex)
        out.append(vertex)
        if len(out) == count:
            break
    assert len(out) == count
    return out


def _opponent_on_a_tile(ruleset):
    """MOVE_ROBBER, player 1 to move, player 2 holding one settlement on ``tile``."""
    state = fresh(seed=1, ruleset=ruleset)
    at_robber(state, 1)
    tile = next(t for t in range(1, T.NUM_TILES + 1) if t != state.robber_tile)
    vertex = T.TILE_VERTICES[tile][0]
    put_building(state, 2, vertex, Piece.SETTLEMENT)
    return state, tile, vertex

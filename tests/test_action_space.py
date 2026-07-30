"""The flat action space and the legality mask.

The load-bearing test here is
:func:`test_every_action_the_rules_ever_offer_is_in_the_space` — if the rules can produce an
action the space cannot express, the mask drops it silently and an agent can never choose it.
"""

import random

import pytest

import catan.topology as T
from catan import action_space as A
from catan import rules
from catan.actions import Action, ActionType
from catan.resources import NUM_RESOURCES
from catan.rulesets import ALL, BASE_GAME, RANKED_1V1
from catan.state import MAX_PLAYERS, Phase
from helpers import (
    complete_setup,
    enough_for_everything,
    fresh,
    in_build_phase,
    play_random_game,
    put_building,
)


# =========================================================================== #
# SHAPE                                                                       #
# =========================================================================== #

def test_the_space_is_a_fixed_size():
    assert A.NUM_ACTIONS == 324
    assert len(A.ACTIONS) == A.NUM_ACTIONS
    assert len(A.INDEX) == A.NUM_ACTIONS, "duplicate actions"


def test_block_sizes_are_what_the_geometry_and_rules_imply():
    assert A.COUNTS[ActionType.END_TURN] == 1
    assert A.COUNTS[ActionType.BUILD_ROAD] == T.NUM_ROADS == 72
    assert A.COUNTS[ActionType.BUILD_SETTLEMENT] == T.NUM_VERTICES == 54
    assert A.COUNTS[ActionType.BUILD_CITY] == T.NUM_VERTICES == 54
    # ordered pairs of distinct resources
    assert A.COUNTS[ActionType.TRADE_WITH_BANK] == NUM_RESOURCES * (NUM_RESOURCES - 1) == 20
    # every tile, times nobody-or-one-of-four
    assert A.COUNTS[ActionType.MOVE_ROBBER] == T.NUM_TILES * (MAX_PLAYERS + 1) == 95
    assert A.COUNTS[ActionType.DISCARD] == NUM_RESOURCES == 5
    assert A.COUNTS[ActionType.BUY_DEV_CARD] == 1
    assert A.COUNTS[ActionType.PLAY_KNIGHT] == 1
    assert A.COUNTS[ActionType.PLAY_ROAD_BUILDING] == 1
    # sorted pairs, doubles included: C(5,2) + 5
    assert A.COUNTS[ActionType.PLAY_YEAR_OF_PLENTY] == 15
    assert A.COUNTS[ActionType.PLAY_MONOPOLY] == NUM_RESOURCES == 5

    assert sum(A.COUNTS.values()) == A.NUM_ACTIONS


def test_every_action_type_has_a_block():
    assert set(A.SLICES) == set(ActionType)


def test_blocks_are_contiguous_and_cover_the_space_exactly_once():
    covered = 0
    for kind in ActionType:
        span = A.SLICES[kind]
        assert span.start == covered, f"a gap or overlap before {kind.name}"
        covered = span.stop
    assert covered == A.NUM_ACTIONS


def test_each_index_holds_the_type_its_block_claims():
    for kind, span in A.SLICES.items():
        for index in range(span.start, span.stop):
            assert A.ACTIONS[index].type == kind


def test_the_size_does_not_depend_on_the_player_count():
    """A network trained on 1v1 must have the same output shape as one on four players."""
    sizes = set()
    for count in (2, 3, 4):
        state = fresh(num_players=count, seed=1)
        sizes.add(len(A.legal_mask(state)))
    assert sizes == {A.NUM_ACTIONS}


# =========================================================================== #
# ENCODE / DECODE                                                             #
# =========================================================================== #

def test_every_action_round_trips():
    for index, action in enumerate(A.ACTIONS):
        assert A.encode(action) == index
        assert A.decode(index) == action


@pytest.mark.parametrize("index", [-1, 324, 999])
def test_decoding_an_out_of_range_index_raises(index):
    with pytest.raises(IndexError):
        A.decode(index)


def test_encoding_an_inexpressible_action_raises_with_a_useful_message():
    """A silent drop here would make an action unreachable for an agent."""
    with pytest.raises(KeyError) as caught:
        A.encode(Action(ActionType.BUILD_ROAD, 999))
    assert "action space" in str(caught.value)


def test_year_of_plenty_pairs_are_stored_sorted_so_one_move_is_one_index():
    span = A.SLICES[ActionType.PLAY_YEAR_OF_PLENTY]
    pairs = [(A.ACTIONS[i].position, A.ACTIONS[i].extra)
             for i in range(span.start, span.stop)]
    assert all(first <= second for first, second in pairs)
    assert len(set(pairs)) == len(pairs) == 15
    # the reversed form encodes to the same index, because the constructor sorts
    from catan.actions import play_year_of_plenty
    assert A.encode(play_year_of_plenty(4, 2)) == A.encode(play_year_of_plenty(2, 4))


def test_a_trade_never_offers_a_resource_for_itself():
    span = A.SLICES[ActionType.TRADE_WITH_BANK]
    for index in range(span.start, span.stop):
        action = A.ACTIONS[index]
        assert action.position != action.extra


def test_describe_is_readable():
    assert A.describe(0) == "000 END_TURN"
    assert A.describe(1) == "001 BUILD_ROAD(1)"


# =========================================================================== #
# THE MASK                                                                    #
# =========================================================================== #

def test_the_mask_matches_legal_actions_exactly():
    state = fresh(seed=1)
    complete_setup(state)
    in_build_phase(state, 1)
    enough_for_everything(state, 1)
    put_building(state, 1, 20)

    mask = A.legal_mask(state)
    expected = {A.encode(action) for action in rules.legal_actions(state)}

    assert len(mask) == A.NUM_ACTIONS
    assert {i for i, flag in enumerate(mask) if flag} == expected
    assert sum(mask) == len(expected)


def test_the_mask_is_a_bytearray_of_zeroes_and_ones():
    state = fresh(seed=1)
    mask = A.legal_mask(state)
    assert isinstance(mask, bytearray)
    assert set(mask) <= {0, 1}


def test_legal_indices_agrees_with_the_mask():
    state = fresh(seed=1)
    complete_setup(state)
    in_build_phase(state, 1)
    enough_for_everything(state, 1)

    mask = A.legal_mask(state)
    assert A.legal_indices(state) == [i for i, flag in enumerate(mask) if flag]


def test_setup_offers_exactly_the_settlement_block():
    state = fresh(seed=1)
    assert state.phase is Phase.SETUP_SETTLEMENT
    mask = A.legal_mask(state)
    span = A.SLICES[ActionType.BUILD_SETTLEMENT]
    assert sum(mask) == T.NUM_VERTICES
    assert all(mask[i] for i in range(span.start, span.stop))
    assert sum(mask[:span.start]) == 0
    assert sum(mask[span.stop:]) == 0


def test_the_mask_is_empty_when_the_driver_must_roll():
    state = fresh(seed=1)
    complete_setup(state)
    assert state.phase is Phase.ROLL
    assert sum(A.legal_mask(state)) == 0, "nothing to play, so roll_dice() is next"


def test_the_mask_is_empty_once_the_game_is_over():
    state = fresh(seed=1)
    state.phase = Phase.GAME_OVER
    state.winner = 1
    assert sum(A.legal_mask(state)) == 0


def test_a_masked_action_is_always_accepted_and_an_unmasked_one_never_is():
    """The mask is the contract an agent acts on; it must be exactly right."""
    rng = random.Random(0)
    state = fresh(seed=2)
    complete_setup(state, rng)

    for _ in range(60):
        if state.phase is Phase.GAME_OVER:
            break
        mask = A.legal_mask(state)
        if not any(mask):
            rules.roll_dice(state)
            continue

        for index, flag in enumerate(mask):
            action = A.decode(index)
            if flag:
                rules.apply(state.clone(), action)
            else:
                with pytest.raises(rules.IllegalAction):
                    rules.apply(state.clone(), action)

        chosen = rng.choice([i for i, flag in enumerate(mask) if flag])
        rules.apply(state, A.decode(chosen))


# =========================================================================== #
# THE LOAD-BEARING GUARANTEE                                                  #
# =========================================================================== #

@pytest.mark.slow
@pytest.mark.parametrize("ruleset", ALL, ids=lambda r: r.name)
@pytest.mark.parametrize("num_players", [2, 4])
def test_every_action_the_rules_ever_offer_is_in_the_space(ruleset, num_players):
    """If the rules can produce an action the space cannot express, the mask drops it
    silently and an agent can never choose it. Checked over whole games."""
    seen = set()

    def check(state):
        for action in rules.legal_actions(state):
            seen.add(A.encode(action))   # raises if inexpressible

    for seed in range(6):
        play_random_game(seed=seed, num_players=num_players, ruleset=ruleset,
                         max_actions=2500, on_step=check)

    assert len(seen) > 100, f"only {len(seen)} distinct actions exercised"


@pytest.mark.slow
def test_whole_games_can_be_played_through_indices_alone():
    """An agent only ever sees integers, so a game must be playable from them."""
    for seed in range(6):
        state = fresh(seed=seed)
        rng = random.Random(seed)
        for _ in range(4000):
            if state.phase is Phase.GAME_OVER:
                break
            mask = A.legal_mask(state)
            if not any(mask):
                rules.roll_dice(state)
                continue
            index = rng.choice([i for i, flag in enumerate(mask) if flag])
            rules.apply(state, A.decode(index))
        assert state.phase is Phase.GAME_OVER or state.turn_number > 0


# =========================================================================== #
# COST                                                                        #
# =========================================================================== #

def test_the_mask_costs_little_beyond_legal_actions():
    """Building the mask must not double the cost of the hot path."""
    import timeit
    state = fresh(seed=1)
    complete_setup(state)
    in_build_phase(state, 1)
    enough_for_everything(state, 1)
    put_building(state, 1, 20)

    legal = timeit.timeit(lambda: rules.legal_actions(state), number=2000) / 2000
    masked = timeit.timeit(lambda: A.legal_mask(state), number=2000) / 2000
    assert masked < legal * 1.5, (
        f"mask {masked * 1e6:.0f} us vs legal_actions {legal * 1e6:.0f} us"
    )

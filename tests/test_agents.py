"""Baseline agents, and the match harness that measures them."""

import pytest

from catan import action_space
from catan.actions import ActionType
from catan.agents import (
    GREEDY_PRIORITY,
    GreedyAgent,
    RandomAgent,
    play_game,
    play_match,
)
from catan.env import CatanEnv
from catan.rulesets import BASE_GAME, RANKED_1V1


AGENTS = [RandomAgent, GreedyAgent]


# =========================================================================== #
# THE INTERFACE                                                               #
# =========================================================================== #

@pytest.mark.parametrize("factory", AGENTS, ids=lambda f: f.__name__)
def test_an_agent_is_just_a_callable_of_observation_and_info(factory):
    env = CatanEnv()
    observation, info = env.reset(seed=1)
    action = factory(0)(observation, info)
    assert isinstance(action, int)
    assert info["mask"][action], "an agent must only ever return a legal index"


@pytest.mark.parametrize("factory", AGENTS, ids=lambda f: f.__name__)
def test_an_agent_only_ever_returns_legal_actions(factory):
    env = CatanEnv(num_players=3)
    agent = factory(0)
    observation, info = env.reset(seed=2)
    for _ in range(600):
        if info["done"]:
            break
        action = agent(observation, info)
        assert info["mask"][action], f"{factory.__name__} chose an illegal action"
        observation, _, _, _, info = env.step(action)


@pytest.mark.parametrize("factory", AGENTS, ids=lambda f: f.__name__)
def test_an_agent_is_reproducible_from_its_seed(factory):
    first = play_game({1: factory(7), 2: factory(8)}, seed=3)
    second = play_game({1: factory(7), 2: factory(8)}, seed=3)
    assert first["scores"] == second["scores"]
    assert first["winner"] == second["winner"]


# =========================================================================== #
# GREEDY                                                                      #
# =========================================================================== #

def test_the_greedy_priority_covers_every_action_type():
    """A type left out would be picked only as a last resort, silently."""
    assert set(GREEDY_PRIORITY) == set(ActionType)
    assert len(GREEDY_PRIORITY) == len(ActionType)


def test_greedy_prefers_building_to_ending_the_turn():
    env = CatanEnv()
    observation, info = env.reset(seed=1)
    agent = GreedyAgent(0)

    seen_build = False
    for _ in range(800):
        if info["done"]:
            break
        action = agent(observation, info)
        kind = action_space.decode(action).type
        offered = {action_space.decode(i).type for i in info["legal"]}
        if kind is ActionType.END_TURN:
            assert not (offered & {ActionType.BUILD_CITY, ActionType.BUILD_SETTLEMENT,
                                   ActionType.BUILD_ROAD, ActionType.BUY_DEV_CARD}), \
                "ended the turn with something better available"
        if kind in {ActionType.BUILD_CITY, ActionType.BUILD_SETTLEMENT}:
            seen_build = True
        observation, _, _, _, info = env.step(action)

    assert seen_build, "greedy never built anything"


def test_greedy_prefers_a_city_to_a_settlement():
    assert GREEDY_PRIORITY.index(ActionType.BUILD_CITY) < \
        GREEDY_PRIORITY.index(ActionType.BUILD_SETTLEMENT)


# =========================================================================== #
# THE HARNESS                                                                 #
# =========================================================================== #

def test_play_game_returns_a_finished_game():
    info = play_game({1: GreedyAgent(0), 2: GreedyAgent(1)}, seed=5)
    assert info["done"] is True
    assert info["winner"] in (1, 2, None)
    if info["winner"] is not None:
        assert info["scores"][info["winner"]] >= RANKED_1V1.victory_points_to_win


def test_play_game_infers_the_player_count_from_the_agents():
    info = play_game({1: RandomAgent(0), 2: RandomAgent(1), 3: RandomAgent(2)}, seed=1)
    assert set(info["scores"]) == {1, 2, 3}


def test_play_game_accepts_a_ruleset_and_a_turn_cap():
    info = play_game({1: RandomAgent(0), 2: RandomAgent(1)},
                     seed=1, ruleset=BASE_GAME, max_turns=8)
    assert info["winner"] is None, "should have been truncated"


def test_play_match_tallies_every_game():
    games = 12
    tally = play_match({1: GreedyAgent(0), 2: RandomAgent(0)}, games=games, seed=1)
    assert sum(tally.values()) == games
    assert set(tally) == {1, 2, "truncated"}


def test_play_match_swaps_seats_so_a_result_is_not_a_first_player_advantage():
    """Catan's first-player advantage is real and large, so a fixed-seat match measures
    the seat as much as the agent."""
    strong_first = play_match({1: GreedyAgent(0), 2: RandomAgent(0)}, games=20, seed=1)
    strong_second = play_match({1: RandomAgent(0), 2: GreedyAgent(0)}, games=20, seed=1)
    # whichever number the greedy agent is given, it should come out ahead
    assert strong_first[1] > strong_first[2]
    assert strong_second[2] > strong_second[1]


# =========================================================================== #
# THE BASELINES ARE ORDERED                                                   #
# =========================================================================== #

@pytest.mark.slow
def test_greedy_beats_random_convincingly():
    """The point of a baseline ladder: a learned policy has something to clear.

    Measured around 70% for greedy over 60 games; asserted well below that so an unlucky
    sample does not fail the suite.
    """
    tally = play_match({1: GreedyAgent(0), 2: RandomAgent(0)}, games=40, seed=100)
    decided = tally[1] + tally[2]
    assert decided >= 35, f"too many truncated games: {tally}"
    assert tally[1] / decided > 0.6, f"greedy should dominate random: {tally}"


@pytest.mark.slow
def test_a_mirror_match_is_near_even():
    """A sanity check on the harness itself: identical agents should split, so a lopsided
    result would mean the seat swapping or the tallying is wrong."""
    tally = play_match({1: RandomAgent(1), 2: RandomAgent(2)}, games=40, seed=200)
    decided = tally[1] + tally[2]
    assert decided >= 35
    assert 0.25 < tally[1] / decided < 0.75, f"suspiciously lopsided: {tally}"

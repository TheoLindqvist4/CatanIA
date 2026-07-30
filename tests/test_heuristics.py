"""Positional judgement, and the agent built on it.

Two things are being checked. First that the evaluation functions mean what they claim —
marginal value really does diminish, a city really is worth double. Second, and more
important, that :class:`HeuristicAgent` plays on public information *only*: the leak test at
the bottom rewrites the opponent's hidden cards and demands the same move back.
"""

import pytest

from catan import action_space, heuristics
from catan.agents import DIFFICULTY, GreedyAgent, HeuristicAgent, RandomAgent, play_match
from catan.board import GENERIC_HARBOUR
from catan.dev_cards import DevCard
from catan.env import CatanEnv
from catan.resources import NUM_RESOURCES, Resource, total
from catan.state import NO_OWNER, Piece
from catan.topology import (
    NUM_TILES,
    NUM_VERTICES,
    TILE_VERTICES,
    VERTEX_NEIGHBOURS,
    VERTEX_TILES,
)
from catan.view import PublicView
from tests.helpers import scramble_hidden_state


@pytest.fixture
def view():
    env = CatanEnv(num_players=2)
    _, info = env.reset(seed=5)
    for _ in range(150):
        if info["done"]:
            break
        _, _, _, _, info = env.step(info["legal"][-1])
    return PublicView(env.state, 1)


# =========================================================================== #
# THE ARITHMETIC                                                              #
# =========================================================================== #

def test_odds_are_the_dice():
    assert heuristics.odds(7) == pytest.approx(6 / 36)
    assert heuristics.odds(6) == heuristics.odds(8) == pytest.approx(5 / 36)
    assert heuristics.odds(2) == heuristics.odds(12) == pytest.approx(1 / 36)
    assert sum(heuristics.odds(n) for n in range(2, 13)) == pytest.approx(1.0)


def test_the_desert_is_worth_nothing(view):
    desert = next(t for t in range(1, NUM_TILES + 1) if view.board.resource_at(t) is None)
    assert heuristics.tile_value(view, desert) == 0.0
    assert heuristics.robber_damage(view, desert, 1) == 0.0


def test_the_robber_zeroes_a_tile(view):
    """And that it is the robber doing it, not the tile being poor."""
    tile = view.robber_tile
    free = heuristics.tile_value(view, tile, count_robber=False)
    assert heuristics.tile_value(view, tile) == 0.0
    if view.board.resource_at(tile) is not None:
        assert free > 0.0


def test_a_vertex_is_worth_its_tiles(view):
    vertex = 20
    assert heuristics.vertex_value(view, vertex) == pytest.approx(
        sum(heuristics.tile_value(view, t) for t in VERTEX_TILES[vertex])
    )


def test_a_vertex_touching_only_the_desert_is_a_float(view):
    """`sum()` of an empty generator is int 0, which used to leak an int out of here."""
    for vertex in range(1, NUM_VERTICES + 1):
        assert isinstance(heuristics.vertex_value(view, vertex), float)


def test_a_city_doubles_income(view):
    state = view._state
    vertex = next(
        v for v in state.buildings_of(1) if state.vertex_piece[v] is Piece.SETTLEMENT
        and any(view.board.resource_at(t) is not None for t in VERTEX_TILES[v])
    )
    before = heuristics.income(view, 1)
    state.vertex_piece[vertex] = Piece.CITY
    after = heuristics.income(view, 1)

    for tile in VERTEX_TILES[vertex]:
        resource = view.board.resource_at(tile)
        if resource is None or view.robber_tile == tile:
            continue
        gain = heuristics.odds(view.board.number_at(tile))
        assert after[resource] == pytest.approx(before[resource] + gain)


def test_a_city_is_worth_the_production_it_repeats(view):
    vertex = next(iter(view.buildings_of(1)))
    assert heuristics.city_value(view, 1, vertex) == heuristics.vertex_value(view, vertex)


# =========================================================================== #
# THE IDEA: MARGINAL VALUE                                                    #
# =========================================================================== #

def test_a_resource_you_already_have_is_worth_less(view):
    """The one observation that separates a plausible opening from a bad one."""
    vertex = next(
        v for v in range(1, NUM_VERTICES + 1)
        if any(view.board.resource_at(t) is not None for t in VERTEX_TILES[v])
    )
    resource = next(
        view.board.resource_at(t) for t in VERTEX_TILES[vertex]
        if view.board.resource_at(t) is not None
    )

    poor = [0.0] * NUM_RESOURCES
    rich = [0.0] * NUM_RESOURCES
    rich[resource] = 3.0

    assert (heuristics.settlement_value(view, 1, vertex, rich)
            < heuristics.settlement_value(view, 1, vertex, poor))


def test_diversity_beats_concentration(view):
    """Two different resources are worth more than twice the same one, all else equal."""
    empty = [0.0] * NUM_RESOURCES
    values = {
        v: heuristics.settlement_value(view, 1, v, empty)
        for v in range(1, NUM_VERTICES + 1)
    }
    kinds = {
        v: len({view.board.resource_at(t) for t in VERTEX_TILES[v]} - {None})
        for v in values
    }
    three = [values[v] for v in values if kinds[v] == 3]
    one = [values[v] for v in values if kinds[v] == 1]
    assert three and one
    assert sum(three) / len(three) > sum(one) / len(one)


def test_a_harbour_adds_value_only_for_what_you_produce(view):
    harbour_vertex = next(
        (v for v in range(1, NUM_VERTICES + 1)
         if any(h is not GENERIC_HARBOUR for h in view.board.harbours_at(v))),
        None,
    )
    assert harbour_vertex is not None, "every board has specific harbours"
    resource = next(h for h in view.board.harbours_at(harbour_vertex) if h is not GENERIC_HARBOUR)

    none = [0.0] * NUM_RESOURCES
    lots = [0.0] * NUM_RESOURCES
    lots[resource] = 5.0
    assert heuristics.port_value(view, harbour_vertex, none) == 0.0
    assert heuristics.port_value(view, harbour_vertex, lots) > 0.0


def test_best_settlement_spot_picks_the_best_one(view):
    spots = heuristics.open_spots(view)
    have = heuristics.income(view, 1)
    best = heuristics.best_settlement_spot(view, 1, spots, have)
    assert best in spots
    assert all(
        heuristics.settlement_value(view, 1, s, have)
        <= heuristics.settlement_value(view, 1, best, have) + 1e-12
        for s in spots
    )


def test_best_settlement_spot_of_nothing_is_nothing(view):
    assert heuristics.best_settlement_spot(view, 1, []) is None


def test_open_spots_obey_the_distance_rule(view):
    for vertex in heuristics.open_spots(view):
        assert view.vertex_owner[vertex] == NO_OWNER
        assert all(view.vertex_owner[n] == NO_OWNER for n in VERTEX_NEIGHBOURS[vertex])


def test_a_road_is_worth_where_it_leads(view):
    """Never negative, and worthless when it reaches nothing — which is what stops the
    agent laying track across the board for its own sake."""
    values = [
        heuristics.road_value(view, 1, road)
        for road in range(1, len(view.edge_owner))
    ]
    assert all(v >= 0.0 for v in values)
    assert any(v > 0.0 for v in values)


def test_the_robber_hurts_cities_twice_as_much(view):
    state = view._state
    tile = next(
        t for t in range(1, NUM_TILES + 1)
        if view.board.resource_at(t) is not None
        and any(state.vertex_owner[v] == 1 for v in TILE_VERTICES[t])
    )
    vertex = next(v for v in TILE_VERTICES[tile] if state.vertex_owner[v] == 1)
    state.vertex_piece[vertex] = Piece.SETTLEMENT
    as_settlement = heuristics.robber_damage(view, tile, 1)
    state.vertex_piece[vertex] = Piece.CITY
    assert heuristics.robber_damage(view, tile, 1) > as_settlement


def test_the_bank_is_the_only_window_onto_other_hands(view):
    """Missing cards are in somebody's hand — public arithmetic, no peeking."""
    scarcest = heuristics.scarcest_in_bank(view)
    assert view.bank[scarcest] == min(view.bank)

    state = view._state
    for resource in Resource:
        actually_held = sum(
            state.hands[p][resource] for p in state.players if p != 1
        )
        assert heuristics.held_by_others(view, resource) == actually_held


# =========================================================================== #
# THE AGENT                                                                   #
# =========================================================================== #

def test_it_is_just_a_callable_like_the_others():
    env = CatanEnv(num_players=2)
    observation, info = env.reset(seed=1)
    assert isinstance(HeuristicAgent(0)(observation, info), int)


def test_it_only_ever_returns_a_legal_action():
    env = CatanEnv(num_players=2)
    agent = HeuristicAgent(0)
    observation, info = env.reset(seed=2)
    for _ in range(400):
        if info["done"]:
            break
        action = agent(observation, info)
        assert action in info["legal"], f"illegal in {info['phase']}"
        observation, _, _, _, info = env.step(action)


def test_it_falls_back_when_driven_without_a_view():
    """Anything that drives an agent by hand should still get a move, not a crash."""
    env = CatanEnv(num_players=2)
    observation, info = env.reset(seed=1)
    info.pop("view")
    assert HeuristicAgent(0)(observation, info) in info["legal"]


@pytest.mark.parametrize("noise", sorted(DIFFICULTY.values()))
def test_every_difficulty_plays_a_whole_game(noise):
    result = play_match(
        {1: HeuristicAgent(1, noise=noise), 2: HeuristicAgent(2, noise=noise)},
        games=2, seed=4,
    )
    assert result[1] + result[2] + result["truncated"] == 2


def test_it_beats_the_greedy_agent():
    """The reason it exists. Measured at ~97% over 40 games; 8 of 10 is a wide margin."""
    result = play_match({1: HeuristicAgent(0), 2: GreedyAgent(0)}, games=10, seed=7)
    assert result[1] >= 8, result


def test_it_beats_the_random_agent():
    result = play_match({1: HeuristicAgent(0), 2: RandomAgent(0)}, games=6, seed=7)
    assert result[1] >= 5, result


def test_the_opening_is_not_random():
    """Setup decides most 1v1 games, so the first placement must be near the best on offer."""
    env = CatanEnv(num_players=2)
    observation, info = env.reset(seed=9)
    chosen = action_space.decode(HeuristicAgent(0)(observation, info)).position

    view = info["view"]
    have = heuristics.income(view, view.me)
    offered = [
        action_space.decode(i).position for i in info["legal"]
    ]
    best = max(heuristics.settlement_value(view, view.me, v, have) for v in offered)
    assert heuristics.settlement_value(view, view.me, chosen, have) == pytest.approx(best)


# =========================================================================== #
# THE LEAK TEST                                                               #
# =========================================================================== #

def test_the_agent_cannot_see_the_opponent_s_cards():
    """Play a game; at every decision, ask again with the hidden cards rewritten.

    A fresh agent with the same seed each time, so the answer is a function of the position
    alone and any difference is information leaking in.
    """
    env = CatanEnv(num_players=2)
    observation, info = env.reset(seed=13)
    checked = 0

    for _ in range(400):
        if info["done"]:
            break

        honest = HeuristicAgent(7)(observation, info)

        scrambled = scramble_hidden_state(env.state.clone(), info["player"])
        assert action_space.legal_indices(scrambled) == info["legal"], (
            "the scramble changed what is legal — the test would be comparing "
            "two different positions"
        )
        cheat_info = dict(info, view=PublicView(scrambled, info["player"]))
        cheating = HeuristicAgent(7)(observation, cheat_info)

        assert honest == cheating, (
            f"turn {info['turn']}, {info['phase']}: the agent moved differently once the "
            f"opponent's hidden cards changed — it is reading them"
        )
        checked += 1
        observation, _, _, _, info = env.step(honest)

    assert checked > 100, f"only {checked} decisions checked"


def test_the_scramble_would_be_noticed_by_an_agent_that_did_cheat():
    """Otherwise the leak test above passes because the mutation does nothing."""
    env = CatanEnv(num_players=2)
    _, info = env.reset(seed=13)
    for _ in range(150):
        if info["done"]:
            break
        _, _, _, _, info = env.step(info["legal"][-1])

    before = [list(env.state.hands[p]) for p in env.state.players]
    scrambled = scramble_hidden_state(env.state.clone(), 1)
    after = [list(scrambled.hands[p]) for p in scrambled.players]
    assert before != after, "the opponent had nothing to scramble; pick a longer game"
    assert list(scrambled.hands[1]) == list(env.state.hands[1]), "my own hand is untouched"

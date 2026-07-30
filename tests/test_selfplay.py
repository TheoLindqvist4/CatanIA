"""Whole-game fuzzing: drive random legal play and assert the state stays coherent.

This is the cheap version of the 10k-game harness Phase 3 wants. It is not checking that
games are *good* — random play is terrible — only that the engine never reaches an
impossible state and never crashes.
"""

import collections

import pytest

pytestmark = pytest.mark.slow

import catan.topology as T
from catan import rules
from catan.resources import BANK_PER_RESOURCE, NUM_RESOURCES, total
from catan.state import (
    MAX_CITIES,
    MAX_ROADS,
    MAX_SETTLEMENTS,
    NO_OWNER,
    Phase,
    Piece,
)
from helpers import play_random_game

GAMES = 60


def assert_invariants(state):
    """Everything that must be true of any reachable state."""
    # --- ownership is coherent ---
    for vertex in range(1, T.NUM_VERTICES + 1):
        owner, piece = state.vertex_owner[vertex], state.vertex_piece[vertex]
        assert (owner == NO_OWNER) == (piece is Piece.NONE), \
            f"vertex {vertex}: owner {owner} but piece {piece}"
        if owner != NO_OWNER:
            assert owner in state.players

    for road in range(1, T.NUM_ROADS + 1):
        assert state.edge_owner[road] == NO_OWNER or state.edge_owner[road] in state.players

    # slot 0 is never used
    assert state.vertex_owner[0] == NO_OWNER
    assert state.edge_owner[0] == NO_OWNER

    # --- the distance rule is never violated ---
    for vertex in range(1, T.NUM_VERTICES + 1):
        if state.vertex_owner[vertex] == NO_OWNER:
            continue
        for neighbour in T.VERTEX_NEIGHBOURS[vertex]:
            assert state.vertex_owner[neighbour] == NO_OWNER, \
                f"adjacent buildings at {vertex} and {neighbour}"

    # --- piece accounting adds up ---
    for player in state.players:
        settlements = sum(
            1 for v in range(1, T.NUM_VERTICES + 1)
            if state.vertex_owner[v] == player
            and state.vertex_piece[v] is Piece.SETTLEMENT
        )
        cities = sum(
            1 for v in range(1, T.NUM_VERTICES + 1)
            if state.vertex_owner[v] == player and state.vertex_piece[v] is Piece.CITY
        )
        roads = len(state.roads_of(player))

        # a city hands its settlement back, so settlements on the board plus the
        # supply equals the starting count
        assert settlements + state.settlements_left[player] == MAX_SETTLEMENTS
        assert cities + state.cities_left[player] == MAX_CITIES
        assert roads + state.roads_left[player] == MAX_ROADS

        assert 0 <= state.settlements_left[player] <= MAX_SETTLEMENTS
        assert 0 <= state.cities_left[player] <= MAX_CITIES
        assert 0 <= state.roads_left[player] <= MAX_ROADS

    # --- hands and the bank are well formed, and cards are conserved ---
    for player in state.players:
        hand = state.hands[player]
        assert len(hand) == NUM_RESOURCES
        assert all(isinstance(n, int) and n >= 0 for n in hand), \
            f"player {player} has a negative or non-integer hand: {hand}"

    assert len(state.bank) == NUM_RESOURCES
    assert all(n >= 0 for n in state.bank), f"the bank went negative: {state.bank}"
    for resource in range(NUM_RESOURCES):
        held = sum(state.hands[p][resource] for p in state.players)
        assert held + state.bank[resource] == BANK_PER_RESOURCE, \
            f"resource {resource}: {held} held + {state.bank[resource]} in bank"

    # --- every road connects to its owner's network ---
    for road in range(1, T.NUM_ROADS + 1):
        owner = state.edge_owner[road]
        if owner == NO_OWNER:
            continue
        endpoints = T.ROAD_VERTICES[road]
        touches = any(
            state.vertex_owner[v] == owner
            or any(state.edge_owner[r] == owner for r in T.VERTEX_ROADS[v] if r != road)
            for v in endpoints
        )
        assert touches, f"road {road} of player {owner} is orphaned"

    # --- phase and turn bookkeeping ---
    assert state.current_player in state.players
    assert state.turn_player in state.players
    if state.phase is Phase.GAME_OVER:
        assert state.winner in state.players
        assert rules.victory_points(state, state.winner) >= 10
    else:
        assert state.winner is None
    if state.in_setup:
        assert state.setup_step < 2 * state.num_players
    if state.last_roll is not None:
        assert 2 <= state.last_roll <= 12

    # --- the robber ---
    assert 1 <= state.robber_tile <= T.NUM_TILES

    # --- discards only pend during the discard phase, and only for players who owe ---
    if state.phase is Phase.DISCARD:
        assert state.pending_discards, "discard phase with nobody owing"
        assert rules.owes_discard(state, state.pending_discards[0]), \
            "the player being asked to discard does not owe any cards"
        assert state.current_player == state.pending_discards[0]
    else:
        assert not state.pending_discards, \
            f"{state.pending_discards} still owe discards in {state.phase.name}"
    assert all(p in state.players for p in state.pending_discards)
    assert len(set(state.pending_discards)) == len(state.pending_discards)
    for player in state.players:
        assert state.discards_owed[player] >= 0
        if state.discards_owed[player] > 0:
            assert player in state.pending_discards
            assert state.discards_owed[player] <= total(state.hands[player])


@pytest.mark.parametrize("num_players", [2, 3, 4])
def test_random_games_never_reach_an_impossible_state(num_players):
    for seed in range(GAMES // 3):
        play_random_game(seed=seed, num_players=num_players, max_actions=1500,
                         on_step=assert_invariants)


def test_the_setup_phase_always_completes():
    for seed in range(GAMES):
        for num_players in (2, 3, 4):
            state = play_random_game(seed=seed, num_players=num_players,
                                     max_actions=4 * num_players)
            assert not state.in_setup, "setup did not finish"
            for player in state.players:
                assert len(state.buildings_of(player)) == 2
                assert len(state.roads_of(player)) == 2


def test_no_player_ever_goes_into_resource_debt():
    """Every build must be fully paid for. A negative hand means a cost was charged
    without a legality check, or vice versa."""
    for seed in range(GAMES):
        state = play_random_game(seed=seed, max_actions=1200)
        for player in state.players:
            assert all(n >= 0 for n in state.hands[player])


def test_victory_points_stay_consistent_with_the_board():
    for seed in range(GAMES):
        state = play_random_game(seed=seed, max_actions=1200)
        for player in state.players:
            expected = sum(
                1 if state.vertex_piece[v] is Piece.SETTLEMENT else 2
                for v in state.buildings_of(player)
            )
            assert rules.victory_points(state, player) == expected


def test_replaying_a_seed_gives_an_identical_game():
    """Reproducibility is the whole point of injecting the RNG."""
    for seed in (0, 1, 2, 7):
        a = play_random_game(seed=seed, max_actions=900)
        b = play_random_game(seed=seed, max_actions=900)
        assert a == b


def test_some_games_are_actually_won():
    """Sanity that the win path is reachable at all, not just that nothing crashes."""
    winners = [
        play_random_game(seed=seed, max_actions=6000).winner
        for seed in range(30)
    ]
    assert any(w is not None for w in winners), "no game ever reached 10 points"


def test_almost_every_game_now_finishes():
    """Bank trading is what made this true.

    Before it, only 4 of 40 random games reached 10 points: a settlement needs four
    different resources, most players' buildings reach only three, and there was no way
    to convert a surplus. With 4:1 (and harbour) trading, nearly all games finish.

    A stalled game is still not a deadlock — END_TURN stays legal — so that is checked
    too rather than assumed.
    """
    finished = 0
    for seed in range(20):
        state = play_random_game(seed=seed, max_actions=6000)
        if state.winner is not None:
            finished += 1
            continue
        if state.phase is Phase.ROLL:
            rules.roll_dice(state)
        assert rules.end_turn() in rules.legal_actions(state)

    assert finished >= 16, f"only {finished}/20 games finished; trading may have regressed"


def test_cards_are_conserved():
    """Every card is either in the bank or in a hand. 19 x 5 = 95, always.

    Catches a payment that forgets to refund the bank, or a payout that mints cards.
    """
    for seed in range(12):
        for num_players in (2, 4):
            state = play_random_game(seed=seed, num_players=num_players,
                                     max_actions=2500)
            in_hands = sum(sum(state.hands[p]) for p in state.players)
            assert in_hands + sum(state.bank) == 5 * BANK_PER_RESOURCE
            assert all(n >= 0 for n in state.bank), "the bank went negative"


def test_the_documented_driver_loop_terminates():
    """Mirrors the loop in README.md and docs/engine.md.

    The first version of that snippet used `actions[0]`, which is always END_TURN during
    BUILD — so it ended turns forever and never terminated. The turn bound and the
    deliberate choice are both load-bearing; this pins the documented example.
    """
    import random

    from catan.state import GameState

    state = GameState(num_players=3, seed=42)
    agent = random.Random(0)

    turns = 0
    while state.phase is not Phase.GAME_OVER and state.turn_number < 500:
        turns += 1
        assert turns < 50_000, "the documented loop does not terminate"
        if state.phase is Phase.ROLL:
            rules.roll_dice(state)
            continue
        actions = rules.legal_actions(state)
        if not actions:
            break
        rules.apply(state, agent.choice(actions))

    assert state.phase is Phase.GAME_OVER or state.turn_number >= 500
    assert_invariants(state)


def test_action_and_roll_counts_are_plausible():
    """A smoke check on distribution: builds happen, rolls cover the range."""
    rolls = collections.Counter()
    builds = 0

    def observe(state):
        nonlocal builds
        if state.last_roll is not None:
            rolls[state.last_roll] += 1
        builds = sum(
            len(state.buildings_of(p)) + len(state.roads_of(p)) for p in state.players
        )

    play_random_game(seed=4, max_actions=2500, on_step=observe)
    assert set(rolls) <= set(range(2, 13))
    assert len(rolls) >= 9, "rolls should cover most of the range"
    assert builds > 8, "more than the setup pieces should get built"

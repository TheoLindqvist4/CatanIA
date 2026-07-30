"""Shared helpers for the engine tests.

These reach past :mod:`catan.rules` and write straight into
:class:`~catan.state.GameState` on purpose: a unit test needs to construct a specific
position without playing 40 legal moves to get there. Tests that exercise the *rules*
must go through :func:`catan.rules.apply`.
"""

import random

from catan import rules
from catan.state import GameState, Phase, Piece


def fresh(num_players=2, seed=0, **kwargs):
    return GameState(num_players=num_players, seed=seed, **kwargs)


def give(state, player, wood=0, brick=0, sheep=0, wheat=0, ore=0):
    """Set a player's hand outright. Order matches Resource: wood brick sheep wheat ore."""
    state.hands[player] = [wood, brick, sheep, wheat, ore]
    return state.hands[player]


def enough_for_everything(state, player, times=10):
    give(state, player, wood=times, brick=times, sheep=times,
         wheat=2 * times, ore=3 * times)


def put_building(state, player, vertex, piece=Piece.SETTLEMENT):
    """Force a building into place, ignoring cost, connection and spacing."""
    state.vertex_owner[vertex] = player
    state.vertex_piece[vertex] = piece
    return vertex


def put_road(state, player, road):
    """Force a road into place, ignoring cost and connection."""
    state.edge_owner[road] = player
    return road


def put_roads(state, player, roads):
    for road in roads:
        put_road(state, player, road)
    return tuple(roads)


def in_build_phase(state, player=None):
    """Jump straight to a build decision for ``player`` (default: player 1)."""
    state.phase = Phase.BUILD
    state.setup_step = 0
    target = player if player is not None else state.player_order[0]
    state.turn_number = state.player_order.index(target)
    assert state.current_player == target
    return state


def complete_setup(state, rng=None):
    """Play the whole setup phase with random legal choices.

    Leaves the state in :attr:`~catan.state.Phase.ROLL`.
    """
    rng = rng or random.Random(0)
    while state.in_setup:
        actions = rules.legal_actions(state)
        assert actions, f"no legal action during {state.phase.name}"
        rules.apply(state, rng.choice(actions))
    return state


def play_random_game(seed=0, max_actions=20_000, prefer_building=True,
                     num_players=2, on_step=None):
    """Build a state and drive a full game with random legal actions."""
    state = GameState(num_players=num_players, seed=seed)
    drive(state, random.Random(seed ^ 0x5EED), max_actions=max_actions,
          prefer_building=prefer_building, on_step=on_step)
    return state


def drive(state, rng, max_actions=20_000, prefer_building=True, on_step=None):
    """Play ``state`` forward with random legal actions.

    Stops at ``max_actions`` so a non-terminating game fails a test instead of hanging —
    without trade, ports or dev cards many games genuinely cannot reach 10 points.

    Args:
        prefer_building: pick a build over ending the turn when both are available,
            which reaches interesting positions far faster than uniform choice.
        on_step: called as ``on_step(state)`` after every mutation, for invariant checks.
    """
    for _ in range(max_actions):
        if state.phase is Phase.GAME_OVER:
            break
        if state.phase is Phase.ROLL:
            rules.roll_dice(state)
        else:
            actions = rules.legal_actions(state)
            if not actions:
                break
            if prefer_building and len(actions) > 1:
                builds = [a for a in actions if a != rules.end_turn()]
                actions = builds or actions
            rules.apply(state, rng.choice(actions))
        if on_step is not None:
            on_step(state)

    return state


def extend_to_free_vertex(state, player, limit=12):
    """Force-place roads until ``player`` can legally settle somewhere; return where.

    Straight after setup a player has no legal settlement spot: both their roads lead to
    a vertex adjacent to their own settlement, which the distance rule blocks. Reaching
    a third spot genuinely requires building outward first.
    """
    from catan.topology import NUM_ROADS, ROAD_VERTICES

    for _ in range(limit):
        for road in state.roads_of(player):
            for vertex in ROAD_VERTICES[road]:
                if rules.respects_distance_rule(state, vertex):
                    return vertex

        reachable = {v for r in state.roads_of(player) for v in ROAD_VERTICES[r]}
        for road in range(1, NUM_ROADS + 1):
            if state.is_road_free(road) and set(ROAD_VERTICES[road]) & reachable:
                put_road(state, player, road)
                break
        else:
            break

    raise AssertionError(f"player {player} could not reach a free vertex")


def roll_sequence(state, count):
    """Roll ``count`` times, resetting the phase between rolls.

    Only for comparing random streams — a real game rolls once per turn.
    """
    rolls = []
    for _ in range(count):
        state.phase = Phase.ROLL
        rolls.append(rules.roll_dice(state))
    return rolls


def snapshot_board(board):
    """Everything about a board that must never change during a game."""
    return (
        list(board.tile_numbers),
        list(board.tile_resources),
        dict(board.vertex_production),
        board.desert_tile,
        {roll: dict(board.producers_for(roll)) for roll in range(2, 13)},
    )

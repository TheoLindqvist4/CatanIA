"""The rules. The only authority on what is legal and what an action does.

Two entry points matter:

* :func:`legal_actions` — every action the current player may take right now.
* :func:`apply` — perform one, or raise if it is not legal.

Both go through the same ``can_*`` predicates, so a move can never be offered by one and
rejected by the other. That was the old bug: ``Game`` computed legal positions in
``check_valid_*`` and then ``place_*`` ignored them, so ``place_road(p, 70)`` succeeded on
an empty board.

Dice are not an action. :func:`roll_dice` is environment stochasticity and the driver
calls it when ``state.phase is Phase.ROLL``; ``legal_actions`` returns nothing then.

:func:`apply` **mutates** ``state`` and returns it. Copy first with
:meth:`~catan.state.GameState.clone` if you need the old one — see
``docs/decisions/0008-mutating-apply-plus-clone.md``.

No I/O, and no use of the global ``random`` module.
"""

from catan import resources
from catan.actions import Action, ActionType, build_city, build_road, build_settlement, end_turn
from catan.board import ROBBER_ROLL
from catan.resources import CITY_COST, ROAD_COST, SETTLEMENT_COST
from catan.state import (
    NO_OWNER,
    PIECE_VICTORY_POINTS,
    PIECE_YIELD,
    VICTORY_POINTS_TO_WIN,
    Phase,
    Piece,
)
from catan.topology import (
    NUM_ROADS,
    NUM_VERTICES,
    ROAD_VERTICES,
    VERTEX_NEIGHBOURS,
    VERTEX_ROADS,
)

DICE_FACES = 6


class IllegalAction(ValueError):
    """Raised when :func:`apply` is given an action the rules do not allow."""


# --------------------------------------------------------------------------- #
# PLACEMENT PREDICATES                                                        #
# --------------------------------------------------------------------------- #

def respects_distance_rule(state, vertex):
    """Whether ``vertex`` and all its neighbours are empty.

    Catan's distance rule: no two buildings may be adjacent. Derived from ownership
    rather than stored, so "empty but blocked" stays distinguishable from "occupied".
    """
    if state.vertex_owner[vertex] != NO_OWNER:
        return False
    return all(
        state.vertex_owner[neighbour] == NO_OWNER
        for neighbour in VERTEX_NEIGHBOURS[vertex]
    )


def touches_own_road(state, player, vertex):
    """Whether ``player`` has a road meeting ``vertex``."""
    return any(state.edge_owner[road] == player for road in VERTEX_ROADS[vertex])


def can_place_setup_settlement(state, vertex):
    """Setup placement: free, and needs no road connection."""
    return respects_distance_rule(state, vertex)


def can_place_setup_road(state, player, road):
    """Setup road: free, and must touch the settlement just placed."""
    if not state.is_road_free(road):
        return False
    if state.roads_left[player] <= 0:
        return False
    return state.last_settlement in ROAD_VERTICES[road]


def can_build_road(state, player, road):
    """A road must be free, affordable, and connected to the player's network.

    Connection means one endpoint either carries the player's own building, or is a
    junction the player can build through — that is, it holds no *opponent* building and
    has one of the player's roads. An opponent's building blocks a road from being
    extended past it.
    """
    if not state.is_road_free(road):
        return False
    if state.roads_left[player] <= 0:
        return False
    if not resources.can_afford(state.hands[player], ROAD_COST):
        return False
    return any(
        _connects_at(state, player, endpoint) for endpoint in ROAD_VERTICES[road]
    )


def _connects_at(state, player, vertex):
    owner = state.vertex_owner[vertex]
    if owner == player:
        return True
    if owner != NO_OWNER:
        return False  # an opponent's building blocks the junction
    return touches_own_road(state, player, vertex)


def can_build_settlement(state, player, vertex):
    """A settlement needs the distance rule, a connecting road, pieces and payment."""
    if state.settlements_left[player] <= 0:
        return False
    if not respects_distance_rule(state, vertex):
        return False
    if not touches_own_road(state, player, vertex):
        return False
    return resources.can_afford(state.hands[player], SETTLEMENT_COST)


def can_build_city(state, player, vertex):
    """A city upgrades one of the player's own settlements."""
    if state.cities_left[player] <= 0:
        return False
    if state.vertex_owner[vertex] != player:
        return False
    if state.vertex_piece[vertex] is not Piece.SETTLEMENT:
        return False
    return resources.can_afford(state.hands[player], CITY_COST)


# --------------------------------------------------------------------------- #
# LEGAL ACTIONS                                                               #
# --------------------------------------------------------------------------- #

def legal_actions(state):
    """Every action ``state.current_player`` may take now.

    Empty during :attr:`~catan.state.Phase.ROLL` (call :func:`roll_dice`) and after the
    game is over.
    """
    player = state.current_player

    if state.phase is Phase.SETUP_SETTLEMENT:
        return [
            build_settlement(v)
            for v in range(1, NUM_VERTICES + 1)
            if can_place_setup_settlement(state, v)
        ]

    if state.phase is Phase.SETUP_ROAD:
        return [
            build_road(r)
            for r in range(1, NUM_ROADS + 1)
            if can_place_setup_road(state, player, r)
        ]

    if state.phase is Phase.BUILD:
        actions = [end_turn()]
        actions += [
            build_road(r)
            for r in range(1, NUM_ROADS + 1)
            if can_build_road(state, player, r)
        ]
        actions += [
            build_settlement(v)
            for v in range(1, NUM_VERTICES + 1)
            if can_build_settlement(state, player, v)
        ]
        actions += [
            build_city(v)
            for v in range(1, NUM_VERTICES + 1)
            if can_build_city(state, player, v)
        ]
        return actions

    return []  # ROLL, GAME_OVER


def is_legal(state, action):
    return action in legal_actions(state)


# --------------------------------------------------------------------------- #
# APPLYING AN ACTION                                                          #
# --------------------------------------------------------------------------- #

def apply(state, action):
    """Perform ``action``, mutating and returning ``state``.

    Raises:
        IllegalAction: if the rules do not allow it.
    """
    if not isinstance(action, Action):
        raise IllegalAction(f"expected an Action, got {action!r}")

    player = state.current_player

    if state.phase is Phase.SETUP_SETTLEMENT:
        _apply_setup_settlement(state, player, action)
    elif state.phase is Phase.SETUP_ROAD:
        _apply_setup_road(state, player, action)
    elif state.phase is Phase.BUILD:
        _apply_build(state, player, action)
    elif state.phase is Phase.ROLL:
        raise IllegalAction("must roll the dice before acting")
    else:
        raise IllegalAction("the game is over")

    return state


def _apply_setup_settlement(state, player, action):
    if action.type is not ActionType.BUILD_SETTLEMENT:
        raise IllegalAction(f"setup expects a settlement, got {action!r}")
    vertex = _check_vertex(action.position)
    if not can_place_setup_settlement(state, vertex):
        raise IllegalAction(f"cannot place a settlement at {vertex}")

    _put_building(state, player, vertex, Piece.SETTLEMENT)
    state.last_settlement = vertex

    # The second settlement pays out its adjacent tiles immediately.
    if state.setup_round == 2:
        for resource in state.board.resources_at(vertex):
            state.hands[player][resource] += 1

    state.phase = Phase.SETUP_ROAD


def _apply_setup_road(state, player, action):
    if action.type is not ActionType.BUILD_ROAD:
        raise IllegalAction(f"setup expects a road, got {action!r}")
    road = _check_road(action.position)
    if not can_place_setup_road(state, player, road):
        raise IllegalAction(f"cannot place a road at {road}")

    _put_road(state, player, road)
    state.last_settlement = None
    _advance_setup(state)


def _advance_setup(state):
    state.setup_step += 1
    if state.setup_step >= 2 * state.num_players:
        state.phase = Phase.ROLL
        state.setup_step = 0
        state.turn_number = 0
    else:
        state.phase = Phase.SETUP_SETTLEMENT


def _apply_build(state, player, action):
    if action.type is ActionType.END_TURN:
        state.turn_number += 1
        state.phase = Phase.ROLL
        return

    if action.type is ActionType.BUILD_ROAD:
        road = _check_road(action.position)
        if not can_build_road(state, player, road):
            raise IllegalAction(f"cannot build a road at {road}")
        resources.pay(state.hands[player], ROAD_COST)
        _put_road(state, player, road)

    elif action.type is ActionType.BUILD_SETTLEMENT:
        vertex = _check_vertex(action.position)
        if not can_build_settlement(state, player, vertex):
            raise IllegalAction(f"cannot build a settlement at {vertex}")
        resources.pay(state.hands[player], SETTLEMENT_COST)
        _put_building(state, player, vertex, Piece.SETTLEMENT)

    elif action.type is ActionType.BUILD_CITY:
        vertex = _check_vertex(action.position)
        if not can_build_city(state, player, vertex):
            raise IllegalAction(f"cannot build a city at {vertex}")
        resources.pay(state.hands[player], CITY_COST)
        state.vertex_piece[vertex] = Piece.CITY
        state.cities_left[player] -= 1
        state.settlements_left[player] += 1  # the settlement returns to the supply

    else:
        raise IllegalAction(f"unknown action {action!r}")

    _check_for_winner(state, player)


def _put_building(state, player, vertex, piece):
    state.vertex_owner[vertex] = player
    state.vertex_piece[vertex] = piece
    state.settlements_left[player] -= 1


def _put_road(state, player, road):
    state.edge_owner[road] = player
    state.roads_left[player] -= 1


def _check_vertex(position):
    if not 1 <= position <= NUM_VERTICES:
        raise IllegalAction(f"vertex must be in 1..{NUM_VERTICES}, got {position}")
    return position


def _check_road(position):
    if not 1 <= position <= NUM_ROADS:
        raise IllegalAction(f"road must be in 1..{NUM_ROADS}, got {position}")
    return position


# --------------------------------------------------------------------------- #
# DICE AND PRODUCTION                                                         #
# --------------------------------------------------------------------------- #

def roll_dice(state):
    """Roll 2d6, pay out production, and move to the build phase.

    Two independent dice, not one uniform draw over 2..12 — the triangular
    distribution is the whole point of Catan's probabilities.

    Returns:
        int: the roll.
    """
    if state.phase is not Phase.ROLL:
        raise IllegalAction(f"cannot roll during {state.phase.name}")

    roll = state.rng.randint(1, DICE_FACES) + state.rng.randint(1, DICE_FACES)
    state.last_roll = roll

    # Phase 2 turns a 7 into: move the robber, steal a card, discard above 7 cards.
    # Until then a 7 simply pays nobody, which the board's payout index guarantees
    # anyway — this branch is documentation, not logic.
    if roll != ROBBER_ROLL:
        distribute(state, roll)

    state.phase = Phase.BUILD
    return roll


def distribute(state, roll):
    """Pay every player for their buildings on ``roll``. Cities yield double."""
    for vertex, productions in state.board.producers_for(roll).items():
        owner = state.vertex_owner[vertex]
        if owner == NO_OWNER:
            continue
        amount = PIECE_YIELD[state.vertex_piece[vertex]]
        hand = state.hands[owner]
        for production in productions:
            hand[production.resource] += amount


# --------------------------------------------------------------------------- #
# SCORING                                                                     #
# --------------------------------------------------------------------------- #

def victory_points(state, player):
    """Points from buildings: 1 per settlement, 2 per city.

    Derived from the board rather than kept as a counter, so it cannot drift out of
    sync. Phase 2 adds Longest Road, Largest Army and victory-point dev cards.
    """
    return sum(
        PIECE_VICTORY_POINTS[state.vertex_piece[vertex]]
        for vertex in range(1, NUM_VERTICES + 1)
        if state.vertex_owner[vertex] == player
    )


def scores(state):
    """``{player: victory points}``."""
    return {player: victory_points(state, player) for player in state.players}


def _check_for_winner(state, player):
    if victory_points(state, player) >= VICTORY_POINTS_TO_WIN:
        state.winner = player
        state.phase = Phase.GAME_OVER


# --------------------------------------------------------------------------- #
# LONGEST ROAD                                                                #
# --------------------------------------------------------------------------- #

def longest_road_length(state, player):
    """Longest continuous chain of ``player``'s roads.

    Two rules, both settled in ``docs/decisions/0006``:

    * **Strict simple path.** A route may not pass through the same intersection twice.
      Since no vertex has more than three roads, this only differs from "never reuse a
      road" where the player owns all three roads at one vertex.
    * **An opponent's building breaks a road.** A chain may *end* at an opponent's
      settlement or city but may not continue through it.

    Search starts from each road in each direction, so the starting vertex is treated as
    a free endpoint — an opponent's building there does not shorten the chain.

    Branches do not count: the search walks a path, so only one arm at a junction is
    ever followed.
    """
    owned = state.roads_of(player)
    if not owned:
        return 0

    adjacency = {}
    for road in owned:
        u, v = ROAD_VERTICES[road]
        adjacency.setdefault(u, []).append((v, road))
        adjacency.setdefault(v, []).append((u, road))

    def blocked(vertex):
        owner = state.vertex_owner[vertex]
        return owner != NO_OWNER and owner != player

    def extend(vertex, used_roads, visited):
        if blocked(vertex):
            return 0
        best = 0
        for neighbour, road in adjacency[vertex]:
            if road in used_roads or neighbour in visited:
                continue
            used_roads.add(road)
            visited.add(neighbour)
            best = max(best, 1 + extend(neighbour, used_roads, visited))
            visited.discard(neighbour)
            used_roads.discard(road)
        return best

    longest = 0
    for road in owned:
        u, v = ROAD_VERTICES[road]
        for start, onward in ((u, v), (v, u)):
            longest = max(longest, 1 + extend(onward, {road}, {start, onward}))
    return longest


def longest_road_holder(state):
    """``(player, length)`` for the longest road, or ``(None, best)`` on a tie.

    Awarding the 2 victory points — including the 5-segment minimum and the
    keep-until-beaten rule — is Phase 2. This only reports the measurement.
    """
    lengths = {player: longest_road_length(state, player) for player in state.players}
    best = max(lengths.values(), default=0)
    leaders = [player for player, length in lengths.items() if length == best]
    return (leaders[0] if len(leaders) == 1 else None), best

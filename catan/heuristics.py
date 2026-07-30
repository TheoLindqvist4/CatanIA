"""Judging positions.

`GreedyAgent` decides *what* to build and then picks *where* at random, which is why it
loses badly to a person. These functions supply the missing half: how good a spot actually
is. They are pure, take a :class:`~catan.view.PublicView`, and so can be tested directly and
reused by anything — including, later, as features or a baseline for a trained policy.

The central idea is **marginal** value. A settlement is not worth the sum of its tiles; it is
worth what those tiles add to what you already produce. A third wheat source is worth far
less than a first ore, and that single observation is most of what separates a plausible
opening from a bad one.

Everything reads only public information, because a :class:`PublicView` is all it is given.
"""

from catan.board import GENERIC_HARBOUR
from catan.resources import BANK_PER_RESOURCE, NUM_RESOURCES, Resource
from catan.state import NO_OWNER, Piece
from catan.topology import (
    NUM_VERTICES,
    ROAD_VERTICES,
    TILE_VERTICES,
    VERTEX_NEIGHBOURS,
    VERTEX_TILES,
)

#: What each resource is worth, roughly, in the 1v1 game.
#:
#: Not all equal. Wheat and ore build cities, which are the efficient route to 15 points —
#: two points each, on ground you already hold. Brick is scarce (3 tiles) and gates early
#: expansion. Sheep buys the least: settlements and development cards, nothing else.
RESOURCE_WEIGHT = {
    Resource.WOOD: 1.00,
    Resource.BRICK: 1.10,
    Resource.SHEEP: 0.80,
    Resource.WHEAT: 1.20,
    Resource.ORE: 1.15,
}

#: Diminishing returns on a resource you already produce. Lower means diversity matters
#: more: at 0.25 a first source is worth 4x, a second about 1.3x.
DIMINISH = 0.25

#: A 3:1 harbour is a modest convenience; a 2:1 is worth what you actually produce of it.
GENERIC_PORT_VALUE = 0.10
SPECIFIC_PORT_FACTOR = 0.30


def odds(number):
    """Probability of rolling ``number`` on two dice."""
    return (6 - abs(7 - number)) / 36


def tile_value(view, tile, count_robber=True):
    """What one tile is worth per turn, weighted by resource. Desert and robber are 0."""
    resource = view.board.resource_at(tile)
    if resource is None:
        return 0.0
    if count_robber and view.robber_tile == tile:
        return 0.0
    return odds(view.board.number_at(tile)) * RESOURCE_WEIGHT[resource]


def vertex_value(view, vertex, count_robber=True):
    """Raw production of a vertex, ignoring what its owner already has."""
    return sum(tile_value(view, tile, count_robber) for tile in VERTEX_TILES[vertex])


def income(view, player, count_robber=True):
    """Expected cards per turn, per resource, from everything ``player`` owns.

    Cities count double, which is why upgrading is usually the strongest move available.
    """
    out = [0.0] * NUM_RESOURCES
    for vertex in view.buildings_of(player):
        multiplier = 2 if view.vertex_piece[vertex] is Piece.CITY else 1
        for tile in VERTEX_TILES[vertex]:
            resource = view.board.resource_at(tile)
            if resource is None:
                continue
            if count_robber and view.robber_tile == tile:
                continue
            out[resource] += odds(view.board.number_at(tile)) * multiplier
    return out


def settlement_value(view, player, vertex, current_income=None):
    """What building at ``vertex`` would be worth to ``player``.

    Marginal, not absolute: each tile is discounted by how much of that resource the player
    already produces, so a spot covering three resources they lack beats a richer one
    covering a fourth wheat.
    """
    have = income(view, player) if current_income is None else current_income
    value = 0.0
    for tile in VERTEX_TILES[vertex]:
        resource = view.board.resource_at(tile)
        if resource is None:
            continue
        rate = odds(view.board.number_at(tile)) * RESOURCE_WEIGHT[resource]
        value += rate / (DIMINISH + have[resource])
    return value + port_value(view, vertex, have)


def port_value(view, vertex, have):
    """A harbour is worth what it lets you convert.

    A 2:1 is only good if you produce that resource; a 3:1 is a small general convenience.
    """
    value = 0.0
    for harbour in view.board.harbours_at(vertex):
        if harbour is GENERIC_HARBOUR:
            value += GENERIC_PORT_VALUE
        else:
            value += SPECIFIC_PORT_FACTOR * have[harbour]
    return value


def city_value(view, player, vertex):
    """Upgrading doubles a vertex's production, so its worth is that production again."""
    return vertex_value(view, vertex)


def best_settlement_spot(view, player, vertices, current_income=None):
    """The most valuable of ``vertices``, or ``None``. Ties break on the lowest id."""
    have = income(view, player) if current_income is None else current_income
    best, best_value = None, float("-inf")
    for vertex in sorted(vertices):
        value = settlement_value(view, player, vertex, have)
        if value > best_value:
            best, best_value = vertex, value
    return best


def open_spots(view):
    """Every vertex the distance rule still allows anyone to build on."""
    return [
        vertex for vertex in range(1, NUM_VERTICES + 1)
        if view.vertex_owner[vertex] == NO_OWNER
        and all(view.vertex_owner[n] == NO_OWNER for n in VERTEX_NEIGHBOURS[vertex])
    ]


def road_value(view, player, road, current_income=None):
    """What a road is worth: the best settlement spot it brings within reach.

    One step of lookahead — the spots at its far end, and the spots one road beyond those.
    A road that leads nowhere is worth nothing, which is what stops the agent laying track
    across the board for its own sake.
    """
    have = income(view, player) if current_income is None else current_income
    reachable = set()
    for endpoint in ROAD_VERTICES[road]:
        reachable.add(endpoint)
        reachable.update(VERTEX_NEIGHBOURS[endpoint])

    buildable = [
        vertex for vertex in reachable
        if view.vertex_owner[vertex] == NO_OWNER
        and all(view.vertex_owner[n] == NO_OWNER for n in VERTEX_NEIGHBOURS[vertex])
    ]
    if not buildable:
        return 0.0
    # discounted, because reaching a spot is not the same as owning it
    return 0.5 * max(settlement_value(view, player, v, have) for v in buildable)


def robber_damage(view, tile, victim):
    """What parking the robber on ``tile`` costs ``victim`` per turn."""
    resource = view.board.resource_at(tile)
    if resource is None:
        return 0.0
    rate = odds(view.board.number_at(tile)) * RESOURCE_WEIGHT[resource]
    damage = 0.0
    for vertex in TILE_VERTICES[tile]:
        if view.vertex_owner[vertex] != victim:
            continue
        damage += rate * (2 if view.vertex_piece[vertex] is Piece.CITY else 1)
    return damage


def scarcest_in_bank(view):
    """The resource players are hoarding, inferred from what the bank is missing.

    Opponent hands are hidden, but the bank is not — and every card missing from it is in
    somebody's hand. That makes an empty-ish bank slot the best public guess at what a
    Monopoly would take, without looking at anything it should not.
    """
    return min(range(NUM_RESOURCES), key=lambda r: view.bank[r])


def held_by_others(view, resource):
    """How many of ``resource`` are in hands other than the bank's. Public arithmetic."""
    return BANK_PER_RESOURCE - view.bank[resource] - view.my_hand[resource]

"""Turn a :class:`~catan.state.GameState` into a fixed-length observation vector.

Three properties matter, and each is a test rather than an intention:

**Fixed length.** :data:`SIZE` floats, always — same in every phase, every ruleset, and at
every player count. A network's input shape cannot depend on the situation.

**Perspective rotation.** ``encode(state, me)`` always puts *me* in player slot 0, with
opponents following in turn order. So one network plays every seat, and a position is
encoded identically no matter which player number happens to hold it.

**Hidden information stays hidden.** An observation contains only what that player may
actually see. Specifically it never contains:

* another player's **hand composition** — only its size, which is public because cards are
  countable
* another player's **development-card composition** — only how many they hold, and how many
  Knights they have played, both public
* the **development deck order**, or how many of each kind remain — only the count
* the **Balanced Dice deck** at all

``tests/test_encoder.py`` checks this by mutating hidden state and asserting the observation
does not move. That is a leak detector, not a documentation exercise.

Layout
------
:data:`LAYOUT` maps a name to the ``slice`` it occupies, so a consumer can pull out the
per-tile, per-vertex or per-road block and reshape it — a graph or convolutional model wants
``(19, TILE_FEATURES)`` rather than a flat run. The blocks are:

    tiles      19 x 19   resource, number, production odds, robber
    vertices   54 x 16   owner, piece, harbour, pip potential, buildability
    roads      72 x  6   owner, buildability
    players     4 x 30   hands and holdings, masked for opponents
    global          36   phase, last roll, bank, ruleset, turn bookkeeping

Every value is scaled into roughly ``[0, 1]``, using exact maxima where one exists (a
resource count cannot exceed the bank's 19) and a documented soft cap otherwise.
"""

from catan import rules
from catan.board import GENERIC_HARBOUR
from catan.dev_cards import DECK_SIZE as DEV_DECK_SIZE
from catan.dev_cards import DECK_COUNTS, ROAD_BUILDING_ROADS, DevCard
from catan.resources import BANK_PER_RESOURCE, BANK_RATE, NUM_RESOURCES, Resource, total
from catan.state import (
    MAX_CITIES,
    MAX_PLAYERS,
    MAX_ROADS,
    MAX_SETTLEMENTS,
    NO_OWNER,
    Phase,
    Piece,
)
from catan.topology import (
    NUM_ROADS,
    NUM_TILES,
    NUM_VERTICES,
    ROAD_VERTICES,
    VERTEX_NEIGHBOURS,
    VERTEX_TILES,
)

# --------------------------------------------------------------------------- #
# Scaling                                                                     #
# --------------------------------------------------------------------------- #

#: Every card of one resource, so a per-resource count cannot exceed it.
MAX_OF_ONE_RESOURCE = BANK_PER_RESOURCE
#: Every resource card in the game, so a hand total cannot exceed it.
MAX_CARDS = BANK_PER_RESOURCE * NUM_RESOURCES
#: The most Knights in the deck, so a knight count cannot exceed it.
MAX_KNIGHTS = DECK_COUNTS[DevCard.KNIGHT]

#: Turn counts are unbounded in principle; games run a few hundred turns, so this is the
#: soft cap beyond which the feature saturates.
TURN_SCALE = 400

#: Rolls run 2..12.
ROLLS = tuple(range(2, 13))

#: Harbour slots per vertex: none, generic, then one per resource.
HARBOUR_KINDS = 2 + NUM_RESOURCES


def _pips(number):
    """Ways to roll ``number`` with two dice, 0..6. 7 is the most likely."""
    return 6 - abs(7 - number)


def _odds(number):
    """Probability of ``number`` on two dice."""
    return _pips(number) / 36


# --------------------------------------------------------------------------- #
# Layout                                                                      #
# --------------------------------------------------------------------------- #

TILE_FEATURES = (
    (NUM_RESOURCES + 1)   # resource one-hot, plus desert
    + len(ROLLS)          # number one-hot
    + 1                   # production odds (0 for the desert)
    + 1                   # robber here
)

VERTEX_FEATURES = (
    (MAX_PLAYERS + 1)     # owner: empty, then one slot per player
    + 1                   # is a city (a settlement is owner set and this clear)
    + HARBOUR_KINDS       # harbour one-hot
    + 1                   # pip potential: summed odds of the adjacent tiles
    + 1                   # satisfies the distance rule
    + 1                   # reachable from my road network
)

ROAD_FEATURES = (
    (MAX_PLAYERS + 1)     # owner
    + 1                   # I could build here, cost aside
)

PLAYER_FEATURES = (
    1                     # in the game
    + 1                   # is me
    + NUM_RESOURCES       # hand composition   (me only; zeros for opponents)
    + 1                   # hand size          (public)
    + len(DevCard)        # dev-card composition (me only)
    + 1                   # dev cards held     (public)
    + 1                   # knights played     (public)
    + 1                   # public victory points
    + 1                   # true victory points (me only)
    + 1                   # holds Largest Army
    + 1                   # holds Longest Road
    + 1                   # longest road length
    + 3                   # settlements, cities, roads left
    + NUM_RESOURCES       # bank trade rates
    + 1                   # discards still owed
)

GLOBAL_FEATURES = (
    len(Phase)            # phase one-hot
    + len(ROLLS) + 1      # last roll one-hot, plus "not rolled yet"
    + 1                   # turn number
    + NUM_RESOURCES       # the bank
    + 1                   # development cards left in the deck
    + 1                   # free roads owed
    + 1                   # a development card was played this turn
    + 1                   # the dice have been rolled this turn
    + 1                   # it is my decision right now
    + 1                   # players in the game
    + 4                   # ruleset: win target, hand limit, friendly robber, balanced dice
)


def _build_layout():
    spans, offset = {}, 0
    for name, width in (
        ("tiles", NUM_TILES * TILE_FEATURES),
        ("vertices", NUM_VERTICES * VERTEX_FEATURES),
        ("roads", NUM_ROADS * ROAD_FEATURES),
        ("players", MAX_PLAYERS * PLAYER_FEATURES),
        ("global", GLOBAL_FEATURES),
    ):
        spans[name] = slice(offset, offset + width)
        offset += width
    return spans, offset


LAYOUT, SIZE = _build_layout()

#: Rows and columns of each repeated block, for reshaping.
SHAPES = {
    "tiles": (NUM_TILES, TILE_FEATURES),
    "vertices": (NUM_VERTICES, VERTEX_FEATURES),
    "roads": (NUM_ROADS, ROAD_FEATURES),
    "players": (MAX_PLAYERS, PLAYER_FEATURES),
}


# --------------------------------------------------------------------------- #
# Perspective                                                                 #
# --------------------------------------------------------------------------- #

def player_slots(state, me):
    """``{player: slot}`` with ``me`` at 0 and the others in turn order after.

    Rotating rather than using raw player numbers is what lets one network play every
    seat: the same position always encodes the same way.
    """
    order = state.player_order
    start = order.index(me)
    rotated = order[start:] + order[:start]
    return {player: slot for slot, player in enumerate(rotated)}


# --------------------------------------------------------------------------- #
# The board-static half of an observation                                     #
# --------------------------------------------------------------------------- #

def _static_template(board):
    """The parts of an observation that depend on the *layout* and never on play.

    Which resource sits on a tile, its number token, its odds, which harbours a vertex can
    reach, and the pip potential of a vertex are all fixed the moment the board is generated.
    Recomputing them on every encode is most of what encoding costs — profiling a training
    rollout put ``_encode_vertices`` at 45% of the total, nearly all of it in one generator
    expression summing three tiles' odds for a board that had not changed in 14,000 calls.

    So it is computed once per :class:`~catan.board.Board` and cached on it. The board is
    immutable and shared across clones, so one template serves an entire training run.
    """
    template = board.__dict__.get("_observation_template")
    if template is not None:
        return template

    out = [0.0] * SIZE

    base = LAYOUT["tiles"].start
    for tile in range(1, NUM_TILES + 1):
        at = base + (tile - 1) * TILE_FEATURES
        resource = board.resource_at(tile)
        number = board.number_at(tile)

        out[at + (NUM_RESOURCES if resource is None else int(resource))] = 1.0
        at += NUM_RESOURCES + 1
        out[at + ROLLS.index(number)] = 1.0
        at += len(ROLLS)
        # a desert never pays out, whatever its token says
        out[at] = 0.0 if resource is None else _odds(number)
        # at + 1 is the robber flag, which moves — left at 0 for the caller

    base = LAYOUT["vertices"].start
    for vertex in range(1, NUM_VERTICES + 1):
        # owner one-hot and the city flag are play, not layout; skip to the harbours
        at = base + (vertex - 1) * VERTEX_FEATURES + (MAX_PLAYERS + 1) + 1

        harbours = board.harbours_at(vertex)
        if not harbours:
            out[at] = 1.0
        else:
            for harbour in harbours:
                out[at + (1 if harbour is GENERIC_HARBOUR else 2 + int(harbour))] = 1.0
        at += HARBOUR_KINDS

        # the classic settlement heuristic: how often this spot pays out at all.
        # The 0.0 start matters: a corner touching only the desert sums an empty
        # generator, and bare sum() would return int 0 into a float vector.
        out[at] = sum(
            (
                _odds(board.number_at(tile))
                for tile in VERTEX_TILES[vertex]
                if board.resource_at(tile) is not None
            ),
            0.0,
        )

    board.__dict__["_observation_template"] = out
    return out


# --------------------------------------------------------------------------- #
# Encoding                                                                    #
# --------------------------------------------------------------------------- #

def encode(state, me=None):
    """The observation ``me`` is entitled to, as a list of :data:`SIZE` floats.

    Args:
        state: the game.
        me: whose view to build. Defaults to ``state.current_player`` — but pass it
            explicitly in search, because during a discard the current player may be an
            opponent.
    """
    if me is None:
        me = state.current_player
    if me not in state.players:
        raise ValueError(f"player must be in 1..{state.num_players}, got {me}")

    out = _static_template(state.board).copy()
    slots = player_slots(state, me)

    _encode_tiles(state, out)
    _encode_vertices(state, out, me, slots)
    _encode_roads(state, out, me, slots)
    _encode_players(state, out, me, slots)
    _encode_global(state, out, me)
    return out


def _encode_tiles(state, out):
    """Only the robber moves; everything else about a tile is in the static template."""
    at = LAYOUT["tiles"].start + (state.robber_tile - 1) * TILE_FEATURES
    out[at + NUM_RESOURCES + 1 + len(ROLLS) + 1] = 1.0


def _encode_vertices(state, out, me, slots):
    """Ownership and buildability. Harbours and pip potential come from the template.

    The two buildability flags are derived in one pass over what is *owned* rather than by
    asking :func:`catan.rules.respects_distance_rule` and
    :func:`catan.rules.touches_own_road` per vertex. Those were 108 calls per encode and
    38% of what encoding cost after the static template; there are at most ten settlements
    and fifteen roads to walk instead of fifty-four vertices to interrogate.

    The rules remain the authority — ``test_buildability_flags_agree_with_the_rules``
    cross-checks every vertex of every board against them, which is what keeps this
    shortcut honest.
    """
    base = LAYOUT["vertices"].start
    owners = state.vertex_owner
    pieces = state.vertex_piece

    blocked = set()
    for vertex in range(1, NUM_VERTICES + 1):
        if owners[vertex] != NO_OWNER:
            blocked.add(vertex)
            blocked.update(VERTEX_NEIGHBOURS[vertex])

    my_junctions = set()
    for road in range(1, NUM_ROADS + 1):
        if state.edge_owner[road] == me:
            my_junctions.update(ROAD_VERTICES[road])

    # offset of the two buildability flags within a vertex block, past the harbour
    # one-hot and the pip potential
    flags = (MAX_PLAYERS + 1) + 1 + HARBOUR_KINDS + 1

    for vertex in range(1, NUM_VERTICES + 1):
        at = base + (vertex - 1) * VERTEX_FEATURES
        owner = owners[vertex]

        out[at + (0 if owner == NO_OWNER else 1 + slots[owner])] = 1.0
        if pieces[vertex] is Piece.CITY:
            out[at + MAX_PLAYERS + 1] = 1.0

        at += flags
        if vertex not in blocked:
            out[at] = 1.0
        if vertex in my_junctions:
            out[at + 1] = 1.0


def _reachable_vertices(state, me):
    """Vertices ``me`` can build outward from.

    A junction works if I have a building on it, or it is empty and one of my roads meets
    it — the same rule as :func:`catan.rules.is_road_connected`, computed once for the
    whole board instead of per road. ``test_buildability_flags_agree_with_the_rules``
    checks the two agree, which is what keeps this shortcut honest.
    """
    touching = set()
    for road in range(1, NUM_ROADS + 1):
        if state.edge_owner[road] == me:
            touching.update(ROAD_VERTICES[road])

    owner = state.vertex_owner
    return {
        vertex for vertex in range(1, NUM_VERTICES + 1)
        if owner[vertex] == me or (owner[vertex] == NO_OWNER and vertex in touching)
    }


def _encode_roads(state, out, me, slots):
    base = LAYOUT["roads"].start
    # Cost aside, so the feature says "reachable" rather than "affordable right now" —
    # affordability is already visible from my hand.
    reachable = _reachable_vertices(state, me)
    for road in range(1, NUM_ROADS + 1):
        at = base + (road - 1) * ROAD_FEATURES
        owner = state.edge_owner[road]
        out[at + (0 if owner == NO_OWNER else 1 + slots[owner])] = 1.0
        at += MAX_PLAYERS + 1
        out[at] = 1.0 if (
            owner == NO_OWNER
            and not reachable.isdisjoint(ROAD_VERTICES[road])
        ) else 0.0


def _encode_players(state, out, me, slots):
    base = LAYOUT["players"].start
    for player, slot in slots.items():
        at = base + slot * PLAYER_FEATURES
        mine = player == me
        hand = state.hands[player]
        held = state.dev_cards[player]

        out[at] = 1.0
        at += 1
        out[at] = 1.0 if mine else 0.0
        at += 1

        # composition is mine alone; the size is public because cards are countable
        if mine:
            for resource in range(NUM_RESOURCES):
                out[at + resource] = hand[resource] / MAX_OF_ONE_RESOURCE
        at += NUM_RESOURCES
        out[at] = total(hand) / MAX_CARDS
        at += 1

        if mine:
            for card in range(len(DevCard)):
                out[at + card] = held[card] / DEV_DECK_SIZE
        at += len(DevCard)
        out[at] = sum(held) / DEV_DECK_SIZE
        at += 1

        out[at] = state.knights_played[player] / MAX_KNIGHTS
        at += 1

        target = state.ruleset.victory_points_to_win
        out[at] = min(rules.public_victory_points(state, player) / target, 1.0)
        at += 1
        # only I know my hidden Victory Point cards, so only I see my true total
        out[at] = min(rules.victory_points(state, player) / target, 1.0) if mine else 0.0
        at += 1

        out[at] = 1.0 if state.largest_army_holder == player else 0.0
        at += 1
        out[at] = 1.0 if state.longest_road_holder == player else 0.0
        at += 1
        out[at] = rules.longest_road_length(state, player) / MAX_ROADS
        at += 1

        out[at] = state.settlements_left[player] / MAX_SETTLEMENTS
        out[at + 1] = state.cities_left[player] / MAX_CITIES
        out[at + 2] = state.roads_left[player] / MAX_ROADS
        at += 3

        for resource, rate in enumerate(rules.trade_rates(state, player)):
            out[at + resource] = rate / BANK_RATE
        at += NUM_RESOURCES

        out[at] = state.discards_owed[player] / MAX_OF_ONE_RESOURCE


def _encode_global(state, out, me):
    at = LAYOUT["global"].start

    out[at + int(state.phase)] = 1.0
    at += len(Phase)

    if state.last_roll is None:
        out[at + len(ROLLS)] = 1.0
    else:
        out[at + ROLLS.index(state.last_roll)] = 1.0
    at += len(ROLLS) + 1

    out[at] = min(state.turn_number / TURN_SCALE, 1.0)
    at += 1

    for resource in range(NUM_RESOURCES):
        out[at + resource] = state.bank[resource] / MAX_OF_ONE_RESOURCE
    at += NUM_RESOURCES

    # how many cards are left is public; which cards they are is not
    out[at] = len(state.dev_deck) / DEV_DECK_SIZE
    at += 1
    out[at] = state.free_roads / ROAD_BUILDING_ROADS
    at += 1
    out[at] = 1.0 if state.dev_card_played_this_turn else 0.0
    at += 1
    out[at] = 1.0 if state.rolled_this_turn else 0.0
    at += 1
    # during a discard the decision may belong to an opponent
    out[at] = 1.0 if state.current_player == me else 0.0
    at += 1
    out[at] = state.num_players / MAX_PLAYERS
    at += 1

    ruleset = state.ruleset
    out[at] = ruleset.victory_points_to_win / 20
    out[at + 1] = ruleset.hand_limit / 20
    out[at + 2] = 1.0 if ruleset.friendly_robber else 0.0
    out[at + 3] = 1.0 if ruleset.balanced_dice else 0.0


# --------------------------------------------------------------------------- #
# Introspection                                                               #
# --------------------------------------------------------------------------- #

def block(observation, name):
    """One named block of an observation, as a list of rows.

    ``block(obs, "tiles")[3]`` is tile 4's features.
    """
    values = observation[LAYOUT[name]]
    if name not in SHAPES:
        return values
    rows, width = SHAPES[name]
    return [values[i * width:(i + 1) * width] for i in range(rows)]


def _validate():
    """Layout consistency, at import."""
    assert LAYOUT["global"].stop == SIZE
    covered = 0
    for name in ("tiles", "vertices", "roads", "players", "global"):
        assert LAYOUT[name].start == covered, f"{name} does not follow the previous block"
        covered = LAYOUT[name].stop
    assert covered == SIZE
    for name, (rows, width) in SHAPES.items():
        assert LAYOUT[name].stop - LAYOUT[name].start == rows * width


_validate()

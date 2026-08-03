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

from catan import dice, resources
from catan.events import Award, Event, EventKind
from catan.actions import (
    DEV_CARD_PLAYS,
    PRE_ROLL_PLAYS,
    Action,
    ActionType,
    build_city,
    build_road,
    build_settlement,
    buy_dev_card,
    discard,
    end_turn,
    move_robber,
    play_knight,
    play_monopoly,
    play_road_building,
    play_year_of_plenty,
    roll,
    trade_with_bank,
)
from catan.board import GENERIC_HARBOUR, ROBBER_ROLL
from catan.dev_cards import (
    AWARD_VICTORY_POINTS,
    LARGEST_ARMY_MINIMUM,
    LONGEST_ROAD_MINIMUM,
    PLAYABLE,
    ROAD_BUILDING_ROADS,
    DevCard,
)
from catan.resources import (
    BANK_RATE,
    CITY_COST,
    DEV_CARD_COST,
    GENERIC_HARBOUR_RATE,
    NUM_RESOURCES,
    ROAD_COST,
    SETTLEMENT_COST,
    SPECIFIC_HARBOUR_RATE,
)
from catan.state import (
    NO_OWNER,
    PIECE_VICTORY_POINTS,
    PIECE_YIELD,
    Phase,
    Piece,
)
from catan.topology import (
    NUM_ROADS,
    NUM_TILES,
    NUM_VERTICES,
    ROAD_VERTICES,
    TILE_VERTICES,
    VERTEX_NEIGHBOURS,
    VERTEX_ROADS,
)

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

    While Road Building is in effect (``state.free_roads``) the cost is waived.
    """
    if not state.is_road_free(road):
        return False
    if state.roads_left[player] <= 0:
        return False
    if state.free_roads <= 0 and not resources.can_afford(state.hands[player], ROAD_COST):
        return False
    return is_road_connected(state, player, road)


def is_road_connected(state, player, road):
    """Whether ``road`` touches ``player``'s network, ignoring cost and pieces.

    The reachability half of :func:`can_build_road`, so callers that only want "could this
    ever be mine" — the observation encoder, for one — do not have to know about payment.
    """
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
# TRADING WITH THE BANK                                                       #
# --------------------------------------------------------------------------- #

def trade_rates(state, player):
    """How many of each resource ``player`` must give the bank for one card back.

    4 by default, 3 with any generic harbour, 2 with the matching specific harbour. A
    harbour is granted by owning a building on either endpoint of its coastal road.
    """
    rates = [BANK_RATE] * NUM_RESOURCES
    for vertex in state.buildings_of(player):
        for harbour in state.board.harbours_at(vertex):
            if harbour is GENERIC_HARBOUR:
                for resource in range(NUM_RESOURCES):
                    rates[resource] = min(rates[resource], GENERIC_HARBOUR_RATE)
            else:
                rates[harbour] = min(rates[harbour], SPECIFIC_HARBOUR_RATE)
    return rates


def can_trade_with_bank(state, player, give, take, rates=None):
    """Whether ``player`` may exchange ``give`` for one ``take`` at their best rate."""
    if give == take:
        return False
    if not (0 <= give < NUM_RESOURCES and 0 <= take < NUM_RESOURCES):
        return False
    if state.bank[take] < 1:
        return False
    rate = (rates or trade_rates(state, player))[give]
    return state.hands[player][give] >= rate


# --------------------------------------------------------------------------- #
# DEVELOPMENT CARDS                                                           #
# --------------------------------------------------------------------------- #

def can_buy_dev_card(state, player):
    """A card must be left in the deck, and the player must afford it."""
    if not state.dev_deck:
        return False
    return resources.can_afford(state.hands[player], DEV_CARD_COST)


def playable_dev_cards(state, player):
    """Cards ``player`` may play right now, as counts indexed by :class:`DevCard`.

    Empty once a card has been played this turn. A card bought this turn is excluded —
    ``dev_cards_new`` tracks those, and they become playable when the turn ends. Victory
    Point cards never appear: they are never played, only held.
    """
    if state.dev_card_played_this_turn:
        return [0] * len(state.dev_cards[player])
    held, fresh = state.dev_cards[player], state.dev_cards_new[player]
    return [
        held[card] - fresh[card] if card in PLAYABLE else 0
        for card in range(len(held))
    ]


def can_play_dev_card(state, player, card):
    """Whether ``player`` holds a playable ``card``, ignoring its own preconditions."""
    if not 0 <= card < len(state.dev_cards[player]):
        return False
    return playable_dev_cards(state, player)[card] > 0


def can_play_year_of_plenty(state, player, first, second):
    """Both resources must be drawable from the bank — two of the same needs two left.

    The pair must be in ascending order. Ore-then-wheat and wheat-then-ore are the same
    move, so only one of them is a legal action; :func:`catan.actions.play_year_of_plenty`
    sorts for you.
    """
    if not can_play_dev_card(state, player, DevCard.YEAR_OF_PLENTY):
        return False
    if not (0 <= first < NUM_RESOURCES and 0 <= second < NUM_RESOURCES):
        return False
    if first > second:
        return False
    needed = [0] * NUM_RESOURCES
    needed[first] += 1
    needed[second] += 1
    return all(state.bank[r] >= needed[r] for r in range(NUM_RESOURCES))


def can_play_monopoly(state, player, resource):
    return (
        can_play_dev_card(state, player, DevCard.MONOPOLY)
        and 0 <= resource < NUM_RESOURCES
    )


def can_play_road_building(state, player):
    """Needs a road piece left and somewhere legal to put one.

    Checked with the cost waived, since the card is what pays.
    """
    if not can_play_dev_card(state, player, DevCard.ROAD_BUILDING):
        return False
    if state.roads_left[player] <= 0:
        return False
    state.free_roads += 1  # probe as if the card were already in effect
    try:
        return any(
            can_build_road(state, player, road)
            for road in range(1, NUM_ROADS + 1)
        )
    finally:
        state.free_roads -= 1


def dev_card_actions(state, player):
    """Every development-card play available to ``player`` right now."""
    actions = []
    if can_play_dev_card(state, player, DevCard.KNIGHT):
        actions.append(play_knight())
    if can_play_road_building(state, player):
        actions.append(play_road_building())
    if can_play_dev_card(state, player, DevCard.YEAR_OF_PLENTY):
        actions += [
            play_year_of_plenty(first, second)
            for first in range(NUM_RESOURCES)
            for second in range(first, NUM_RESOURCES)
            if can_play_year_of_plenty(state, player, first, second)
        ]
    if can_play_dev_card(state, player, DevCard.MONOPOLY):
        actions += [play_monopoly(r) for r in range(NUM_RESOURCES)]
    return actions


# --------------------------------------------------------------------------- #
# AWARDS: LARGEST ARMY AND LONGEST ROAD                                       #
# --------------------------------------------------------------------------- #

def _update_award(current, scores, minimum):
    """Who holds an award after ``scores`` change.

    Both awards work the same way: you need at least ``minimum``, you need to be the
    sole leader to take it from someone, and the holder keeps it on a tie. If the holder
    drops behind and the new best is tied, nobody holds it until someone is clearly
    ahead.
    """
    holder = current
    if holder is not None and scores[holder] < minimum:
        holder = None

    best = max(scores.values(), default=0)
    if best < minimum:
        return None
    if holder is not None and scores[holder] >= best:
        return holder

    leaders = sorted(p for p, score in scores.items() if score == best)
    return leaders[0] if len(leaders) == 1 else None


def update_awards(state):
    """Recompute both award holders.

    Called after anything that can change them — building a road, but also building a
    settlement or city, which can *break* an opponent's road and take Longest Road off
    them.

    A change of holder is worth 2 points and is the sort of thing a player must be told
    about, so it is announced.
    """
    _set_award(
        state, Award.LARGEST_ARMY, "largest_army_holder",
        {p: state.knights_played[p] for p in state.players},
        LARGEST_ARMY_MINIMUM,
    )
    _set_award(
        state, Award.LONGEST_ROAD, "longest_road_holder",
        longest_road_lengths(state),
        LONGEST_ROAD_MINIMUM,
    )


def _set_award(state, award, attribute, scores, minimum):
    before = getattr(state, attribute)
    after = _update_award(before, scores, minimum)
    if after == before:
        return
    setattr(state, attribute, after)
    if before is not None:
        state.events.append(Event(EventKind.AWARD, before, position=int(award), amount=0))
    if after is not None:
        state.events.append(Event(EventKind.AWARD, after, position=int(award), amount=1))


# --------------------------------------------------------------------------- #
# THE ROBBER                                                                  #
# --------------------------------------------------------------------------- #

def discard_count(state, player):
    """How many cards ``player`` gives up on a 7: half the hand, rounded down.

    Holding 9 means discarding 4 and keeping 5 — half *rounded down* is what you lose,
    so an odd hand keeps the extra card.

    Read once, when the 7 is rolled, and stored in ``state.discards_owed``. Recomputing
    it as the hand shrinks would move the target and stop the discards early.
    """
    return resources.total(state.hands[player]) // 2


def must_discard(state, player):
    """Whether ``player``'s hand is over the limit, i.e. a 7 costs them cards.

    The limit comes from the ruleset: 7 in the base game, 9 in ranked 1v1.
    """
    return resources.total(state.hands[player]) > state.ruleset.hand_limit


def is_robber_protected(state, player):
    """Whether Friendly Robber shields ``player`` from being robbed or blocked.

    Ranked 1v1 protects anyone at or below 2 **public** victory points — settlements,
    cities and the two awards count; hidden Victory Point cards do not. It gives whoever
    is behind early some breathing room.
    """
    if not state.ruleset.friendly_robber:
        return False
    return public_victory_points(state, player) <= state.ruleset.friendly_robber_threshold


def owes_discard(state, player):
    """Whether ``player`` still has cards to give up for the current 7."""
    return state.discards_owed[player] > 0


def occupants_of(state, tile, excluding=NO_OWNER):
    """Players with a building on ``tile``, other than ``excluding``."""
    return tuple(sorted({
        state.vertex_owner[vertex]
        for vertex in TILE_VERTICES[tile]
        if state.vertex_owner[vertex] not in (NO_OWNER, excluding)
    }))


def victims_at(state, tile, robber):
    """Players ``robber`` could steal from if the robber went to ``tile``.

    Anyone with a building on the tile who holds at least one card and is not the robber
    themselves — stealing from an empty hand is not a choice the rules offer — and, under
    Friendly Robber, is not protected.
    """
    return tuple(
        player for player in occupants_of(state, tile, excluding=robber)
        if resources.total(state.hands[player]) > 0
        and not is_robber_protected(state, player)
    )


def robber_destinations(state, player):
    """Tiles the robber may be moved to.

    Never the tile it is already on — it must move. Under Friendly Robber, never a tile
    where a *protected* opponent has a building: they cannot be blocked either, not just
    not robbed.
    """
    candidates = [
        tile for tile in range(1, NUM_TILES + 1) if tile != state.robber_tile
    ]
    if not state.ruleset.friendly_robber:
        return candidates

    allowed = [
        tile for tile in candidates
        if not any(
            is_robber_protected(state, occupant)
            for occupant in occupants_of(state, tile, excluding=player)
        )
    ]
    # Unreachable in practice: a protected player holds at most 2 public points, so at
    # most two buildings, so at most six blocked tiles out of eighteen. Guarded anyway,
    # because an empty list here would deadlock the game rather than fail loudly.
    return allowed or candidates


def can_move_robber(state, player, tile, victim):
    """Whether the robber may go to ``tile`` and rob ``victim`` (0 for nobody).

    ``victim`` of 0 is only legal when the tile offers nobody to rob.
    """
    if not 1 <= tile <= NUM_TILES:
        return False
    if tile not in robber_destinations(state, player):
        return False
    options = victims_at(state, tile, player)
    return victim in options if options else victim == 0


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

    if state.phase is Phase.DISCARD:
        # One card at a time, so the choice stays a small discrete action.
        return [
            discard(resource)
            for resource in range(NUM_RESOURCES)
            if state.hands[player][resource] > 0
        ]

    if state.phase is Phase.MOVE_ROBBER:
        actions = []
        for tile in robber_destinations(state, player):
            options = victims_at(state, tile, player)
            actions += [move_robber(tile, victim) for victim in (options or (0,))]
        return actions

    if state.phase is Phase.ROLL:
        # Only a Knight may be played *before* rolling, to block a tile before it
        # produces. See :data:`catan.actions.PRE_ROLL_PLAYS` for why the other three are
        # held back until the dice are down.
        #
        # Playing one is **optional**, so whenever a card is available the choice to
        # decline has to be available too. Without it, a player holding a Knight is forced
        # to play it every turn, which is not the game.
        #
        # When no card can be played there is no decision at all, and the list stays empty
        # so the environment rolls by itself rather than asking for a click that has only
        # one answer.
        cards = [
            action for action in dev_card_actions(state, player)
            if action.type in PRE_ROLL_PLAYS
        ]
        return cards + [roll()] if cards else []

    if state.phase is Phase.BUILD:
        actions = [end_turn()]
        hand = state.hands[player]

        # The cheap gates first. Scanning 72 roads or 54 vertices to discover the player
        # cannot afford any of them is the common case and pure waste — an empty hand is
        # far more frequent than a full one. The per-action predicates below are still the
        # authority; this only skips loops that provably cannot yield anything.
        if state.roads_left[player] > 0 and (
            state.free_roads > 0 or resources.can_afford(hand, ROAD_COST)
        ):
            actions += [
                build_road(r)
                for r in range(1, NUM_ROADS + 1)
                if can_build_road(state, player, r)
            ]
        if state.settlements_left[player] > 0 and resources.can_afford(
            hand, SETTLEMENT_COST
        ):
            actions += [
                build_settlement(v)
                for v in range(1, NUM_VERTICES + 1)
                if can_build_settlement(state, player, v)
            ]
        if state.cities_left[player] > 0 and resources.can_afford(hand, CITY_COST):
            actions += [
                build_city(v)
                for v in range(1, NUM_VERTICES + 1)
                if can_build_city(state, player, v)
            ]
        rates = trade_rates(state, player)  # computed once for all 20 candidates
        actions += [
            trade_with_bank(give, take)
            for give in range(NUM_RESOURCES)
            for take in range(NUM_RESOURCES)
            if can_trade_with_bank(state, player, give, take, rates)
        ]
        if can_buy_dev_card(state, player):
            actions.append(buy_dev_card())
        actions += dev_card_actions(state, player)
        return actions

    return []  # GAME_OVER


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
    elif state.phase is Phase.DISCARD:
        _apply_discard(state, player, action)
    elif state.phase is Phase.MOVE_ROBBER:
        _apply_move_robber(state, player, action)
    elif state.phase is Phase.BUILD:
        _apply_build(state, player, action)
    elif state.phase is Phase.ROLL:
        if action.type is ActionType.ROLL:
            roll_dice(state)
        elif action.type in PRE_ROLL_PLAYS:
            _apply_dev_card(state, player, action)
        else:
            # Includes the three card plays that are legal this turn but not yet: the
            # rejection has to live here as well as in `legal_actions`, because `apply` is
            # the authority and a search or a hand-built action never consults the list.
            raise IllegalAction(f"must roll the dice before {action!r}")
    else:
        raise IllegalAction("the game is over")

    return state


def _apply_discard(state, player, action):
    if action.type is not ActionType.DISCARD:
        raise IllegalAction(f"must discard a card, got {action!r}")
    resource = action.position
    if not 0 <= resource < NUM_RESOURCES:
        raise IllegalAction(f"not a resource: {action!r}")
    if state.hands[player][resource] <= 0:
        raise IllegalAction(f"no {resources.Resource(resource).name.lower()} to discard")

    state.hands[player][resource] -= 1
    state.bank[resource] += 1
    state.discards_owed[player] -= 1
    state.events.append(Event(EventKind.DISCARDED, player, resource=resource))
    _advance_after_discards(state)


def _advance_after_discards(state):
    """Drop anyone who has finished discarding, then move on to the robber."""
    while state.pending_discards and not owes_discard(state, state.pending_discards[0]):
        state.pending_discards.pop(0)
    state.phase = Phase.DISCARD if state.pending_discards else Phase.MOVE_ROBBER


def _apply_move_robber(state, player, action):
    if action.type is not ActionType.MOVE_ROBBER:
        raise IllegalAction(f"must move the robber, got {action!r}")
    tile, victim = action.position, action.extra
    if not can_move_robber(state, player, tile, victim):
        raise IllegalAction(f"cannot {action!r}")

    state.robber_tile = tile
    state.events.append(Event(EventKind.ROBBER_MOVED, player, position=tile))
    if victim:
        _steal_one_card(state, player, victim)

    # A Knight can send us here before the dice are rolled, in which case the player
    # still has a roll coming.
    state.phase = Phase.BUILD if state.rolled_this_turn else Phase.ROLL


def _steal_one_card(state, thief, victim):
    """Take one card at random from ``victim``.

    Drawn uniformly over their *cards*, not their resource types, so a hand of five wood
    and one ore gives up wood five times out of six.
    """
    hand = state.hands[victim]
    total = resources.total(hand)
    if total <= 0:
        return
    pick = state.rng.randrange(total)
    for resource, count in enumerate(hand):
        if pick < count:
            hand[resource] -= 1
            state.hands[thief][resource] += 1
            state.events.append(
                Event(EventKind.STOLE, thief, resource=resource, other=victim))
            return
        pick -= count


def _apply_setup_settlement(state, player, action):
    if action.type is not ActionType.BUILD_SETTLEMENT:
        raise IllegalAction(f"setup expects a settlement, got {action!r}")
    vertex = _check_vertex(action.position)
    if not can_place_setup_settlement(state, vertex):
        raise IllegalAction(f"cannot place a settlement at {vertex}")

    _put_building(state, player, vertex, Piece.SETTLEMENT)
    state.last_settlement = vertex

    # The second settlement pays out its adjacent tiles immediately. Drawn from the
    # bank like any other production — at most 3 cards from a full bank, so the
    # shortage rule cannot bite here.
    if state.setup_round == 2:
        for resource in state.board.resources_at(vertex):
            state.hands[player][resource] += 1
            state.bank[resource] -= 1

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


def _apply_dev_card(state, player, action):
    """Play one development card. Valid in both ROLL and BUILD."""
    if action.type is ActionType.PLAY_KNIGHT:
        if not can_play_dev_card(state, player, DevCard.KNIGHT):
            raise IllegalAction("no Knight available to play")
        _spend_dev_card(state, player, DevCard.KNIGHT)
        state.knights_played[player] += 1
        update_awards(state)
        state.phase = Phase.MOVE_ROBBER
        # The robber move itself can win the game via Largest Army.
        _check_for_winner(state, player)
        return

    if action.type is ActionType.PLAY_ROAD_BUILDING:
        if not can_play_road_building(state, player):
            raise IllegalAction("cannot play Road Building")
        _spend_dev_card(state, player, DevCard.ROAD_BUILDING)
        # Granted as credit rather than forced placements, so the player may interleave
        # other actions. Unused credit lapses at the end of the turn.
        state.free_roads += ROAD_BUILDING_ROADS
        return

    if action.type is ActionType.PLAY_YEAR_OF_PLENTY:
        first, second = action.position, action.extra
        if not can_play_year_of_plenty(state, player, first, second):
            raise IllegalAction(f"cannot {action!r}")
        _spend_dev_card(state, player, DevCard.YEAR_OF_PLENTY)
        for resource in (first, second):
            state.hands[player][resource] += 1
            state.bank[resource] -= 1
        return

    if action.type is ActionType.PLAY_MONOPOLY:
        resource = action.position
        if not can_play_monopoly(state, player, resource):
            raise IllegalAction(f"cannot {action!r}")
        _spend_dev_card(state, player, DevCard.MONOPOLY)
        haul = 0
        for other in state.players:
            if other == player:
                continue
            taken = state.hands[other][resource]
            state.hands[other][resource] = 0
            state.hands[player][resource] += taken
            haul += taken
        state.events.append(
            Event(EventKind.MONOPOLISED, player, resource=resource, amount=haul))
        return

    raise IllegalAction(f"not a development card play: {action!r}")


def _spend_dev_card(state, player, card):
    state.dev_cards[player][card] -= 1
    state.dev_card_played_this_turn = True
    state.events.append(Event(EventKind.PLAYED_DEV, player, position=int(card)))


def _apply_build(state, player, action):
    if action.type is ActionType.END_TURN:
        state.events.append(Event(EventKind.TURN_ENDED, player))
        state.turn_number += 1
        state.phase = Phase.ROLL
        state.dev_card_played_this_turn = False
        state.rolled_this_turn = False
        state.free_roads = 0  # unused Road Building credit lapses
        # Cards bought this turn become playable now.
        state.dev_cards_new[player] = [0] * len(state.dev_cards_new[player])
        return

    if action.type in DEV_CARD_PLAYS:
        _apply_dev_card(state, player, action)
        return

    if action.type is ActionType.BUY_DEV_CARD:
        if not can_buy_dev_card(state, player):
            raise IllegalAction("cannot buy a development card")
        _pay(state, player, DEV_CARD_COST)
        card = state.dev_deck.pop()
        state.dev_cards[player][card] += 1
        state.dev_cards_new[player][card] += 1
        state.dev_bought[player] += 1
        state.events.append(Event(EventKind.BOUGHT_DEV, player))
        # A Victory Point card can win the game the moment it is drawn.
        _check_for_winner(state, player)
        return

    if action.type is ActionType.TRADE_WITH_BANK:
        give, take = action.position, action.extra
        if not can_trade_with_bank(state, player, give, take):
            raise IllegalAction(f"cannot {action!r}")
        rate = trade_rates(state, player)[give]
        hand = state.hands[player]
        hand[give] -= rate
        state.bank[give] += rate
        state.bank[take] -= 1
        hand[take] += 1
        state.events.append(
            Event(EventKind.TRADED, player, resource=give, amount=rate, other=take))
        return  # a trade cannot win the game

    if action.type is ActionType.BUILD_ROAD:
        road = _check_road(action.position)
        if not can_build_road(state, player, road):
            raise IllegalAction(f"cannot build a road at {road}")
        if state.free_roads > 0:
            state.free_roads -= 1  # paid for by Road Building
        else:
            _pay(state, player, ROAD_COST)
        _put_road(state, player, road)

    elif action.type is ActionType.BUILD_SETTLEMENT:
        vertex = _check_vertex(action.position)
        if not can_build_settlement(state, player, vertex):
            raise IllegalAction(f"cannot build a settlement at {vertex}")
        _pay(state, player, SETTLEMENT_COST)
        _put_building(state, player, vertex, Piece.SETTLEMENT)

    elif action.type is ActionType.BUILD_CITY:
        vertex = _check_vertex(action.position)
        if not can_build_city(state, player, vertex):
            raise IllegalAction(f"cannot build a city at {vertex}")
        _pay(state, player, CITY_COST)
        state.vertex_piece[vertex] = Piece.CITY
        state.cities_left[player] -= 1
        state.settlements_left[player] += 1  # the settlement returns to the supply
        state.events.append(
            Event(EventKind.BUILT, player, position=vertex, other=int(Piece.CITY)))

    else:
        raise IllegalAction(f"unknown action {action!r}")

    # Any build can change Longest Road — a road by extending one, a settlement or city
    # by *breaking* an opponent's.
    update_awards(state)
    _check_for_winner(state, player)


def _pay(state, player, cost):
    """Spend ``cost``, returning the cards to the bank.

    Cards are conserved: every card is either in the bank or in a hand. Without the
    refund the bank would drain and production would stop.
    """
    resources.pay(state.hands[player], cost)
    for resource, amount in enumerate(cost):
        state.bank[resource] += amount
        # everyone watching sees what a purchase costs; see GameState's public record
        state.spent[player][resource] += amount


def _put_building(state, player, vertex, piece):
    state.vertex_owner[vertex] = player
    state.vertex_piece[vertex] = piece
    state.last_build_turn[player] = state.turn_number
    state.settlements_left[player] -= 1
    state.events.append(
        Event(EventKind.BUILT, player, position=vertex, other=int(piece)))


def _put_road(state, player, road):
    state.edge_owner[road] = player
    state.roads_left[player] -= 1
    state.last_build_turn[player] = state.turn_number
    state.events.append(Event(EventKind.BUILT, player, position=road, other=0))


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
    """Roll 2d6, resolve the result, and hand control to the next decision.

    Two independent dice, not one uniform draw over 2..12 — the triangular
    distribution is the whole point of Catan's probabilities.

    A 7 pays nobody. Instead everyone over the hand limit discards down to half
    (:attr:`Phase.DISCARD`, in turn order starting from the roller), and then the roller
    moves the robber (:attr:`Phase.MOVE_ROBBER`). Anything else pays production and goes
    straight to :attr:`Phase.BUILD`.

    Returns:
        int: the roll.
    """
    if state.phase is not Phase.ROLL:
        raise IllegalAction(f"cannot roll during {state.phase.name}")

    if state.ruleset.balanced_dice:
        first, second = dice.draw_balanced(state)
    else:
        first, second = dice.roll_plain(state.rng)

    roll = first + second
    state.last_roll = roll
    state.roll_counts[roll] += 1
    state.rolled_this_turn = True
    state.events.append(Event(EventKind.ROLLED, state.turn_player, amount=roll))

    if roll == ROBBER_ROLL:
        begin_robber(state)
    else:
        distribute(state, roll)
        state.phase = Phase.BUILD

    return roll


def begin_robber(state):
    """Set up the aftermath of a 7: who discards, then the robber move.

    Split out of :func:`roll_dice` so tests can reach this position without waiting for
    a 7, and so there is only one place that decides who owes what. Duplicating it in a
    test helper is how the discard count silently drifted once already.

    Discard order starts at the roller and follows turn order, so it depends on the seat
    order rather than on player numbering.
    """
    roller = state.turn_player
    start = state.player_order.index(roller)
    rotated = state.player_order[start:] + state.player_order[:start]

    state.pending_discards = [p for p in rotated if must_discard(state, p)]
    for player in state.pending_discards:
        state.discards_owed[player] = discard_count(state, player)
    _advance_after_discards(state)


def distribute(state, roll):
    """Pay every player for their buildings on ``roll``. Cities yield double.

    The tile under the robber produces nothing, for anybody.

    Bounded by the bank. The official shortage rule: if the bank cannot cover everything
    owed of a resource, **nobody** receives any of it — unless exactly one player is owed
    it, in which case they take whatever is left. So the payout has to be tallied per
    resource before any card moves.

    Returns:
        dict: ``{player: [received per resource]}``, for logging and tests.
    """
    owed = {player: [0] * NUM_RESOURCES for player in state.players}

    for vertex, productions in state.board.producers_for(roll).items():
        owner = state.vertex_owner[vertex]
        if owner == NO_OWNER:
            continue
        amount = PIECE_YIELD[state.vertex_piece[vertex]]
        for production in productions:
            if production.tile == state.robber_tile:
                continue  # blocked by the robber
            owed[owner][production.resource] += amount

    paid = {player: [0] * NUM_RESOURCES for player in state.players}

    for resource in range(NUM_RESOURCES):
        claimants = [p for p in state.players if owed[p][resource] > 0]
        if not claimants:
            continue

        demand = sum(owed[p][resource] for p in claimants)
        available = state.bank[resource]

        if demand <= available:
            grants = [(p, owed[p][resource]) for p in claimants]
        elif len(claimants) == 1:
            grants = [(claimants[0], available)]
        else:
            continue  # short, and more than one claimant: nobody gets any

        for player, amount in grants:
            state.hands[player][resource] += amount
            state.bank[resource] -= amount
            state.produced[player][resource] += amount
            paid[player][resource] = amount
            state.events.append(
                Event(EventKind.PRODUCED, player, resource=resource, amount=amount))

    return paid


# --------------------------------------------------------------------------- #
# SCORING                                                                     #
# --------------------------------------------------------------------------- #

def victory_points(state, player):
    """Total victory points: buildings, both awards, and Victory Point cards.

    1 per settlement, 2 per city, 2 for Largest Army, 2 for Longest Road, and 1 per
    Victory Point card held. Derived rather than kept as a counter, so it cannot drift.

    Victory Point cards count while *held* — they are never played, and stay hidden from
    opponents until they win the game. Phase 3's encoder has to mask them.
    """
    points = sum(
        PIECE_VICTORY_POINTS[state.vertex_piece[vertex]]
        for vertex in range(1, NUM_VERTICES + 1)
        if state.vertex_owner[vertex] == player
    )
    if state.largest_army_holder == player:
        points += AWARD_VICTORY_POINTS
    if state.longest_road_holder == player:
        points += AWARD_VICTORY_POINTS
    return points + state.dev_cards[player][DevCard.VICTORY_POINT]


def public_victory_points(state, player):
    """What opponents can see: everything except hidden Victory Point cards."""
    return victory_points(state, player) - state.dev_cards[player][DevCard.VICTORY_POINT]


def scores(state):
    """``{player: victory points}``."""
    return {player: victory_points(state, player) for player in state.players}


def _check_for_winner(state, player):
    if victory_points(state, player) >= state.ruleset.victory_points_to_win:
        state.winner = player
        state.phase = Phase.GAME_OVER
        state.events.append(Event(EventKind.GAME_OVER, player))


# --------------------------------------------------------------------------- #
# LONGEST ROAD                                                                #
# --------------------------------------------------------------------------- #

def longest_road_lengths(state):
    """``{player: longest road}`` for everyone, memoised.

    The search is exponential, and both :func:`update_awards` and the observation encoder
    want it for every player — so it is computed once per distinct board position and
    shared.

    The memo is keyed on **the ownership arrays themselves**, not invalidated by hand.
    Hand invalidation would be a rule every future mutation site had to remember, including
    test helpers that write straight into the arrays; a derived key simply misses instead.
    Building and hashing the key costs a few microseconds against tens per search.
    """
    key = (tuple(state.edge_owner), tuple(state.vertex_owner))
    if state._longest_road_key == key:
        return state._longest_road_lengths

    lengths = {player: _longest_road_length(state, player) for player in state.players}
    state._longest_road_key = key
    state._longest_road_lengths = lengths
    return lengths


def longest_road_length(state, player):
    """Longest continuous chain of ``player``'s roads. Memoised via
    :func:`longest_road_lengths`."""
    return longest_road_lengths(state)[player]


def _longest_road_length(state, player):
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

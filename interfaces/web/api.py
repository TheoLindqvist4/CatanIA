"""The game as plain dictionaries — no HTTP, no framework.

Every decision the web interface makes happens here, in Python, where it can be tested.
:mod:`interfaces.web.server` is a thin shim that maps URLs onto these functions; swapping it
for FastAPI or Flask would touch nothing in this module.

**Hidden information is filtered here, on the way out.** If an opponent's hand reaches the
JSON it reaches the browser, and someone will read it. :func:`view` is the only function that
builds a response, so it is the only place that has to get this right —
``tests/test_web_api.py`` asserts no response ever carries an opponent's cards.

**The client renders and reports clicks; it decides nothing.** No legality, no board
generation. The last time this project had board logic in JavaScript it was a second
implementation that could disagree with the engine, which is the problem the rewrite removed.
"""

import itertools

from catan import action_space, rules
from catan.actions import ActionType
from catan.agents import GreedyAgent, RandomAgent
from catan.board import GENERIC_HARBOUR
from catan.dev_cards import DevCard
from catan.env import CatanEnv
from catan.events import describe as describe_event
from catan.resources import NUM_RESOURCES, Resource
from catan.rulesets import BASE_GAME, RANKED_1V1
from catan.state import NO_OWNER, Phase, Piece
from catan.topology import NUM_ROADS, NUM_TILES, NUM_VERTICES, ROAD_VERTICES
from interfaces.render import Geometry

#: The human always sits in seat 1, so the client never has to ask which side it is on.
HUMAN = 1

OPPONENTS = {"greedy": GreedyAgent, "random": RandomAgent}
RULESETS = {"ranked1v1": RANKED_1V1, "base": BASE_GAME}

#: Board art, keyed the way the client asks for it.
TILE_IMAGES = {
    Resource.WOOD: "wood", Resource.BRICK: "brick", Resource.SHEEP: "sheep",
    Resource.WHEAT: "weat", Resource.ORE: "stone", None: "desert",
}
RESOURCE_NAMES = [resource.name.lower() for resource in Resource]
DEV_CARD_NAMES = [card.name.lower() for card in DevCard]

#: Action types chosen from a panel rather than by clicking the board.
PANEL_TYPES = (
    ActionType.END_TURN,
    ActionType.BUY_DEV_CARD,
    ActionType.PLAY_KNIGHT,
    ActionType.PLAY_ROAD_BUILDING,
    ActionType.PLAY_YEAR_OF_PLENTY,
    ActionType.PLAY_MONOPOLY,
    ActionType.TRADE_WITH_BANK,
    ActionType.DISCARD,
)


class Game:
    """One game in progress, plus the log the player has not seen yet."""

    _ids = itertools.count(1)

    def __init__(self, opponent="greedy", rules_name="ranked1v1", seed=None):
        if opponent not in OPPONENTS:
            raise ValueError(f"unknown opponent {opponent!r}")
        if rules_name not in RULESETS:
            raise ValueError(f"unknown ruleset {rules_name!r}")

        self.id = next(self._ids)
        self.opponent_name = opponent
        self.rules_name = rules_name
        self.env = CatanEnv(num_players=2, ruleset=RULESETS[rules_name])
        self.opponent = OPPONENTS[opponent](seed)
        self.log = []

        _, self.info = self.env.reset(seed=seed)
        self._record(self.info)
        self._let_opponent_play()

    # ------------------------------------------------------------------ #

    @property
    def state(self):
        return self.env.state

    def _record(self, info):
        names = {HUMAN: "You", 3 - HUMAN: "Opponent"}
        for event in info.get("events", ()):
            self.log.append(describe_event(event, names))

    def _let_opponent_play(self):
        """Play the opponent's decisions until it is the human's turn again.

        Bounded rather than `while True`: a bug that leaves the opponent to move forever
        would otherwise hang the request instead of failing.
        """
        for _ in range(10_000):
            if self.info["done"] or self.info["player"] == HUMAN:
                return
            observation = self.env.observe(self.info["player"])
            action = self.opponent(observation, self.info)
            _, _, _, _, self.info = self.env.step(action)
            self._record(self.info)
        raise RuntimeError("the opponent would not stop moving")

    def play(self, index):
        """Apply the human's action, then let the opponent reply.

        Raises:
            ValueError: if it is not the human's turn, or the action is not legal. The
                engine already refuses illegal actions; this adds the turn check, which
                only a shared interface needs.
        """
        if self.info["done"]:
            raise ValueError("the game is over")
        if self.info["player"] != HUMAN:
            raise ValueError("it is not your turn")
        if index not in self.info["legal"]:
            raise ValueError(f"action {index} is not legal right now")

        _, _, _, _, self.info = self.env.step(index)
        self._record(self.info)
        self._let_opponent_play()
        return self.view()

    def view(self):
        return view(self)


# --------------------------------------------------------------------------- #
# Geometry — static, so the client fetches it once                            #
# --------------------------------------------------------------------------- #

def geometry(hex_width=110):
    """Pixel coordinates for every tile, vertex and road.

    The same lattice :mod:`interfaces.render` maps to pixels for the PNG, so the browser and
    the image agree by construction rather than by two people writing the same layout twice.
    """
    plan = Geometry(hex_width=hex_width)
    return {
        "width": plan.width,
        "height": plan.height,
        "hexWidth": plan.hex_width,
        "hexHeight": plan.hex_height,
        "edge": plan.edge,
        "tiles": [
            {"id": tile, "x": plan.tile(tile)[0], "y": plan.tile(tile)[1]}
            for tile in range(1, NUM_TILES + 1)
        ],
        "vertices": [
            {"id": vertex, "x": plan.vertex(vertex)[0], "y": plan.vertex(vertex)[1]}
            for vertex in range(1, NUM_VERTICES + 1)
        ],
        "roads": [
            _road_geometry(plan, road) for road in range(1, NUM_ROADS + 1)
        ],
    }


def _road_geometry(plan, road):
    first, second = (plan.vertex(v) for v in ROAD_VERTICES[road])
    centre, angle = plan.road(road)
    return {
        "id": road,
        "x1": first[0], "y1": first[1],
        "x2": second[0], "y2": second[1],
        "cx": centre[0], "cy": centre[1],
        "angle": angle,
    }


# --------------------------------------------------------------------------- #
# The view a player is entitled to                                            #
# --------------------------------------------------------------------------- #

def view(game):
    """Everything the human may see, as JSON-ready data."""
    state, info = game.state, game.info
    your_turn = not info["done"] and info["player"] == HUMAN

    return {
        "gameId": game.id,
        "you": HUMAN,
        "opponent": game.opponent_name,
        "rules": game.rules_name,
        "phase": state.phase.name,
        "phaseHint": _hint(state, info, your_turn),
        "turn": info["turn"],
        "lastRoll": info["last_roll"],
        "yourTurn": your_turn,
        "currentPlayer": info["player"],
        "done": info["done"],
        "winner": info["winner"],
        "victoryTarget": state.ruleset.victory_points_to_win,
        "handLimit": state.ruleset.hand_limit,
        "bank": {RESOURCE_NAMES[r]: state.bank[r] for r in range(NUM_RESOURCES)},
        "devDeck": len(state.dev_deck),
        "robber": state.robber_tile,
        "board": _board(state),
        "pieces": _pieces(state),
        "players": [_player(state, info, p) for p in state.players],
        "actions": _actions(state, info, your_turn),
        "log": game.log[-40:],
    }


def _hint(state, info, your_turn):
    """One line telling the player what is being asked of them."""
    if info["done"]:
        if info["winner"] is None:
            return "The game ended without a winner."
        return "You win!" if info["winner"] == HUMAN else "The opponent wins."
    if not your_turn:
        return "Opponent is thinking…"
    return {
        Phase.SETUP_SETTLEMENT: "Place a settlement — click a highlighted spot.",
        Phase.SETUP_ROAD: "Place a road next to it — click a highlighted edge.",
        Phase.DISCARD: "A 7 was rolled. Discard a card.",
        Phase.MOVE_ROBBER: "Move the robber — click a tile.",
        Phase.BUILD: "Build, trade, or end your turn.",
        Phase.ROLL: "Play a development card, or roll.",
    }.get(state.phase, "Your move.")


def _board(state):
    return {
        "tiles": [
            {
                "id": tile,
                "resource": TILE_IMAGES[state.board.resource_at(tile)],
                "number": (None if state.board.resource_at(tile) is None
                           else state.board.number_at(tile)),
                "robber": state.robber_tile == tile,
            }
            for tile in range(1, NUM_TILES + 1)
        ],
        "harbours": [
            {
                "road": road,
                "kind": "3:1" if harbour is GENERIC_HARBOUR
                        else f"2:1 {Resource(harbour).name.lower()}",
            }
            for road, harbour in sorted(state.board.harbours.items())
        ],
    }


def _pieces(state):
    return {
        "buildings": [
            {
                "vertex": vertex,
                "player": state.vertex_owner[vertex],
                "kind": "city" if state.vertex_piece[vertex] is Piece.CITY else "settlement",
            }
            for vertex in range(1, NUM_VERTICES + 1)
            if state.vertex_owner[vertex] != NO_OWNER
        ],
        "roads": [
            {"road": road, "player": state.edge_owner[road]}
            for road in range(1, NUM_ROADS + 1)
            if state.edge_owner[road] != NO_OWNER
        ],
    }


def _player(state, info, player):
    """One player's public standing, plus their own cards if it is the human.

    The filter is here and nowhere else. Anything added to this dict for an opponent is
    visible in the browser.
    """
    mine = player == HUMAN
    entry = {
        "id": player,
        "you": mine,
        "handCount": sum(state.hands[player]),
        "devCount": sum(state.dev_cards[player]),
        "knights": state.knights_played[player],
        "publicVictoryPoints": rules.public_victory_points(state, player),
        "longestRoad": rules.longest_road_length(state, player),
        "largestArmy": state.largest_army_holder == player,
        "longestRoadHolder": state.longest_road_holder == player,
        "settlementsLeft": state.settlements_left[player],
        "citiesLeft": state.cities_left[player],
        "roadsLeft": state.roads_left[player],
    }
    if mine or info["done"]:
        # revealed at the end so the player can see what they were up against
        entry["hand"] = {
            RESOURCE_NAMES[r]: state.hands[player][r] for r in range(NUM_RESOURCES)
        }
        entry["dev"] = {
            DEV_CARD_NAMES[c]: state.dev_cards[player][c] for c in range(len(DevCard))
        }
        entry["victoryPoints"] = rules.victory_points(state, player)
        entry["tradeRates"] = {
            RESOURCE_NAMES[r]: rate
            for r, rate in enumerate(rules.trade_rates(state, player))
        }
    return entry


def _actions(state, info, your_turn):
    """What the human may do: board targets to click, and panel choices."""
    if not your_turn:
        return {"board": {}, "panel": []}

    board = {
        kind.name: {str(position): index for position, index in targets.items()}
        for kind, targets in action_space.clickable(state).items()
    }

    panel = []
    for kind, options in action_space.grouped(state).items():
        if kind not in PANEL_TYPES:
            continue
        for (position, extra), index in sorted(options.items()):
            panel.append({
                "index": index,
                "type": kind.name,
                "label": _panel_label(kind, position, extra),
            })
    panel.sort(key=lambda item: (item["type"] != "END_TURN", item["label"]))
    return {"board": board, "panel": panel}


def _panel_label(kind, position, extra):
    if kind is ActionType.END_TURN:
        return "End turn"
    if kind is ActionType.BUY_DEV_CARD:
        return "Buy development card"
    if kind is ActionType.PLAY_KNIGHT:
        return "Play knight"
    if kind is ActionType.PLAY_ROAD_BUILDING:
        return "Play road building"
    if kind is ActionType.PLAY_MONOPOLY:
        return f"Monopoly: {RESOURCE_NAMES[position]}"
    if kind is ActionType.PLAY_YEAR_OF_PLENTY:
        return f"Year of plenty: {RESOURCE_NAMES[position]} + {RESOURCE_NAMES[extra]}"
    if kind is ActionType.TRADE_WITH_BANK:
        return f"Trade {RESOURCE_NAMES[position]} → {RESOURCE_NAMES[extra]}"
    if kind is ActionType.DISCARD:
        return f"Discard {RESOURCE_NAMES[position]}"
    return kind.name.replace("_", " ").title()


# --------------------------------------------------------------------------- #
# A tiny store, so the server stays stateless-looking                         #
# --------------------------------------------------------------------------- #

class Games:
    """Games in memory, keyed by id. One process, one player — no database."""

    def __init__(self):
        self._games = {}

    def new(self, opponent="greedy", rules_name="ranked1v1", seed=None):
        game = Game(opponent=opponent, rules_name=rules_name, seed=seed)
        self._games[game.id] = game
        return game

    def get(self, game_id):
        try:
            return self._games[int(game_id)]
        except (KeyError, TypeError, ValueError):
            raise KeyError(f"no game {game_id!r}") from None

    def __len__(self):
        return len(self._games)

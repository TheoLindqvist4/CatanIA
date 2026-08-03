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
import pathlib
import random

from catan import action_space, encoder, rules
from catan.actions import ActionType
from catan.agents import DIFFICULTY, GreedyAgent, HeuristicAgent, RandomAgent
from catan.board import GENERIC_HARBOUR
from catan.dev_cards import DevCard
from catan.env import CatanEnv
from catan.events import EventKind
from catan.events import describe as describe_event
from catan.resources import NUM_RESOURCES, Resource
from catan.rulesets import BASE_GAME, RANKED_1V1
from catan.state import NO_OWNER, Phase, Piece
from catan.topology import NUM_ROADS, NUM_TILES, NUM_VERTICES, ROAD_VERTICES
from interfaces import render
from interfaces.web.recorder import Recorder
from interfaces.render import PLAYER_COLOURS as RENDER_COLOURS, Geometry

#: The human always sits in seat 1, so the client never has to ask which side it is on.
HUMAN = 1

#: Selectable opponents, each a factory taking a seed. A trained policy drops in here as
#: one more entry, without the interface changing.
OPPONENTS = {
    "hard": lambda seed: HeuristicAgent(seed, noise=DIFFICULTY["hard"]),
    "medium": lambda seed: HeuristicAgent(seed, noise=DIFFICULTY["medium"]),
    "easy": lambda seed: HeuristicAgent(seed, noise=DIFFICULTY["easy"]),
    "greedy": GreedyAgent,
    "random": RandomAgent,
}
RULESETS = {"ranked1v1": RANKED_1V1, "base": BASE_GAME}

def _register_learned():
    """Offer the champion, if there is a usable one.

    Read from ``models/``, never from ``checkpoints/``: a training run owns the latter and
    rewrites it constantly, and its "best so far" is only the best within that run. A game
    must not be affected by a fine-tune happening at the same time, and must never be handed
    a model that a run later turned out to have made worse. See :mod:`training.champion`.

    Every failure — no file, no PyTorch, a model built for a different observation or action
    space — means the same thing here: offer the heuristic instead.
    """
    try:
        from training import champion
    except ImportError:
        return                                    # no torch on this checkout

    if champion.load() is None:
        return
    OPPONENTS["learned"] = lambda seed: champion.load(temperature=0.35, seed=seed)


_register_learned()

#: The strongest opponent available. The trained policy beats the heuristic comfortably, so
#: it is the default — but it ships as a file under ``models/``, and a champion trained
#: against a different ``encoder.SIZE`` will not load, so a fresh clone (or one without
#: PyTorch, or one mid-way through an observation change) falls back to the heuristic.
DEFAULT_OPPONENT = "learned" if "learned" in OPPONENTS else "hard"

#: How each opponent is named in the interface, in the order it should be offered.
#:
#: Here rather than in the page, because *which* opponents exist is decided at import: a
#: champion whose observation no longer matches the encoder does not load, and an option
#: hardcoded in the HTML would still be clickable and would fail with a 400. The page asks
#: what is available; it does not assume.
OPPONENT_LABELS = {
    "learned": "Learned (strongest)",
    "hard": "Hard",
    "medium": "Medium",
    "easy": "Easy",
    "greedy": "Greedy (weak)",
    "random": "Random",
}


def opponents():
    """The selectable opponents, as the client should list them."""
    return [
        {
            "name": name,
            "label": OPPONENT_LABELS[name],
            "default": name == DEFAULT_OPPONENT,
        }
        for name in OPPONENT_LABELS
        if name in OPPONENTS
    ]

#: Board art, keyed the way the client asks for it.
TILE_IMAGES = {
    Resource.WOOD: "wood", Resource.BRICK: "brick", Resource.SHEEP: "sheep",
    Resource.WHEAT: "weat", Resource.ORE: "stone", None: "desert",
}
RESOURCE_NAMES = [resource.name.lower() for resource in Resource]

#: Resource name -> the file that pictures it. The art set predates the code and calls wheat
#: "weat" and ore "stone"; the client should not have to know that, so the mapping is served
#: rather than hard-coded in JavaScript where it would be a second source of truth.
RESOURCE_IMAGES = {
    resource.name.lower(): TILE_IMAGES[resource] for resource in Resource
}
DEV_CARD_NAMES = [card.name.lower() for card in DevCard]

#: Action types chosen from a panel rather than by clicking the board.
PANEL_TYPES = (
    ActionType.ROLL,
    ActionType.END_TURN,
    ActionType.BUY_DEV_CARD,
    ActionType.PLAY_KNIGHT,
    ActionType.PLAY_ROAD_BUILDING,
    ActionType.PLAY_YEAR_OF_PLENTY,
    ActionType.PLAY_MONOPOLY,
    ActionType.TRADE_WITH_BANK,
    ActionType.DISCARD,
)


def _new_card(before, after):
    """Which development card was added, as a readable name, or ``None``.

    A purchase adds exactly one, so the first index that grew is the card drawn.
    """
    for card in range(len(DevCard)):
        if after[card] > before[card]:
            return DEV_CARD_NAMES[card].replace("_", " ")
    return None


class Game:
    """One game in progress, plus the log the player has not seen yet.

    Args:
        paced: hand the opponent's decisions back one at a time instead of playing its
            whole reply inside :meth:`play`. See :meth:`advance`.
    """

    _ids = itertools.count(1)

    def __init__(self, opponent=None, rules_name="ranked1v1", seed=None,
                 record_game=False, paced=False):
        opponent = DEFAULT_OPPONENT if opponent is None else opponent
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
        self.paced = paced
        # Off by default for the same reason recording is: a paced game is one somebody is
        # *watching*, and only the HTTP layer knows that. Tests, benchmarks and scripts want
        # the opponent's whole reply in one call, and none of them would ever call
        # :meth:`advance`, so pacing them would simply leave the opponent stuck.

        # A seed is chosen here rather than left to the engine. `reset(seed=None)` seeds
        # from the OS and the value is then unknowable, which would make the recording
        # unreplayable — and a recording that cannot be replayed is a summary, not a record.
        self.seed = random.randrange(1 << 30) if seed is None else seed
        _, self.info = self.env.reset(seed=self.seed)
        self.recorder = Recorder(metadata={
            "opponent": opponent,
            "rules": rules_name,
            "seed": self.seed,
        }, human=HUMAN) if record_game else None
        # Off by default on purpose. A game is worth recording because a *person* played
        # it, and this class is also driven by tests, benchmarks and scripts — which would
        # otherwise bury the handful of real games under hundreds of synthetic ones.
        # :mod:`interfaces.web.server` is the only caller that knows a human is involved,
        # so it is the one that turns this on.

        self._record(self.info)
        if not paced:
            self._let_opponent_play()

    # ------------------------------------------------------------------ #

    @property
    def state(self):
        return self.env.state

    def _record(self, info, drew=None):
        """Turn this step's events into log lines.

        ``drew`` names the development card the *human* just bought, if any. It is worked
        out here by comparing their hand before and after, rather than recorded on the
        event: ``info["events"]`` is handed to agents, so a card id on the event would put
        hidden information somewhere an opponent could read it. The web layer knows whose
        side it is on; the engine deliberately does not.
        """
        names = {HUMAN: "You", 3 - HUMAN: "Opponent"}
        for event in info.get("events", ()):
            line = describe_event(event, names)
            if drew is not None and event.kind is EventKind.BOUGHT_DEV                     and event.player == HUMAN:
                line = f"{line}: {drew}"
            self.log.append(line)

    @property
    def awaiting_opponent(self):
        """Whether a decision is outstanding and it is not the human's.

        ``info["player"]`` is the authority rather than the turn order: the game is not
        strictly alternating, and during a discard the decision belongs to whoever is over
        the hand limit.
        """
        return not self.info["done"] and self.info["player"] != HUMAN

    def _opponent_move(self):
        """Play exactly one of the opponent's decisions.

        Returns:
            bool: whether there was one to play. ``False`` means the game is over or it is
            the human's move, so a caller can loop on it without a separate check.
        """
        if not self.awaiting_opponent:
            return False
        observation = self.env.observe(self.info["player"])
        action = self.opponent(observation, self.info)
        if self.recorder:
            self.recorder.record(self.info, action)
        _, _, _, _, self.info = self.env.step(action)
        self._record(self.info)
        if self.info["done"] and self.recorder:
            self.recorder.finish(self.info)
        return True

    def _let_opponent_play(self):
        """Play the opponent's decisions until it is the human's turn again.

        Bounded rather than `while True`: a bug that leaves the opponent to move forever
        would otherwise hang the request instead of failing.
        """
        for _ in range(10_000):
            if not self._opponent_move():
                return
        raise RuntimeError("the opponent would not stop moving")

    def advance(self):
        """Play one of the opponent's decisions and return the position after it.

        This is what makes the opponent watchable. Its whole reply — four setup placements,
        or a roll and a robber move and three builds — arrives from the engine in under a
        millisecond, so playing it all inside :meth:`play` means the board simply looks
        different afterwards and the player never sees what happened. One decision per call
        lets the client draw each of them, in the order they were actually made.

        A no-op when it is the human's move or the game is over, so the client can call it
        without first deciding whether it should — that decision is made here, with the
        engine's ``info``, and not in the browser.
        """
        if self._opponent_move() and self.recorder:
            self.recorder.save()
        return self.view()

    def play(self, index):
        """Apply the human's action, then let the opponent reply.

        Unless this game is :attr:`paced`, in which case the reply is left for
        :meth:`advance` to hand back a decision at a time.

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

        before = list(self.state.dev_cards[HUMAN])
        if self.recorder:
            self.recorder.record(self.info, index)
        _, _, _, _, self.info = self.env.step(index)
        self._record(self.info, drew=_new_card(before, self.state.dev_cards[HUMAN]))
        if self.info["done"] and self.recorder:
            self.recorder.finish(self.info)
        if not self.paced:
            self._let_opponent_play()
        if self.recorder:
            self.recorder.save()
        return self.view()

    def view(self):
        return view(self)


# --------------------------------------------------------------------------- #
# Geometry — static, so the client fetches it once                            #
# --------------------------------------------------------------------------- #

def geometry(hex_width=110):
    """Everything the client needs once and then never again.

    Pixel coordinates for every tile, vertex and road — the same lattice
    :mod:`interfaces.render` maps to pixels for the PNG, so the browser and the image agree
    by construction rather than by two people writing the same layout twice — plus the art
    names, the sizes both renderers draw at, and which opponents exist.
    """
    plan = Geometry(hex_width=hex_width)
    return {
        # Not board layout, but static and fetched in the same breath, which saves the page
        # a second round trip before it can offer a New game button that works.
        "opponents": opponents(),
        # Which picture goes with which name. Shared with interfaces/render.py's PNG
        # renderer, so the board on screen and the board in a saved image use one asset set.
        "art": {
            "resources": RESOURCE_IMAGES,
            "colours": {
                slot + 1: colour for slot, colour in enumerate(RENDER_COLOURS)
            },
            # The sizes the PNG renderer uses, served rather than restated in JavaScript.
            # Two renderers drawing the same assets at different sizes is the kind of quiet
            # divergence this project keeps removing.
            "scales": {
                "settlement": render.SETTLEMENT_SCALE,
                "city": render.CITY_SCALE,
                "number": render.NUMBER_SCALE,
                "spot": render.SPOT_SCALE,
                "roadLength": render.ROAD_LENGTH_SCALE,
                "robber": render.ROBBER_SCALE,
            },
            # Not sizes but placements, in fractions of one hex height below a tile's
            # centre. The number token has one because the art leaves a blank panel for it
            # that is not in the middle of the hex — see render.NUMBER_OFFSET.
            "offsets": {
                "number": render.NUMBER_OFFSET,
            },
        },
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
        # Whether the client should ask for another move to be played. Decided here, from
        # the engine's `info`, rather than inferred in the browser from `yourTurn` and
        # `done` — the client renders and reports clicks; it works nothing out.
        "awaitingOpponent": game.awaiting_opponent,
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
        # Only reachable when a card *could* be played; playing it is optional.
        Phase.ROLL: "You may play a development card first — or just roll.",
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
        # Bought this turn, so not yet playable. Shown separately rather than folded in,
        # because "you have a knight" and "you have a knight you cannot use yet" are
        # different things to know when deciding a move.
        entry["devNew"] = {
            DEV_CARD_NAMES[c]: state.dev_cards_new[player][c] for c in range(len(DevCard))
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
    panel.sort(key=lambda item: (item["type"] not in ("ROLL", "END_TURN"), item["label"]))
    return {"board": board, "panel": panel}


def _panel_label(kind, position, extra):
    if kind is ActionType.ROLL:
        # Only ever offered when a development card could be played first, so the label
        # says what declining means rather than just "roll".
        return "Roll the dice (keep your cards)"
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

    def new(self, opponent=None, rules_name="ranked1v1", seed=None, record_game=False,
            paced=False):
        game = Game(opponent=opponent, rules_name=rules_name, seed=seed,
                    record_game=record_game, paced=paced)
        self._games[game.id] = game
        return game

    def get(self, game_id):
        try:
            return self._games[int(game_id)]
        except (KeyError, TypeError, ValueError):
            raise KeyError(f"no game {game_id!r}") from None

    def __len__(self):
        return len(self._games)

"""Everything mutable about a game in progress, and nothing else.

This replaces the old model, where occupancy lived as a shrinking "available" set on the
board. That set collapsed three distinct facts into one bit — *empty*, *blocked by the
distance rule*, and *occupied by player N* — and recorded no owner, which made
road-blocking, city upgrades and every AI observation impossible. Availability is now
*derived* from ownership rather than stored.

Layout: parallel arrays indexed by 1-based topology id, slot 0 unused.

    vertex_owner[54]   player number, or NO_OWNER
    vertex_piece[54]   Piece.NONE / SETTLEMENT / CITY
    edge_owner[72]     player number, or NO_OWNER
    hands[player]      list[int] of length 5, indexed by Resource

Arrays rather than sets of positions because the observation encoder (Phase 3) wants
fixed-width vectors, and because :meth:`GameState.clone` is a hot path for search.

The :class:`~catan.board.Board` is immutable and shared by reference across clones — see
``docs/decisions/0009-immutable-board-mutable-state.md``. Anything that changes during
play belongs here, including the robber when Phase 2 adds it.
"""

import random
from enum import IntEnum

from catan.board import Board
from catan.resources import BANK_PER_RESOURCE, NUM_RESOURCES, empty_hand
from catan.topology import NUM_ROADS, NUM_VERTICES

#: Sentinel for an unowned vertex or road.
NO_OWNER = 0


class Piece(IntEnum):
    NONE = 0
    SETTLEMENT = 1
    CITY = 2


class Phase(IntEnum):
    """Which decision the game is waiting for.

    The two setup phases alternate per placement, so every setup step is one atomic
    action — which is what an RL agent needs.
    """

    SETUP_SETTLEMENT = 0
    SETUP_ROAD = 1
    ROLL = 2
    BUILD = 3
    GAME_OVER = 4


#: Victory points a piece is worth.
PIECE_VICTORY_POINTS = {Piece.NONE: 0, Piece.SETTLEMENT: 1, Piece.CITY: 2}

#: Resources a piece collects per matching tile.
PIECE_YIELD = {Piece.NONE: 0, Piece.SETTLEMENT: 1, Piece.CITY: 2}

MAX_SETTLEMENTS = 5
MAX_CITIES = 4
MAX_ROADS = 15

VICTORY_POINTS_TO_WIN = 10

MIN_PLAYERS = 2
MAX_PLAYERS = 4


class GameState:
    def __init__(self, num_players=2, seed=None, rng=None, board=None,
                 player_order=None):
        """
        Args:
            num_players: 2 to 4.
            seed: seed for a fresh generator. Reproduces an entire game.
            rng: an existing ``random.Random`` to use instead.
            board: an existing board to reuse; generated from ``rng`` if omitted.
            player_order: turn order. Defaults to ``[1, 2, ...]``; pass a shuffled
                list, or call :meth:`randomize_order` before setup begins.
        """
        if not MIN_PLAYERS <= num_players <= MAX_PLAYERS:
            raise ValueError(
                f"num_players must be in {MIN_PLAYERS}..{MAX_PLAYERS}, got {num_players}"
            )

        self.rng = rng if rng is not None else random.Random(seed)
        self.num_players = num_players
        self.board = board if board is not None else Board(rng=self.rng)

        self.player_order = (
            list(player_order) if player_order is not None
            else list(range(1, num_players + 1))
        )
        if sorted(self.player_order) != list(range(1, num_players + 1)):
            raise ValueError(f"player_order must be a permutation of 1..{num_players}")

        # Occupancy. Index 0 unused throughout.
        self.vertex_owner = [NO_OWNER] * (NUM_VERTICES + 1)
        self.vertex_piece = [Piece.NONE] * (NUM_VERTICES + 1)
        self.edge_owner = [NO_OWNER] * (NUM_ROADS + 1)

        # Per player, index 0 unused.
        self.hands = [None] + [empty_hand() for _ in range(num_players)]
        self.settlements_left = [0] + [MAX_SETTLEMENTS] * num_players
        self.cities_left = [0] + [MAX_CITIES] * num_players
        self.roads_left = [0] + [MAX_ROADS] * num_players

        #: The bank's supply, indexed by Resource. Cards are conserved: every card is
        #: either here or in a hand, which ``test_cards_are_conserved`` checks.
        self.bank = [BANK_PER_RESOURCE] * NUM_RESOURCES

        self.phase = Phase.SETUP_SETTLEMENT
        self.setup_step = 0
        #: The settlement just placed, which the next setup road must touch.
        self.last_settlement = None

        self.turn_number = 0
        self.last_roll = None
        self.winner = None

    # ------------------------------------------------------------------ #
    # PLAYERS AND TURN ORDER                                             #
    # ------------------------------------------------------------------ #

    @property
    def players(self):
        return range(1, self.num_players + 1)

    @property
    def setup_sequence(self):
        """Placement order: round one forwards, round two reversed (a snake)."""
        return self.player_order + self.player_order[::-1]

    @property
    def setup_round(self):
        """1 or 2 during setup, otherwise ``None``."""
        if not self.in_setup:
            return None
        return 1 if self.setup_step < self.num_players else 2

    @property
    def in_setup(self):
        return self.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD)

    @property
    def current_player(self):
        """Whose decision the game is waiting on."""
        if self.in_setup:
            return self.setup_sequence[self.setup_step]
        if self.phase is Phase.GAME_OVER:
            return self.winner
        return self.player_order[self.turn_number % self.num_players]

    def randomize_order(self):
        """Shuffle turn order. Only meaningful before setup starts."""
        self.rng.shuffle(self.player_order)
        return list(self.player_order)

    # ------------------------------------------------------------------ #
    # QUERIES                                                            #
    # ------------------------------------------------------------------ #

    def buildings_of(self, player):
        """Vertices where ``player`` has a settlement or city."""
        return tuple(
            v for v in range(1, NUM_VERTICES + 1) if self.vertex_owner[v] == player
        )

    def roads_of(self, player):
        """Roads owned by ``player``."""
        return tuple(
            r for r in range(1, NUM_ROADS + 1) if self.edge_owner[r] == player
        )

    def is_vertex_free(self, vertex):
        return self.vertex_owner[vertex] == NO_OWNER

    def is_road_free(self, road):
        return self.edge_owner[road] == NO_OWNER

    def hand(self, player):
        return self.hands[player]

    # ------------------------------------------------------------------ #
    # CLONING                                                            #
    # ------------------------------------------------------------------ #

    def clone(self, rng=None):
        """A copy that shares the immutable board.

        By default the clone snapshots this state's generator, so it is a true
        point-in-time copy and replays identically.

        **Pass ``rng=state.rng`` in search.** Snapshotting a Mersenne Twister copies 625
        words, which measures at ~17 us per clone against ~1.3 us when the stream is
        shared — 92% of the cost, for a property rollouts do not want anyway. Sharing
        also makes sibling rollouts diverge, which is the point of sampling them.
        """
        other = object.__new__(GameState)

        other.board = self.board  # immutable, shared on purpose
        other.num_players = self.num_players
        other.player_order = list(self.player_order)

        if rng is not None:
            other.rng = rng
        else:
            other.rng = random.Random()
            other.rng.setstate(self.rng.getstate())

        other.vertex_owner = list(self.vertex_owner)
        other.vertex_piece = list(self.vertex_piece)
        other.edge_owner = list(self.edge_owner)

        other.hands = [None] + [list(hand) for hand in self.hands[1:]]
        other.settlements_left = list(self.settlements_left)
        other.cities_left = list(self.cities_left)
        other.roads_left = list(self.roads_left)
        other.bank = list(self.bank)

        other.phase = self.phase
        other.setup_step = self.setup_step
        other.last_settlement = self.last_settlement
        other.turn_number = self.turn_number
        other.last_roll = self.last_roll
        other.winner = self.winner

        return other

    # ------------------------------------------------------------------ #
    # COMPARISON                                                         #
    # ------------------------------------------------------------------ #

    _COMPARED = (
        "num_players", "player_order", "vertex_owner", "vertex_piece", "edge_owner",
        "hands", "bank", "settlements_left", "cities_left", "roads_left", "phase",
        "setup_step", "last_settlement", "turn_number", "last_roll", "winner",
    )

    def __eq__(self, other):
        """The same position on the same board layout.

        Ignores the RNG: two states can be identical positions reached by different
        draws. Compares boards by *value*, not identity — replaying a seed builds an
        equal board in a new object, and that is the same game.
        """
        if not isinstance(other, GameState):
            return NotImplemented
        if self.board != other.board:
            return False
        return all(
            getattr(self, name) == getattr(other, name) for name in self._COMPARED
        )

    def __repr__(self):
        return (
            f"GameState(players={self.num_players}, phase={self.phase.name}, "
            f"turn={self.turn_number}, current={self.current_player})"
        )

"""The Catan board: generated geometry from :mod:`catan.topology` plus a mutable layout.

.. deprecated::
   Superseded by :mod:`catan.board` + :class:`catan.state.GameState`, which separate the
   immutable layout from ownership. Kept only so the legacy terminal demo keeps working;
   it is deleted when ``interfaces/cli.py`` lands in Phase 4. Do not add to it.

Tiles are held in **flat arrays indexed by tile id**, not in the ragged 3-4-5-4-3 rows
they are drawn in. The rows were leaking into everything: number placement had to track
how much of each row was filled, and every tile lookup went through a (row, col)
translation. Tiles are numbered 1..19 in row-major order, so the row view is a pure
presentation concern — :attr:`grid` and :attr:`tile_grid` rebuild it on demand for
display.

This module is I/O-free. Nothing prints; :meth:`display_board` *returns* a string. A
training loop must be able to build millions of boards without touching stdout.
Randomness is injected, so a board is reproducible from a seed.

See ``docs/decisions/`` for why the geometry is generated and why tiles are flat.

Known Phase 1 work (see ROADMAP.md), deliberately left alone here: this class still
owns ``settlement_positions`` / ``road_positions`` as flat "available" sets, so *empty*,
*blocked by the distance rule* and *occupied by player N* are indistinguishable and no
owner is recorded. That moves to ``GameState``.
"""

import random
from typing import NamedTuple

from catan.topology import (
    NUM_ROADS,
    NUM_TILES,
    NUM_VERTICES,
    ROAD_NEIGHBOURS,
    ROAD_VERTICES,
    ROW_LENGTHS,
    TILE_ADJACENCY,
    TILE_ROADS,
    VERTEX_NEIGHBOURS,
    VERTEX_ROADS,
    VERTEX_TILES,
    check_id,
)

DESERT = 'Desert'

#: The roll that produces nothing and (from Phase 2) moves the robber.
ROBBER_ROLL = 7


class Production(NamedTuple):
    """One tile's contribution to one vertex."""

    tile: int
    number: int
    resource: str


class Board:
    #: Standard token distribution: one 2, two each of 3-6 and 8-11, one 12 (18
    #: tokens), plus a 7 standing in for the desert. See
    #: docs/decisions/0004-desert-as-the-seven-tile.md.
    NUMBERS = (2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12)

    #: Standard resource distribution, 19 tiles total. Insertion order is part of the
    #: generation sequence, so a given seed keeps producing the same board.
    TILE_COUNTS = {
        'Ore': 3,
        'Weat': 4,
        'Sheep': 4,
        'Brick': 3,
        'Wood': 4,
        DESERT: 1,
    }

    #: Number pairs that may not be adjacent, on top of "no equal neighbours".
    UNBALANCED_PAIRS = (frozenset((6, 8)), frozenset((2, 12)))

    def __init__(self, rng=None, max_generation_attempts=100):
        """
        Args:
            rng: a ``random.Random`` instance. Injected for reproducibility; a fresh
                unseeded one is created if omitted.
            max_generation_attempts: bound on balanced-layout retries. Placement is
                greedy and can paint itself into a corner; empirically it needs a
                median of 2 attempts and a maximum of 5, so the default is ample.
                Bounded so a bad constraint set fails loudly instead of hanging.
        """
        self.rng = rng if rng is not None else random.Random()

        # tile id -> number token / resource name. Index 0 unused.
        self.tile_numbers = [None] * (NUM_TILES + 1)
        self.tile_resources = [None] * (NUM_TILES + 1)

        # vertex -> every tile touching it, as Production records
        self.vertex_production = {}
        # roll -> {vertex: productions paying out on that roll}
        self._producers_by_roll = {}

        # Positions still legal to build on. Sets, not lists: every use is a
        # membership test or a removal.
        self.settlement_positions = set(range(1, NUM_VERTICES + 1))
        self.road_positions = set(range(1, NUM_ROADS + 1))

        self.generate_board(max_attempts=max_generation_attempts)
        self.assign_tiles()
        self.index_production()

    # ------------------------------------------------------------------ #
    # CREATION OF THE BOARD                                              #
    # ------------------------------------------------------------------ #

    def generate_board(self, max_attempts=100):
        """Place the number tokens, greedily, subject to :meth:`is_number_valid`.

        Tiles are filled in id order. Because a tile is only ever validated against
        neighbours that already carry a number, the constraint is checked
        incrementally without needing to know how far generation has got.

        Raises:
            RuntimeError: if no valid layout is found within ``max_attempts``.
        """
        for _ in range(max_attempts):
            numbers = list(self.NUMBERS)
            self.rng.shuffle(numbers)
            placed = [None] * (NUM_TILES + 1)

            for tile in range(1, NUM_TILES + 1):
                for index, number in enumerate(numbers):
                    if self._number_fits(placed, tile, number):
                        placed[tile] = numbers.pop(index)
                        break
                else:
                    break  # dead end, reshuffle
            else:
                if not numbers:
                    self.tile_numbers = placed
                    return

        raise RuntimeError(
            f"could not generate a valid board in {max_attempts} attempts"
        )

    def _number_fits(self, placed, tile, number):
        """Whether ``number`` may go on ``tile`` given the numbers already in ``placed``."""
        neighbours = [
            placed[other]
            for other in TILE_ADJACENCY[tile]
            if placed[other] is not None
        ]
        if number in neighbours:
            return False
        return not any(
            frozenset((number, neighbour)) in self.UNBALANCED_PAIRS
            for neighbour in neighbours
        )

    def assign_tiles(self):
        """Assign a resource to every tile. The 7 tile is always the desert."""
        remaining = dict(self.TILE_COUNTS)
        self.tile_resources = [None] * (NUM_TILES + 1)

        for tile in range(1, NUM_TILES + 1):
            if self.tile_numbers[tile] == ROBBER_ROLL:
                self.tile_resources[tile] = DESERT
                remaining[DESERT] -= 1

        pool = [
            resource
            for resource, count in remaining.items()
            for _ in range(count)
        ]
        self.rng.shuffle(pool)

        open_tiles = [
            tile for tile in range(1, NUM_TILES + 1)
            if self.tile_resources[tile] is None
        ]
        for tile, resource in zip(open_tiles, pool):
            self.tile_resources[tile] = resource

    def index_production(self):
        """Precompute what each vertex collects, and what each roll pays out.

        Indexing by roll turns a dice payout from a 54-vertex scan into one dict
        lookup. The desert is excluded here, which is what makes a 7 structurally
        inert rather than something every caller has to remember to filter.
        """
        self.vertex_production = {
            vertex: tuple(
                Production(tile, self.tile_numbers[tile], self.tile_resources[tile])
                for tile in VERTEX_TILES[vertex]
            )
            for vertex in range(1, NUM_VERTICES + 1)
        }

        self._producers_by_roll = {roll: {} for roll in range(2, 13)}
        for vertex, productions in self.vertex_production.items():
            for production in productions:
                if production.resource == DESERT:
                    continue
                bucket = self._producers_by_roll[production.number]
                bucket.setdefault(vertex, []).append(production)

        self._producers_by_roll = {
            roll: {vertex: tuple(items) for vertex, items in bucket.items()}
            for roll, bucket in self._producers_by_roll.items()
        }
        return self.vertex_production

    # ------------------------------------------------------------------ #
    # TILE ACCESS                                                        #
    # ------------------------------------------------------------------ #

    def number_at(self, tile):
        """Number token on a tile id."""
        return self.tile_numbers[check_id(tile, NUM_TILES, "tile")]

    def resource_at(self, tile):
        """Resource name on a tile id."""
        return self.tile_resources[check_id(tile, NUM_TILES, "tile")]

    def numbers_around(self, tile):
        """Number tokens on the tiles adjacent to ``tile``."""
        return tuple(
            self.tile_numbers[other]
            for other in TILE_ADJACENCY[check_id(tile, NUM_TILES, "tile")]
            if self.tile_numbers[other] is not None
        )

    def is_number_valid(self, tile, number):
        """Whether ``number`` satisfies the balanced-board rule at ``tile``.

        No equal neighbours, and no 6/8 or 2/12 pair. This is a house rule, not
        official Catan, which uses a fixed spiral — see
        docs/decisions/0005-balanced-board-generation.md.
        """
        return self._number_fits(self.tile_numbers, tile, number)

    @property
    def desert_tile(self):
        """The tile carrying the desert."""
        return self.tile_resources.index(DESERT)

    # ------------------------------------------------------------------ #
    # THE RAGGED-ROW VIEW — presentation only                            #
    # ------------------------------------------------------------------ #

    @property
    def grid(self):
        """Number tokens laid out in the 3-4-5-4-3 rows, for display."""
        return self._as_rows(self.tile_numbers)

    @property
    def tile_grid(self):
        """Resources laid out in the 3-4-5-4-3 rows, for display."""
        return self._as_rows(self.tile_resources)

    @staticmethod
    def _as_rows(flat):
        rows, start = [], 1
        for length in ROW_LENGTHS:
            rows.append(flat[start:start + length])
            start += length
        return rows

    def display_board(self):
        """Return a human-readable rendering. Does not print — the core stays I/O-free."""
        return "\n".join(
            " ".join(f"{number}({resource})" for number, resource in zip(*rows))
            for rows in zip(self.grid, self.tile_grid)
        )

    # ------------------------------------------------------------------ #
    # ADJACENCY — thin, validating wrappers over the generated topology   #
    # ------------------------------------------------------------------ #

    def get_adjacent_tiles(self, tile):
        """Tile ids sharing a road with ``tile``."""
        return TILE_ADJACENCY[check_id(tile, NUM_TILES, "tile")]

    def get_roads_of_tile(self, tile):
        """The six roads on ``tile``'s boundary."""
        return TILE_ROADS[check_id(tile, NUM_TILES, "tile")]

    def get_adjacents_for_positions(self, position):
        """Vertices one road away from ``position``."""
        return VERTEX_NEIGHBOURS[check_id(position, NUM_VERTICES, "settlement position")]

    def get_adjacent_roads_from_settlement(self, position):
        """Roads meeting at ``position``."""
        return VERTEX_ROADS[check_id(position, NUM_VERTICES, "settlement position")]

    def get_adjacent_roads_from_road(self, road_position):
        """Roads sharing an endpoint with ``road_position``."""
        return ROAD_NEIGHBOURS[check_id(road_position, NUM_ROADS, "road position")]

    def get_adjacent_settlement_from_road(self, road_position):
        """The two endpoint vertices of ``road_position``."""
        return ROAD_VERTICES[check_id(road_position, NUM_ROADS, "road position")]

    def get_tiles_for_position(self, position):
        """Tile ids touching ``position``."""
        return VERTEX_TILES[check_id(position, NUM_VERTICES, "settlement position")]

    # ------------------------------------------------------------------ #
    # AVAILABILITY                                                        #
    # ------------------------------------------------------------------ #

    def is_settlement_position_available(self, number):
        """Whether a settlement may still be built at ``number``."""
        return number in self.settlement_positions

    def is_road_position_available(self, number):
        """Whether a road may still be built at ``number``."""
        return number in self.road_positions

    def delete_settlement_position(self, number):
        """Consume ``number`` and its neighbours (Catan's distance-2 rule).

        Phase 1 replaces this: removing the neighbours enforces the spacing rule but
        also erases the distinction between "occupied" and "merely blocked", and
        records no owner.
        """
        check_id(number, NUM_VERTICES, "settlement position")
        self.settlement_positions.discard(number)
        for adjacent in VERTEX_NEIGHBOURS[number]:
            self.settlement_positions.discard(adjacent)

    def delete_road_position(self, number):
        """Consume road ``number``."""
        check_id(number, NUM_ROADS, "road position")
        self.road_positions.discard(number)

    # ------------------------------------------------------------------ #
    # GETTERS — sorted copies: stable output, no shared mutables          #
    # ------------------------------------------------------------------ #

    def get_available_settlements(self):
        return sorted(self.settlement_positions)

    def get_available_road(self):
        return sorted(self.road_positions)

    def get_available_road_from_settlement(self, settlement_position):
        """Roads at ``settlement_position`` that are still unbuilt.

        Filtered against occupancy — unfiltered, the setup phase offered roads that
        had already been built.
        """
        return sorted(
            road
            for road in self.get_adjacent_roads_from_settlement(settlement_position)
            if road in self.road_positions
        )

    def producers_for(self, roll):
        """``{vertex: (Production, ...)}`` for everything that pays out on ``roll``.

        The desert is already excluded, so a 7 returns an empty mapping.
        """
        if roll not in self._producers_by_roll:
            raise ValueError(f"roll must be in 2..12, got {roll}")
        return self._producers_by_roll[roll]

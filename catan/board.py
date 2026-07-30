"""One board layout: which number and which resource sits on each tile.

**Immutable after construction.** Nothing here changes during a game, so
:meth:`GameState.clone` shares the board by reference instead of copying it — which is
what makes cloning cheap enough for MCTS. Anything that *does* change during play lives
in :class:`~catan.state.GameState`, including the robber (Phase 2), even though a robber
feels like a board thing.

Tiles are held in flat arrays indexed by tile id. The 3-4-5-4-3 rows they are drawn in
are a presentation concern; :attr:`rows` rebuilds them on demand.

This module performs no I/O and takes its randomness by injection.
See ``docs/decisions/0009-immutable-board-mutable-state.md``.
"""

import random
from typing import NamedTuple

from catan.resources import DESERT, Resource
from catan.topology import (
    COASTAL_CYCLE,
    NUM_TILES,
    NUM_VERTICES,
    ROAD_VERTICES,
    ROW_LENGTHS,
    TILE_ADJACENCY,
    VERTEX_TILES,
    check_id,
)

#: The roll that produces nothing and moves the robber.
ROBBER_ROLL = 7

#: Rolls a pair of dice can produce.
ROLLS = range(2, 13)

#: A generic 3:1 harbour, as opposed to a resource-specific 2:1 one.
GENERIC_HARBOUR = None

#: The nine harbours: four generic 3:1, plus one 2:1 for each resource.
HARBOUR_TYPES = (GENERIC_HARBOUR,) * 4 + tuple(Resource)

#: Gaps between consecutive harbours walking :data:`COASTAL_CYCLE`. 3+3+4 repeated three
#: times is exactly 30, so nine harbours land evenly and no vertex serves two of them.
HARBOUR_SPACING = (3, 3, 4) * 3


class Production(NamedTuple):
    """One tile's contribution to one vertex."""

    tile: int
    number: int
    resource: Resource


class Board:
    #: Standard token distribution: one 2, two each of 3-6 and 8-11, one 12 (18
    #: tokens), plus a 7 standing in for the desert.
    #: See docs/decisions/0004-desert-as-the-seven-tile.md.
    NUMBERS = (2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12)

    #: Standard resource distribution, 19 tiles total. Insertion order is part of the
    #: generation sequence, so a given seed keeps producing the same board.
    TILE_COUNTS = {
        Resource.ORE: 3,
        Resource.WHEAT: 4,
        Resource.SHEEP: 4,
        Resource.BRICK: 3,
        Resource.WOOD: 4,
        DESERT: 1,
    }

    #: Number pairs that may not be adjacent, on top of "no equal neighbours".
    #: A house rule — see docs/decisions/0005-balanced-board-generation.md.
    UNBALANCED_PAIRS = (frozenset((6, 8)), frozenset((2, 12)))

    def __init__(self, rng=None, max_generation_attempts=100):
        """
        Args:
            rng: a ``random.Random``. Injected for reproducibility.
            max_generation_attempts: bound on balanced-layout retries. Placement is
                greedy and can dead-end; empirically it needs a median of 2 attempts
                and a maximum of 5. Bounded so a bad constraint set fails loudly
                instead of hanging.
        """
        rng = rng if rng is not None else random.Random()

        #: tile id -> number token. Index 0 unused.
        self.tile_numbers = self._generate_numbers(rng, max_generation_attempts)
        #: tile id -> resource, or ``DESERT``. Index 0 unused.
        self.tile_resources = self._assign_resources(rng)

        #: vertex -> every tile touching it, as ``Production`` records.
        self.vertex_production = self._index_vertices()
        #: roll -> ``{vertex: productions}``, desert already excluded.
        self._producers_by_roll = self._index_rolls()

        #: coastal road -> harbour type (``GENERIC_HARBOUR`` or a ``Resource``).
        self.harbours = self._place_harbours(rng)
        #: vertex -> the harbour types usable from it.
        self._harbours_by_vertex = self._index_harbours()

        self.desert_tile = self.tile_resources.index(DESERT, 1)

    # ------------------------------------------------------------------ #
    # GENERATION                                                          #
    # ------------------------------------------------------------------ #

    def _generate_numbers(self, rng, max_attempts):
        """Place the number tokens greedily, subject to the balanced-board rule.

        Tiles are filled in id order, and a tile is only ever validated against
        neighbours that already carry a number, so the constraint is checked
        incrementally without tracking how far generation has got.
        """
        for _ in range(max_attempts):
            numbers = list(self.NUMBERS)
            rng.shuffle(numbers)
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
                    return placed

        raise RuntimeError(
            f"could not generate a valid board in {max_attempts} attempts"
        )

    def _number_fits(self, placed, tile, number):
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

    def _assign_resources(self, rng):
        """Assign a resource to every tile. The 7 tile is always the desert."""
        remaining = dict(self.TILE_COUNTS)
        resources = [None] * (NUM_TILES + 1)
        is_desert = [False] * (NUM_TILES + 1)

        for tile in range(1, NUM_TILES + 1):
            if self.tile_numbers[tile] == ROBBER_ROLL:
                is_desert[tile] = True
                remaining[DESERT] -= 1

        pool = [
            resource
            for resource, count in remaining.items()
            for _ in range(count)
        ]
        rng.shuffle(pool)

        open_tiles = [t for t in range(1, NUM_TILES + 1) if not is_desert[t]]
        for tile, resource in zip(open_tiles, pool):
            resources[tile] = resource
        return resources

    def _index_vertices(self):
        return {
            vertex: tuple(
                Production(tile, self.tile_numbers[tile], self.tile_resources[tile])
                for tile in VERTEX_TILES[vertex]
            )
            for vertex in range(1, NUM_VERTICES + 1)
        }

    def _index_rolls(self):
        """Precompute what each roll pays out.

        Turns a payout from a 54-vertex scan into one dict lookup. The desert is
        excluded here, which makes a 7 structurally inert rather than something every
        caller must remember to filter.
        """
        buckets = {roll: {} for roll in ROLLS}
        for vertex, productions in self.vertex_production.items():
            for production in productions:
                if production.resource is DESERT:
                    continue
                buckets[production.number].setdefault(vertex, []).append(production)
        return {
            roll: {vertex: tuple(items) for vertex, items in bucket.items()}
            for roll, bucket in buckets.items()
        }

    def _place_harbours(self, rng):
        """Put the nine harbours on coastal roads, evenly spaced, types shuffled.

        Positions come from walking :data:`COASTAL_CYCLE` with
        :data:`HARBOUR_SPACING`; only which harbour lands where is random. See
        ``docs/decisions/0010-harbour-placement.md`` — real Catan prints harbours on a
        fixed sea frame, and this is an even-spacing approximation of it.
        """
        slots, index = [], 0
        for gap in HARBOUR_SPACING:
            slots.append(COASTAL_CYCLE[index])
            index += gap

        types = list(HARBOUR_TYPES)
        rng.shuffle(types)
        return dict(zip(slots, types))

    def _index_harbours(self):
        """Both endpoints of a harbour road can use it."""
        by_vertex = {}
        for road, harbour in self.harbours.items():
            for vertex in ROAD_VERTICES[road]:
                by_vertex.setdefault(vertex, set()).add(harbour)
        return {vertex: frozenset(kinds) for vertex, kinds in by_vertex.items()}

    # ------------------------------------------------------------------ #
    # ACCESS                                                              #
    # ------------------------------------------------------------------ #

    def harbours_at(self, vertex):
        """The harbour types a building on ``vertex`` would grant. Usually empty."""
        return self._harbours_by_vertex.get(vertex, frozenset())

    @property
    def harbour_vertices(self):
        """Every vertex that grants a harbour."""
        return tuple(sorted(self._harbours_by_vertex))

    def number_at(self, tile):
        return self.tile_numbers[check_id(tile, NUM_TILES, "tile")]

    def resource_at(self, tile):
        return self.tile_resources[check_id(tile, NUM_TILES, "tile")]

    def producers_for(self, roll):
        """``{vertex: (Production, ...)}`` for everything that pays out on ``roll``.

        The desert is already excluded, so a 7 returns an empty mapping.
        """
        if roll not in self._producers_by_roll:
            raise ValueError(f"roll must be in 2..12, got {roll}")
        return self._producers_by_roll[roll]

    def production_at(self, vertex):
        """Every tile touching ``vertex``, desert included."""
        return self.vertex_production[check_id(vertex, NUM_VERTICES, "vertex")]

    def resources_at(self, vertex):
        """The resources ``vertex`` collects, desert excluded. Used by setup payout."""
        return tuple(
            production.resource
            for production in self.vertex_production[
                check_id(vertex, NUM_VERTICES, "vertex")
            ]
            if production.resource is not DESERT
        )

    # ------------------------------------------------------------------ #
    # PRESENTATION                                                        #
    # ------------------------------------------------------------------ #

    @property
    def rows(self):
        """``[(numbers, resources), ...]`` in the 3-4-5-4-3 layout, for display."""
        out, start = [], 1
        for length in ROW_LENGTHS:
            out.append((
                self.tile_numbers[start:start + length],
                self.tile_resources[start:start + length],
            ))
            start += length
        return out

    def render(self):
        """Return a human-readable rendering. Does not print."""
        return "\n".join(
            " ".join(
                f"{number}({'Desert' if resource is DESERT else Resource(resource).name.title()})"
                for number, resource in zip(numbers, resources)
            )
            for numbers, resources in self.rows
        )

    # ------------------------------------------------------------------ #
    # IDENTITY                                                            #
    # ------------------------------------------------------------------ #

    @property
    def layout(self):
        """The whole board as one hashable value: what a seed determines."""
        return (
            tuple(self.tile_numbers),
            tuple(self.tile_resources),
            tuple(sorted(self.harbours.items(), key=lambda item: item[0])),
        )

    def __eq__(self, other):
        """Two boards are equal when they have the same layout.

        Value equality, not identity: the same seed replayed produces an equal board in
        a different object, and two such games are the same game. Clones share one board
        object, which the ``is`` short-circuit keeps free.
        """
        if self is other:
            return True
        if not isinstance(other, Board):
            return NotImplemented
        return self.layout == other.layout

    def __hash__(self):
        """Safe because the board never changes after construction."""
        return hash(self.layout)

    def __repr__(self):
        return f"Board(desert_tile={self.desert_tile})"

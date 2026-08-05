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


def roll_odds(number):
    """The share of rolls that produce ``number``, out of 36. ``None`` (a desert) is 0.

    Two dice, so 8 pays out five times as often as 2. Here rather than in three callers
    because it is the same fact each time.
    """
    return 0.0 if number is None else (6 - abs(7 - number)) / 36.0

#: The nine harbours: four generic 3:1, plus one 2:1 for each resource.
HARBOUR_TYPES = (GENERIC_HARBOUR,) * 4 + tuple(Resource)

#: Gaps between consecutive harbours walking :data:`COASTAL_CYCLE`, as a multiset that is
#: shuffled per board. Nine harbours over a 30-road coastline needs gaps summing to 30;
#: with every gap 3 or 4 that forces exactly six 3s and three 4s (6x3 + 3x4 = 30). Keeping
#: gaps in {3, 4} is what stops harbours clustering or leaving a stretch of coast bare,
#: and guarantees no vertex ever serves two of them.
HARBOUR_SPACING = (3,) * 6 + (4,) * 3


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
        """Put the nine harbours on coastal roads: random, but evenly spaced.

        Both the starting point on the coastline and the rotation of the gap pattern
        are drawn from ``rng``, so harbours land somewhere different every game while
        never clustering — 90 distinct position sets, times the type shuffle.

        Randomising is sanctioned by the rules: harbour positions are "either fixed or
        randomized depending on your group's preference", and the nine harbour tokens
        may be shuffled. The published rules do not name the coastal edges the printed
        frame uses, so there is no official list to follow.
        See ``docs/decisions/0010-harbour-placement.md``.
        """
        coast = len(COASTAL_CYCLE)
        gaps = list(HARBOUR_SPACING)
        rng.shuffle(gaps)

        index = rng.randrange(coast)
        slots = []
        for gap in gaps:
            slots.append(COASTAL_CYCLE[index % coast])
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

    def expected_production(self, vertex):
        """Expected cards **per resource, per roll** at ``vertex``, as a tuple of length 5.

        The board already knows which tiles a corner touches and what number each carries;
        this is the two multiplied together and summed. It is the quantity a person means by
        "that spot is an 8 on ore and a 6 on wheat", and the one the observation was missing:
        ``pip potential`` is this list *summed*, which cannot tell three sheep from an even
        spread — see ``docs/decisions/0024-what-a-placement-can-see.md``.

        One definition, used by the encoder and by ``training.alphazero.study``. Two would
        drift, and the one in the study is the one a person reads.

        **Computed once per board.** It is a pure function of the layout, and the layout is
        frozen at construction — but it was being recomputed on every call, and every encode
        calls it once per owned vertex. A tuple rather than a list because the cache hands
        out the same object every time and a caller that mutated it would corrupt the board
        for the rest of the run; all three call sites only read.

        ``check_id`` stays on this side of the cache deliberately. Indexing a per-vertex
        table with an unvalidated id is the ``VERTEX_TILES``/``TILE_VERTICES`` failure mode
        ``CLAUDE.md`` records: ``table[0 - 1]`` is vertex 54's row, which type-checks, runs,
        and is silently wrong.
        """
        return self._expected_production()[check_id(vertex, NUM_VERTICES, "vertex")]

    def _expected_production(self):
        cached = self.__dict__.get("_expected_production_table")
        if cached is None:
            from catan.resources import NUM_RESOURCES

            cached = [None] * (NUM_VERTICES + 1)
            for vertex in range(1, NUM_VERTICES + 1):
                per = [0.0] * NUM_RESOURCES
                for production in self.vertex_production[vertex]:
                    if production.resource is not DESERT:
                        per[int(production.resource)] += roll_odds(production.number)
                cached[vertex] = tuple(per)
            self.__dict__["_expected_production_table"] = cached
        return cached

    def resource_scarcity(self):
        """Expected cards per roll of each resource across the **whole board**.

        What "ore is scarce on this board" means, as five numbers. Balanced generation fixes
        the resource *counts*, but not which numbers they land on, so this genuinely varies
        from board to board.
        """
        from catan.resources import NUM_RESOURCES

        per = [0.0] * NUM_RESOURCES
        for tile in range(1, NUM_TILES + 1):
            resource = self.resource_at(tile)
            if resource is not None:
                per[int(resource)] += roll_odds(self.number_at(tile))
        return per

    def harbour_distances(self):
        """``{vertex: [steps to the nearest harbour of each kind]}``, generic first.

        Distance in **vertices walked along the coast and inland**, which is the number of
        roads it would take to reach a corner carrying that harbour. ``None`` where a kind
        does not exist on this board — three of the nine harbours are resource-specific
        duplicates of nothing, so two resources have no harbour at all on most boards.

        Board geometry and harbour placement are both fixed once generated, so this is
        computed once and cached with everything else that never changes. A vertex knowing
        it *is* a harbour was already encoded; a vertex knowing one is two roads away was
        not, which is most of why a trained agent ignored them.
        """
        cached = self.__dict__.get("_harbour_distances")
        if cached is not None:
            return cached

        from collections import deque

        from catan.resources import NUM_RESOURCES
        from catan.topology import VERTEX_NEIGHBOURS

        kinds = 1 + NUM_RESOURCES                  # generic, then one per resource
        distances = {v: [None] * kinds for v in range(1, NUM_VERTICES + 1)}

        for kind in range(kinds):
            sources = [
                vertex for vertex in range(1, NUM_VERTICES + 1)
                for harbour in self.harbours_at(vertex)
                if (kind == 0 and harbour is GENERIC_HARBOUR)
                or (kind > 0 and harbour is not GENERIC_HARBOUR and int(harbour) == kind - 1)
            ]
            if not sources:
                continue
            seen = {vertex: 0 for vertex in sources}
            queue = deque(sources)
            while queue:
                vertex = queue.popleft()
                for neighbour in VERTEX_NEIGHBOURS[vertex]:
                    if neighbour not in seen:
                        seen[neighbour] = seen[vertex] + 1
                        queue.append(neighbour)
            for vertex, steps in seen.items():
                distances[vertex][kind] = steps

        self.__dict__["_harbour_distances"] = distances
        return distances

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

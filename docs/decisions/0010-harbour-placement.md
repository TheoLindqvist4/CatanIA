# 0010 — Harbours are randomised per board, but evenly spaced

**Status:** accepted · Phase 2

## Context

Real Catan puts the nine harbours in notches on the printed sea frame. This project
generates the board rather than reading physical components, so harbour positions have to
be decided.

`Images/Catan_board.png` does not settle it — the notches visible around its coast are the
sea-frame clips, not harbours.

**What the published rules actually say** (researched rather than assumed):

- There are **9 harbour pieces**: **four 3:1** generic and **five 2:1**, one per resource.
- The benefit requires "a settlement or city on a harbor", giving 3:1, or 2:1 at a harbour
  showing that resource.
- Harbour positions are **"either fixed or randomized depending on your group's
  preference"**, and you may **"shuffle the nine harbor tokens"** and place them on the
  harbours shown on the sea tiles. Board Game Arena exposes a "shuffle frame pieces" option
  for the same purpose.
- The rules **do not name which coastal edges** the printed frame's notches sit on. There is
  no official list to copy.

So randomising harbours is an officially sanctioned setup option, not a deviation — and
since no coastal-edge list is published, there is nothing more faithful available.

## Decision

**Randomise harbour positions per board, constrained to stay evenly spaced.** Both the
starting point on the coastline and the gap arrangement come from the injected RNG, so a
seed still reproduces a board exactly.

`topology.COASTAL_CYCLE` walks the 30 coastal roads as a closed loop. The walk exists
because every perimeter vertex has exactly two coastal roads — the coastline is a single
cycle — and it is canonical: it starts at the lowest-numbered coastal road and leaves by
whichever endpoint leads to the lower-numbered neighbour.

Nine harbours over 30 roads needs gaps summing to 30. Holding every gap to **3 or 4** forces
exactly six 3s and three 4s (6×3 + 3×4 = 30), so `HARBOUR_SPACING` is that multiset,
**shuffled per board**, starting from a random point on the cycle.

Keeping gaps in {3, 4} is what does the work: harbours can never cluster, never leave a
stretch of coast bare, and — because the minimum gap is 3 — **no vertex ever serves two
harbours**, which keeps `trade_rates` simple.

Measured over 4,000 seeds: **280 distinct position sets**, every gap 3 or 4, and every one of
the 30 coastal roads can host a harbour. Times the type shuffle (9 slots, 4 interchangeable
generics → 9!/4! = 15,120 arrangements), there is ample variety for training.

An earlier version fixed the positions and used a repeating 3-3-4 pattern. That yielded only
**10** distinct layouts, because the pattern has period 10 and the set is invariant under a
shift of 10 — most start offsets collapsed onto each other. Shuffling the multiset instead of
rotating a fixed pattern is what widened it to 280.

## Two positions per harbour, and only one is usable

A harbour sits on a coastal **road**, so **both** its endpoint vertices grant it — two spots
to aim for. But those two vertices are adjacent by definition, so the distance rule means the
first building to claim one **excludes the other**. Each harbour therefore ends up serving at
most one player, which is the correct behaviour and is asserted by
`test_only_one_player_can_ever_benefit_from_a_harbour` over full random games.

## Consequences

**Good**

- Sanctioned by the rules, and the most faithful option available given that no coastal-edge
  list is published.
- Harbours vary across games, so an agent cannot overfit to one coastline — 280 position
  layouts rather than a single fixed one.
- Never clustered and never sparse, so no board is unplayably harbour-poor in one region.
- Reproducible from a seed, like everything else.
- Falls out of the generated geometry, so it generalises to other board sizes with no new
  data: change `ROW_LENGTHS` and the coastline walk follows.

**Cost**

- A given board's harbours are not the ones on any particular physical set, so the *value* of
  specific settlement spots differs from a real board. The trading **rules** — rates, counts,
  the both-endpoints grant — are all standard; only the geography varies, and it varies in a
  way the rules explicitly allow.
- If a fixed layout is ever wanted (to match a physical board, or to remove a source of
  variance during evaluation), `_place_harbours` is the only thing to change.

## Alternatives considered

- **A fixed layout guessed at as "official".** Rejected: no published coastal-edge list
  exists, and presenting a guess as official would be worse than a defensible randomisation
  that the rules sanction.
- **Fully random positions, unconstrained.** Rejected: it can cluster several harbours on one
  short stretch and leave a third of the coast bare, which no physical board does. It could
  also put two harbours on roads sharing a vertex, complicating `trade_rates` for no benefit.
- **Harbours on `CORNER_VERTICES`.** Rejected — this is the trap
  [board-geometry.md](../board-geometry.md#6-coastline) warns about. A harbour belongs to a
  coastal *edge* and grants both endpoints; the 18 corner vertices are the wrong set, and the
  12 notch vertices are coastal too.

## Enforced by

`test_there_are_nine_harbours_four_generic_and_one_per_resource`,
`test_harbours_sit_on_coastal_roads`,
`test_harbour_spacing_covers_the_coastline_exactly`,
`test_harbour_positions_are_randomised_by_seed`,
`test_randomised_harbours_are_still_evenly_spaced`,
`test_harbours_are_evenly_spread_and_never_share_a_vertex`,
`test_both_endpoints_of_a_harbour_road_grant_it`,
`test_only_one_player_can_ever_benefit_from_a_harbour`,
`test_harbours_are_part_of_the_board_layout`.

## Sources

- [Catan official rules overview (catan.com)](https://www.catan.com/sites/default/files/2021-06/catan-family-rules-overview.pdf)
- [How to play Catan — official rules (UltraBoardGames)](https://www.ultraboardgames.com/catan/game-rules.php)
- [Setup options / variants (Board Game Arena forum)](https://forum.boardgamearena.com/viewtopic.php?t=26914)
- [Catan rules: setup, turn order, robber, scoring](https://www.catangenerator.org/rule)

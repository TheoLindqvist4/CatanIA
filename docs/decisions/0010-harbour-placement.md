# 0010 — Harbour positions are fixed and evenly spaced; only the types are shuffled

**Status:** accepted, with a caveat · Phase 2 · **revisit if fidelity matters**

## Context

Real Catan puts the nine harbours on the printed sea frame, so both their **positions** and
their **types** are fixed by the physical components. The "variable" setup lets you shuffle
the frame pieces, which moves which harbour is where but not where harbours can be.

This project generates the board rather than reading a physical frame, so harbour positions
have to be decided. `Images/Catan_board.png` does not settle it — the notches visible around
its coast are the sea-frame clips, not harbours.

I do not have the official coastal pattern to hand, and guessing at it and presenting the
guess as official would be worse than choosing something defensible and saying so.

## Decision

Derive the coastline as an ordered walk and space the harbours evenly along it.

`topology.COASTAL_CYCLE` walks the 30 coastal roads as a closed loop. It is canonical:
it starts at the lowest-numbered coastal road and leaves by whichever endpoint leads to the
lower-numbered neighbour, so it is identical on every run. (The walk exists because every
perimeter vertex has exactly two coastal roads — the coastline is a single cycle.)

`Board.HARBOUR_SPACING = (3, 3, 4) * 3` gives the gaps between consecutive harbours.
3 + 3 + 4 = 10, three times over = **exactly 30**, so nine harbours land evenly and the walk
closes without a seam.

The nine types — four generic 3:1 and one 2:1 per resource — are **shuffled** with the
injected RNG. So:

- **positions never vary** — the same nine coastal roads on every board
- **types vary by seed** — which is what Catan's variable setup does

A harbour sits on an *edge*, so **both** its endpoint vertices grant it. With this spacing no
vertex ever serves two harbours, so the 9 harbours occupy 18 distinct vertices.

## Consequences

**Good**

- Harbours are always reachable and never clustered, so no board is unplayably harbour-poor
  in one region.
- Reproducible from a seed, like everything else.
- Falls out of the generated geometry, so it generalises to other board sizes with no new
  data — a 5–6 player board changes `ROW_LENGTHS` and the coastline walk follows.
- The 18 harbour vertices are distinct, which keeps `trade_rates` simple: no vertex can
  grant two harbours.

**⚠️ Not faithful**

The positions are almost certainly not the official ones. Consequences for training:

- The *value* of specific board positions differs from a real board, so an agent's learned
  preference for particular settlement spots will not transfer exactly.
- The *rules* of trading are unaffected — rates, counts and the both-endpoints rule are all
  standard. Only the geography differs.

To fix it, replace the derived slots with an explicit list of the official coastal roads.
`_place_harbours` is the only thing to change, and the spacing test is the only test that
would need updating. Worth doing before any evaluation against a real board or a human.

## Alternatives considered

- **Random harbour positions per seed.** Rejected: it adds variance the real game does not
  have, and can produce clustered or harbour-starved coasts.
- **Harbours on `CORNER_VERTICES`.** Rejected — this is the trap
  [board-geometry.md](../board-geometry.md#6-coastline) warns about. A harbour belongs to a
  coastal *edge* and grants both its endpoints; the 18 corner vertices are the wrong set,
  and the 12 notch vertices are coastal too.

## Enforced by

`test_there_are_nine_harbours_four_generic_and_one_per_resource`,
`test_harbours_sit_on_coastal_roads`,
`test_harbour_spacing_covers_the_coastline_exactly`,
`test_harbours_are_evenly_spread_and_never_share_a_vertex`,
`test_both_endpoints_of_a_harbour_road_grant_it`,
`test_harbour_positions_are_fixed_only_the_types_move`,
`test_harbours_are_part_of_the_board_layout`.

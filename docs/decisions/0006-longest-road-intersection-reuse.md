# 0006 — Longest road: may a route reuse an intersection?

**Status:** ✅ **accepted — strict simple path** · implemented in Phase 1

## Context

`find_longest_path` searches for the longest chain of a player's roads. It tracks which
**roads** have been used and never reuses one, but it does not track which **intersections**
have been visited.

Because no vertex on a hex board has more than three roads, that distinction only bites in
one shape: a vertex where the player owns **all three** roads. A route can then enter and
leave it twice.

Concrete case, road set `{7, 11, 12, 19, 20, 25, 26, 27}`:

```
road 7  = (4, 8)     road 20 = (13, 18)
road 11 = (8, 12)    road 25 = (17, 23)
road 12 = (8, 13)    road 26 = (18, 23)
road 19 = (12, 17)   road 27 = (18, 24)

vertex degrees within the player's network:
  4:1   8:3   12:2   13:2   17:2   18:3   23:2   24:1
```

- **Intersections may repeat → 7.** `4 → 8 → 13 → 18 → 23 → 17 → 12 → 8`, using roads
  7, 12, 20, 26, 25, 19, 11. Vertex 8 is passed through twice; all three of its roads are
  consumed.
- **Strict simple path → 6.** Stopping at `4 → 8 → 13 → 18 → 23 → 17 → 12`.

Both were computed independently to confirm the gap is real, not an artefact.

The rulebook says only "the longest continuous route of roads" and that branches do not
count. It does not address this case, and published implementations differ.

## Decision

**Strict simple path.** A route may not pass through the same intersection twice. The
decided case answers **6**.

Implemented in `catan.rules.longest_road_length`, together with the related rule that had
simply never been written: **an opponent's building breaks a road.** A chain may *end* at
an opponent's settlement or city but may not continue through it. Both were done in one
pass because both change the same search, and ownership — which Phase 1 introduced — is
what made the second one possible at all.

The legacy `Game_2_players.find_longest_path` was updated to the strict reading too, so
the repository has one ruling. It cannot apply the opponent-break rule, having no ownership
information; `catan.rules` is the version that counts.

### Implementation note

The search starts from each road in each direction rather than from each vertex. That
treats the starting vertex as a free endpoint, which matters: an opponent's building at the
*end* of a chain must not shorten it, only one in the *middle*. Starting from vertices
instead would have under-counted any chain whose endpoints were both blocked.

## ⚠️ Consequence: a closed loop counts as one less than its roads

This was not in the options above and is worth knowing.

The six roads around a single hex form a cycle over six vertices. A simple path can visit
all six vertices but only **five** of the roads — closing the loop would revisit the start.
So a player who rings a hex scores 5, not 6. The roads-only reading would have said 6.

Pinned by `test_a_closed_loop_of_six_roads_counts_as_five`. A loop plus any one extra road
scores 6, because the tail gives the path somewhere to end.

Whether that is desirable is a fair question — "walk all six roads continuously" is an
intuitive reading of the rulebook's "longest continuous route". It is *consistent* with the
ruling taken, which is why it stands. If it should change, this record and that test are
the two places to change.

## Alternatives rejected

- **Roads-only** (the original behaviour). Answers 7 on the decided case, and 6 on a loop.
  Simpler, and arguably a literal reading of "you cannot use the same road twice", but it
  lets a route walk through one intersection twice, which does not correspond to anything
  physical on the board.
- **Configurable, defaulting to strict.** Two rule sets means two things to train against
  and two sets of learned position values, for no gain once a ruling is picked.

## Enforced by

`tests/test_longest_road.py` — in particular
`test_a_route_may_not_reuse_an_intersection`,
`test_a_closed_loop_of_six_roads_counts_as_five`,
`test_an_opponent_building_in_the_middle_splits_the_chain`,
`test_an_opponent_building_at_an_endpoint_does_not_shorten_anything`,
`test_your_own_buildings_never_break_your_road`.

Awarding the 2 victory points — the 5-segment minimum and keep-until-beaten — remains
Phase 2. `test_the_award_itself_is_not_implemented_yet` pins that a long road does not yet
affect the score.

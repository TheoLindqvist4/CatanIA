# 0003 — I/O-free core, injected randomness, instance state

**Status:** accepted · Phase 0

## Context

Three properties are non-negotiable for training an agent, and none of them held.

1. **`Board.__init__` printed.** It called `display_board()` and dumped the whole
   intersection dictionary to stdout. A training loop building millions of boards would
   have flooded stdout and paid the formatting cost every time.
2. **Importing `Game_2_players` played a game.** A module-level `game = Game_2_players()`
   meant `import Game_2_players` emitted 2,395 characters and ran a full scripted game —
   verified. Any test harness or training script importing the module inherited that.
3. **State was shared between instances.** `player_order` and `turn_number` on `Game`,
   `dice_value` on `Dice`, and every count in `Deck` were *class* attributes. Verified:
   `g1.player_order is g2.player_order → True`, and `random.shuffle` mutated the class list
   in place. Two concurrent games would have silently reordered each other's turns.

Additionally, randomness came from the global `random` module, so episodes could not be
reproduced and unrelated games shared one stream.

## Decision

- **No I/O in the core.** `display_board()` returns a string. Nothing under `Board`,
  `Dice`, `Deck` or `topology` writes to stdout. `print`/`input` are confined to the
  interactive placement methods and `demo()` for now, and move to `interfaces/cli.py` in
  Phase 4.
- **`if __name__ == "__main__"` guard.** Constructing a `Game` sets up a game; it does not
  play one. The old scripted flow lives in `demo()`.
- **All mutable state per-instance.**
- **Randomness injected.** `Board`, `Dice` and `Game` take a `random.Random`. `Game`
  accepts a `seed` and shares one generator with the board and dice, so a whole game is
  reproducible from one integer.
- **Deterministic enumeration.** Every legal-move list is sorted before being returned.
  `list(set(...))` ordering is arbitrary, which would have made rollouts irreproducible
  even with a fixed seed.
- **Bounded retries.** Board generation is greedy and can dead-end. It retries up to
  `max_generation_attempts` (default 100; empirically a median of 2 and a max of 5 are
  needed) and then raises, rather than looping forever.

## Consequences

**Good**

- Parallel self-play is safe: no cross-instance state.
- A seed reproduces a board, a turn order and a dice sequence exactly.
- Tests can assert the core is silent — `test_constructing_a_board_prints_nothing` and
  `test_importing_the_game_module_does_not_play_a_game` are regressions, not style checks.
- A malformed constraint set fails loudly instead of hanging.

**Cost**

- Callers must thread an `rng` through if they want reproducibility. Omitting it still
  works and produces an unseeded generator, which is the right default for a human playing
  interactively.

## Enforced by

`test_constructing_a_board_prints_nothing`,
`test_display_board_returns_a_string_instead_of_printing`,
`test_importing_the_game_module_does_not_play_a_game`,
`test_board_does_not_consume_the_global_random_stream`,
`test_game_does_not_consume_the_global_random_stream`,
`test_class_attributes_no_longer_exist`,
`test_player_order_is_not_shared_between_games`,
`test_legal_move_enumeration_is_sorted_for_reproducibility`,
`test_generation_failure_is_bounded_not_a_hang`.

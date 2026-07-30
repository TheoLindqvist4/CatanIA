"""Two-player game driver.

.. deprecated::
   Superseded by :mod:`catan.rules` + :class:`catan.state.GameState`. Kept only so the
   legacy terminal demo keeps working; it is deleted when ``interfaces/cli.py`` lands in
   Phase 4. Do not add to it — new rules go in :mod:`catan.rules`.

Still transitional: Phase 1 splits this into a pure ``rules.py`` (``legal_actions`` /
``apply``) plus a ``GameState``, and moves every ``print``/``input`` into
``interfaces/cli.py``. What Phase 0 changed here:

* ``player_order`` and ``turn_number`` were *class* attributes, so ``random.shuffle``
  mutated state shared by every game instance in the process — fatal for parallel
  self-play. They are now per-instance.
* Constructing a game no longer *plays* one, and importing this module no longer runs
  a game. The old hardcoded script lives in :meth:`demo`.
* Randomness comes from one injected, seedable generator.
* Legal-move enumeration returns sorted lists, so rollouts are reproducible.

Still outstanding, by design (see ROADMAP.md): building costs no resources, and
``place_road``/``place_settlement`` do not check connectivity.
"""

import random

from Board import Board
from Deck import Deck
from Dice import Dice
from Player import Player
from catan.topology import ROAD_VERTICES


class Game_2_players:
    def __init__(self, seed=None, rng=None):
        """
        Args:
            seed: seed for a fresh generator. Use this for reproducible games.
            rng: an existing ``random.Random`` to share instead.
        """
        self.rng = rng if rng is not None else random.Random(seed)

        self.player_order = [1, 2]
        self.turn_number = 0

        self.players = {
            1: Player(),
            2: Player(),
        }
        self.board = Board(rng=self.rng)
        self.deck = Deck()
        self.dice_1 = Dice(rng=self.rng)
        self.dice_2 = Dice(rng=self.rng)

    def randomize_order(self):
        self.rng.shuffle(self.player_order)
        return list(self.player_order)

    # ------------------------------------------------------------------ #
    # PLACING SETTLEMENTS AND ROADS                                      #
    # ------------------------------------------------------------------ #

    def place_settlement(self, player_number, position):
        player = self.players[player_number]

        if player.player_settlement <= 0:
            return "You do not have any more settlements."

        is_valid, message = self.check_if_position_settlement_is_valid(position)
        if not is_valid:
            return message

        player.player_settlement -= 1
        self.board.delete_settlement_position(position)
        player.player_settlement_position.add(position)

        return f"Settlement placed at position {position}."

    def place_road(self, player_number, position):
        player = self.players[player_number]

        if player.player_roads <= 0:
            return "You do not have any more roads."

        is_valid, message = self.check_if_position_road_is_valid(position)
        if not is_valid:
            return message

        player.player_roads -= 1
        self.board.delete_road_position(position)
        player.player_road_position.add(position)

        return f"Road placed at position {position}."

    # ------------------------------------------------------------------ #
    # CHECKING IF THE POSITION IS VALID                                  #
    # ------------------------------------------------------------------ #

    def check_if_position_settlement_is_valid(self, position):
        if position not in self.board.settlement_positions:
            return False, "You cannot put a settlement in this position."
        return True, "Valid position for settlement."

    def check_if_position_road_is_valid(self, position):
        if position not in self.board.road_positions:
            return False, "You cannot put a road in this position."
        return True, "Valid position for road."

    def check_valid_settlement_once_game_has_begun(self, player):
        """Vertices at the end of one of ``player``'s roads that are still buildable."""
        valid_settlement_positions = set()

        for road_position in self.players[player].player_road_position:
            for settlement_position in self.board.get_adjacent_settlement_from_road(road_position):
                is_valid, _ = self.check_if_position_settlement_is_valid(settlement_position)
                if is_valid:
                    valid_settlement_positions.add(settlement_position)

        return sorted(valid_settlement_positions)

    def check_valid_road_once_game_has_begun(self, player):
        """Roads adjacent to one of ``player``'s roads that are still buildable."""
        valid_road_positions = set()

        for road_position in self.players[player].player_road_position:
            for candidate in self.board.get_adjacent_roads_from_road(road_position):
                is_valid, _ = self.check_if_position_road_is_valid(candidate)
                if is_valid:
                    valid_road_positions.add(candidate)

        return sorted(valid_road_positions)

    # ------------------------------------------------------------------ #
    # FIRST TURN OF THE GAME, PLACING THE SETTLEMENTS                    #
    # ------------------------------------------------------------------ #

    def _place_starting_pair(self, player_num):
        """Prompt ``player_num`` for one settlement and an adjoining road."""
        while True:
            print(f"PLAYER {player_num}, PLEASE CHOOSE A SETTLEMENT POSITION ON THE BOARD:")
            print("Available settlements:", self.board.get_available_settlements())

            position = self.get_user_number()
            settlement_message = self.place_settlement(player_num, position)
            print(settlement_message)
            if "Settlement placed" not in settlement_message:
                continue
            print(f"you have now a total of= {self.players[player_num].player_settlement} settlements")

            print(f"PLAYER {player_num}, PLEASE CHOOSE A ROAD POSITION ON THE BOARD:")
            available_roads = self.board.get_available_road_from_settlement(position)
            print("Available roads:", available_roads)

            # Loop until the road is genuinely placed. This used to break out
            # unconditionally, so a rejected choice cost the player their road.
            while True:
                road_position = self.get_user_number()
                if road_position not in available_roads:
                    print("Invalid choice. Please choose a valid road position from the list:",
                          available_roads)
                    continue
                road_message = self.place_road(player_num, road_position)
                print(road_message)
                if "Road placed" in road_message:
                    print("")
                    return

    def placing_first_settlement(self):
        for player_num in self.player_order:
            self._place_starting_pair(player_num)
            print("")

    def placing_second_settlement(self):
        for player_num in reversed(self.player_order):
            self._place_starting_pair(player_num)
            print("")

    def print_order(self):
        print("Player turn order:", self.player_order)

    # ------------------------------------------------------------------ #
    # TURNS                                                              #
    # ------------------------------------------------------------------ #

    def turn(self, turn_number):
        players_turn = self.whos_turn_is_it(turn_number)
        print("")
        print(f"Player {players_turn}'s turn")
        self.dice_1.roll_dice()
        self.dice_2.roll_dice()
        total_dice = self.dice_1.dice_value + self.dice_2.dice_value
        self.give_cards_to_players(total_dice)
        print(f"The total of the dice is: {total_dice}")

        can_build_road, can_build_settlement, can_build_city, can_buy_dev_cards = \
            self.players[players_turn].check_player_actions()

        for label, allowed in (
            ("build a road", can_build_road),
            ("build a settlement", can_build_settlement),
            ("build a city", can_build_city),
            ("buy development cards", can_buy_dev_cards),
        ):
            print(f"You {'can' if allowed else 'cannot'} {label}")

        self.turn_number += 1

    def whos_turn_is_it(self, turn_number):
        return self.player_order[turn_number % len(self.player_order)]

    def give_cards_to_players(self, dice_number):
        """Pay out production for ``dice_number``.

        The board already excludes the desert from its payout index, so a 7 pays
        nobody without a special case here.

        Phase 1: cities are ignored (``player_city_position`` is never read, so a city
        would produce nothing), and the bank is unbounded.
        """
        producers = self.board.producers_for(dice_number)

        for player_number, player in self.players.items():
            for position, productions in producers.items():
                if position not in player.player_settlement_position:
                    continue
                for production in productions:
                    player.player_ressources[production.resource] += 1
                    print(f"Player {player_number} receives 1 {production.resource} "
                          f"for settlement at position {position}")

    # ------------------------------------------------------------------ #
    # LONGEST ROAD                                                       #
    # ------------------------------------------------------------------ #

    def find_longest_path(self, player_number):
        """Longest chain of ``player_number``'s roads, ignoring branches.

        **Strict simple path**: a route may not pass through the same intersection
        twice. Settled in ``docs/decisions/0006-longest-road-intersection-reuse.md``.
        For ``{7, 11, 12, 19, 20, 25, 26, 27}`` this gives 6; the earlier
        roads-only version gave 7 by passing through vertex 8 three times.

        This module has no ownership information, so it cannot apply the rule that an
        opponent's building breaks a road. :func:`catan.rules.longest_road_length`
        implements both and is the version that counts.
        """
        player_roads = self.players[player_number].get_player_road_positions()

        graph = {}
        for road in player_roads:
            u, v = ROAD_VERTICES[road]
            graph.setdefault(u, []).append((v, road))
            graph.setdefault(v, []).append((u, road))

        def dfs(intersection, used_roads, visited):
            longest_here = 0
            for neighbour, road in graph[intersection]:
                if road in used_roads or neighbour in visited:
                    continue
                used_roads.add(road)
                visited.add(neighbour)
                longest_here = max(longest_here, 1 + dfs(neighbour, used_roads, visited))
                visited.discard(neighbour)
                used_roads.discard(road)
            return longest_here

        return max(
            (dfs(start, set(), {start}) for start in graph),
            default=0,
        )

    # ------------------------------------------------------------------ #
    # DEMO / CLI                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_user_number():
        while True:
            try:
                return int(input("Enter a number: "))
            except ValueError:
                print("Invalid input. Please enter a valid number.")

    def demo(self):
        """The old hardcoded smoke script, kept runnable but off the import path."""
        print(self.board.display_board())
        print("")

        for road in (11, 12, 20, 26, 25, 19, 27, 7):
            self.place_road(1, road)
        for road in (3, 16):
            self.place_road(2, road)

        for player in (1, 2):
            print(f"Player {player} has "
                  f"{self.check_valid_settlement_once_game_has_begun(player)} valid positions")
        for player in (1, 2):
            print(f"Player {player} has "
                  f"{self.check_valid_road_once_game_has_begun(player)} valid road positions")

        print("")
        for player in (1, 2):
            roads = sorted(self.players[player].get_player_road_positions())
            print(f"Player {player} roads: {roads}")
            print(f"Player {player} has a max length road of: {self.find_longest_path(player)}")


if __name__ == "__main__":
    Game_2_players(seed=0).demo()

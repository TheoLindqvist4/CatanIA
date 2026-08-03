"""What one player may see of a game.

A heuristic agent needs the board — where the tiles are, who owns what, what it holds. It
must **not** see the opponent's cards, or either deck. The observation vector already
enforces that, but decoding `encoder.SIZE` floats back into meaning to write a heuristic
would be absurd.

So :class:`PublicView` wraps the state and exposes an **explicit allow-list**. Anything not
listed simply is not reachable, which makes cheating impossible rather than merely tested
for — the same reasoning as filtering the web responses server-side, and as generating the
geometry instead of hand-writing it.

    view = PublicView(state, me)
    view.board                 # fine, the board is on the table
    view.hand_size(opponent)   # fine, cards are countable
    view.my_hand               # fine, they are mine
    view.state                 # AttributeError
    view.dev_deck              # AttributeError

Constructing one is a couple of attribute assignments, so the environment can hand a fresh
view to every agent on every step.
"""

from catan import rules
from catan.resources import total

#: Fields of :class:`~catan.state.GameState` that are public knowledge and forwarded as-is.
FORWARDED = frozenset({
    "board", "ruleset", "num_players", "player_order",
    "vertex_owner", "vertex_piece", "edge_owner",
    "bank", "robber_tile",
    "phase", "turn_number", "last_roll", "rolled_this_turn",
    "knights_played", "settlements_left", "cities_left", "roads_left",
    "largest_army_holder", "longest_road_holder",
    "free_roads", "dev_card_played_this_turn", "discards_owed",
    "pending_discards", "winner", "setup_step", "last_settlement",
})


class PublicView:
    """A read-only window onto the game, from ``me``'s point of view."""

    __slots__ = ("_state", "me")

    def __init__(self, state, me):
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "me", me)

    # -- forwarding ---------------------------------------------------- #

    def __getattr__(self, name):
        if name in FORWARDED:
            return getattr(self._state, name)
        raise AttributeError(
            f"{name!r} is not public — a player may not see it. "
            f"Public fields: {', '.join(sorted(FORWARDED))}"
        )

    def __setattr__(self, name, value):
        raise AttributeError("a view is read-only")

    # -- players ------------------------------------------------------- #

    @property
    def players(self):
        return self._state.players

    @property
    def opponents(self):
        return tuple(p for p in self._state.players if p != self.me)

    # -- cards: mine in full, theirs by count -------------------------- #

    @property
    def my_hand(self):
        """My resource cards, as counts indexed by Resource."""
        return list(self._state.hands[self.me])

    @property
    def my_dev_cards(self):
        return list(self._state.dev_cards[self.me])

    @property
    def my_playable_dev_cards(self):
        return rules.playable_dev_cards(self._state, self.me)

    def hand_size(self, player):
        """How many resource cards a player holds. Public — cards are countable."""
        return total(self._state.hands[player])

    def dev_card_count(self, player):
        """How many development cards a player holds, not which."""
        return sum(self._state.dev_cards[player])

    @property
    def dev_deck_size(self):
        """How many cards are left to buy. Public; the order is not."""
        return len(self._state.dev_deck)

    # -- derived, all from public information -------------------------- #

    def public_victory_points(self, player):
        return rules.public_victory_points(self._state, player)

    def longest_road(self, player):
        return rules.longest_road_length(self._state, player)

    def trade_rates(self, player=None):
        return rules.trade_rates(self._state, self.me if player is None else player)

    def buildings_of(self, player):
        return self._state.buildings_of(player)

    def roads_of(self, player):
        return self._state.roads_of(player)

    def __repr__(self):
        return f"PublicView(player={self.me}, turn={self._state.turn_number})"

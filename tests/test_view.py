"""The public window onto a game.

The point of :class:`PublicView` is what it *refuses*, so most of this file is about
things that must raise.
"""

import pytest

from catan import rules
from catan.dev_cards import DevCard
from catan.env import CatanEnv
from catan.resources import Resource, total
from catan.state import GameState
from catan.view import FORWARDED, PublicView


@pytest.fixture
def game():
    """A game a few turns in, so the view has something to show."""
    env = CatanEnv(num_players=2)
    _, info = env.reset(seed=11)
    for _ in range(120):
        if info["done"]:
            break
        _, _, _, _, info = env.step(info["legal"][-1])
    return env.state


# =========================================================================== #
# WHAT IT REFUSES                                                             #
# =========================================================================== #

HIDDEN = ["hands", "dev_cards", "dev_deck", "dice_deck", "events", "rng"]


@pytest.mark.parametrize("field", HIDDEN)
def test_hidden_state_is_not_reachable(game, field):
    view = PublicView(game, 1)
    with pytest.raises(AttributeError, match="not public"):
        getattr(view, field)


def test_the_state_itself_is_not_reachable(game):
    """Otherwise the allow-list is one attribute deep and means nothing."""
    view = PublicView(game, 1)
    for escape in ("state", "_state" if "_state" in FORWARDED else "game"):
        with pytest.raises(AttributeError):
            getattr(view, escape)


def test_every_hidden_field_of_the_state_is_either_forwarded_deliberately_or_refused(game):
    """A new field on GameState defaults to hidden — it has to be listed to be seen.

    This is the test that keeps the allow-list honest as the engine grows.
    """
    view = PublicView(game, 1)
    for field in GameState.__slots__ if hasattr(GameState, "__slots__") else vars(game):
        if field.startswith("_") or field in FORWARDED:
            continue
        if hasattr(type(view), field):        # deliberately reimplemented, e.g. players
            continue
        with pytest.raises(AttributeError):
            getattr(view, field)


def test_a_view_is_read_only(game):
    view = PublicView(game, 1)
    with pytest.raises(AttributeError, match="read-only"):
        view.robber_tile = 5
    with pytest.raises(AttributeError, match="read-only"):
        view.anything_at_all = 5


def test_my_hand_is_a_copy(game):
    """Handing out the live list would let an agent edit the game it is playing."""
    view = PublicView(game, 1)
    before = list(game.hands[1])
    view.my_hand[Resource.WOOD] += 99
    assert list(game.hands[1]) == before


# =========================================================================== #
# WHAT IT SHOWS                                                               #
# =========================================================================== #

@pytest.mark.parametrize("field", sorted(FORWARDED))
def test_forwarded_fields_are_the_state_s_own(game, field):
    assert getattr(PublicView(game, 1), field) is getattr(game, field)


def test_cards_are_countable_but_not_readable(game):
    view = PublicView(game, 1)
    for opponent in view.opponents:
        assert view.hand_size(opponent) == total(game.hands[opponent])
        assert view.dev_card_count(opponent) == sum(game.dev_cards[opponent])
    assert view.dev_deck_size == len(game.dev_deck)


def test_my_own_cards_are_mine_to_read(game):
    view = PublicView(game, 1)
    assert view.my_hand == list(game.hands[1])
    assert view.my_dev_cards == list(game.dev_cards[1])
    assert view.my_playable_dev_cards == rules.playable_dev_cards(game, 1)


def test_opponents_excludes_me(game):
    assert PublicView(game, 1).opponents == (2,)
    assert PublicView(game, 2).opponents == (1,)


def test_derived_values_agree_with_the_rules(game):
    view = PublicView(game, 1)
    for player in (1, 2):
        assert view.public_victory_points(player) == rules.public_victory_points(game, player)
        assert view.longest_road(player) == rules.longest_road_length(game, player)
        assert view.buildings_of(player) == game.buildings_of(player)
        assert view.roads_of(player) == game.roads_of(player)
    assert view.trade_rates() == rules.trade_rates(game, 1)
    assert view.trade_rates(2) == rules.trade_rates(game, 2)


def test_public_points_hide_victory_point_cards(game):
    """The whole reason a separate 'public' score exists."""
    game.dev_cards[2][DevCard.VICTORY_POINT] += 3
    view = PublicView(game, 1)
    assert view.public_victory_points(2) == rules.victory_points(game, 2) - 3


def test_the_env_hands_every_agent_a_view():
    env = CatanEnv(num_players=2)
    _, info = env.reset(seed=3)
    assert isinstance(info["view"], PublicView)
    assert info["view"].me == info["player"]

    _, _, _, _, info = env.step(info["legal"][0])
    assert info["view"].me == info["player"], "the view must follow whoever is to move"


def test_constructing_one_is_cheap():
    """It happens on every step of every agent, so it may not do work."""
    import timeit

    env = CatanEnv(num_players=2)
    env.reset(seed=1)
    seconds = timeit.timeit(lambda: PublicView(env.state, 1), number=20_000) / 20_000
    assert seconds < 5e-6, f"{seconds * 1e9:.0f} ns to build a view"

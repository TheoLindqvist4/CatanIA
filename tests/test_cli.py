"""The terminal interface.

The CLI is the only module allowed to print, so these tests are mostly about the board
drawing and the argument handling — the game itself is covered elsewhere.
"""

import re

import pytest

import catan.topology as T
from catan.actions import ActionType
from catan.dev_cards import DevCard
from catan.rulesets import BASE_GAME, RANKED_1V1
from catan.state import Piece
from helpers import fresh, give, put_building, put_road

cli = pytest.importorskip("interfaces.cli")

ANSI_PATTERN = re.compile(r"\033\[[0-9;]*m")


# =========================================================================== #
# THE BOARD                                                                   #
# =========================================================================== #

def test_the_board_is_a_rectangle_of_text():
    board = cli.text_board(fresh(seed=1), colour=False)
    lines = board.split("\n")
    assert len(lines) > 20
    assert any(line.strip() for line in lines)


def test_every_tile_appears_with_its_resource_and_number():
    state = fresh(seed=1)
    board = cli.text_board(state, colour=False)
    for tile in range(1, T.NUM_TILES + 1):
        resource = state.board.resource_at(tile)
        code = cli.RESOURCE_CODES[resource]
        assert code in board, f"tile {tile} ({code}) missing"
        if resource is not None:
            assert f"{code}{state.board.number_at(tile):>2}" in board


def test_the_desert_shows_no_number():
    state = fresh(seed=1)
    board = cli.text_board(state, colour=False)
    # the desert carries a 7 internally; it must not be printed
    assert f"{cli.RESOURCE_CODES[None]} 7" not in board


def test_the_robber_is_marked():
    state = fresh(seed=1)
    assert "(R)" in cli.text_board(state, colour=False)


def test_a_settlement_shows_as_its_player_digit_and_a_city_as_a_letter():
    state = fresh(seed=1)
    before = cli.text_board(state, colour=False)

    put_building(state, 2, 20, Piece.SETTLEMENT)
    with_settlement = cli.text_board(state, colour=False)
    assert with_settlement != before
    assert "2" in with_settlement

    state.vertex_piece[20] = Piece.CITY
    with_city = cli.text_board(state, colour=False)
    assert with_city != with_settlement
    assert cli.CITY_LETTERS[1] in with_city


def test_roads_are_drawn():
    state = fresh(seed=1)
    plain = cli.text_board(state, colour=False)
    put_road(state, 1, 30)
    assert cli.text_board(state, colour=False) != plain


def test_a_vertical_road_draws_a_pipe():
    state = fresh(seed=1)
    vertical = next(
        road for road in range(1, T.NUM_ROADS + 1)
        if T.VERTEX_XY[T.ROAD_VERTICES[road][0]][0]
        == T.VERTEX_XY[T.ROAD_VERTICES[road][1]][0]
    )
    put_road(state, 1, vertical)
    assert "|" in cli.text_board(state, colour=False)


def test_a_slanted_road_draws_a_slash():
    state = fresh(seed=1)
    slanted = next(
        road for road in range(1, T.NUM_ROADS + 1)
        if T.VERTEX_XY[T.ROAD_VERTICES[road][0]][0]
        != T.VERTEX_XY[T.ROAD_VERTICES[road][1]][0]
    )
    put_road(state, 1, slanted)
    board = cli.text_board(state, colour=False)
    assert "/" in board or "\\" in board


def test_buildable_spots_can_be_marked():
    state = fresh(seed=1)
    plain = cli.text_board(state, colour=False)
    marked = cli.text_board(state, colour=False, spots_for=1)
    assert "." in marked and "." not in plain


def test_colour_can_be_turned_off():
    state = fresh(seed=1)
    put_building(state, 1, 20)
    put_road(state, 1, 30)

    assert ANSI_PATTERN.search(cli.text_board(state, colour=True))
    assert not ANSI_PATTERN.search(cli.text_board(state, colour=False))


def test_colour_does_not_change_the_layout():
    """Stripping the escapes should give back exactly the plain board."""
    state = fresh(seed=1)
    put_building(state, 1, 20)
    put_road(state, 1, 30)

    coloured = ANSI_PATTERN.sub("", cli.text_board(state, colour=True))
    plain = cli.text_board(state, colour=False)
    assert [line.rstrip() for line in coloured.split("\n")] == \
        [line.rstrip() for line in plain.split("\n")]


def test_the_board_draws_at_every_player_count():
    for count in (2, 3, 4):
        state = fresh(seed=1, num_players=count)
        assert cli.text_board(state, colour=False)


# =========================================================================== #
# STATUS LINES                                                                #
# =========================================================================== #

def test_a_summary_hides_what_an_opponent_should_not_see():
    state = fresh(seed=1)
    give(state, 1, wood=3, ore=2)
    state.dev_cards[1][DevCard.VICTORY_POINT] = 2

    hidden = cli.player_summary(state, 1, reveal=False)
    assert "5 cards" in hidden
    assert "wood" not in hidden and "victory" not in hidden

    shown = cli.player_summary(state, 1, reveal=True)
    assert "victory_point" in shown


def test_a_summary_reports_public_points_when_hiding():
    state = fresh(seed=1)
    put_building(state, 1, 20, Piece.CITY)
    state.dev_cards[1][DevCard.VICTORY_POINT] = 3

    assert "public vp 2" in cli.player_summary(state, 1, reveal=False)
    assert "vp 5" in cli.player_summary(state, 1, reveal=True)


def test_a_summary_flags_the_awards():
    state = fresh(seed=1)
    state.largest_army_holder = 1
    state.longest_road_holder = 1
    line = cli.player_summary(state, 1)
    assert "[army]" in line and "[road]" in line


def test_an_empty_hand_reads_as_nothing():
    state = fresh(seed=1)
    assert cli.hand_summary(state, 1) == "nothing"


# =========================================================================== #
# ACTION DESCRIPTIONS                                                         #
# =========================================================================== #

def test_every_action_can_be_described():
    """An action with no description would show as a blank choice to a human."""
    from catan import action_space
    for index in range(action_space.NUM_ACTIONS):
        text = cli.describe(index)
        assert isinstance(text, str) and text.strip()
        assert "_" not in text or "victory" in text, f"unpolished label: {text!r}"


def test_descriptions_name_the_thing_being_built():
    from catan import action_space
    from catan.actions import build_city, build_road, build_settlement, end_turn

    assert cli.describe(action_space.encode(end_turn())) == "end turn"
    assert "settlement at 20" in cli.describe(action_space.encode(build_settlement(20)))
    assert "city at 20" in cli.describe(action_space.encode(build_city(20)))
    assert cli.describe(action_space.encode(build_road(5))).startswith("road at 5")


def test_a_road_description_includes_its_endpoints():
    """A road id alone means nothing to a person; the vertices place it."""
    from catan import action_space
    from catan.actions import build_road
    text = cli.describe(action_space.encode(build_road(5)))
    assert str(T.ROAD_VERTICES[5][0]) in text and str(T.ROAD_VERTICES[5][1]) in text


# =========================================================================== #
# COMMAND LINE                                                                #
# =========================================================================== #

@pytest.mark.slow
def test_a_game_between_agents_runs_to_a_result(capsys):
    assert cli.main(["--agents", "greedy", "random", "--seed", "3", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "wins with" in out or "truncated" in out


def test_quiet_mode_prints_almost_nothing(capsys):
    cli.main(["--agents", "random", "random", "--seed", "1", "--quiet",
              "--max-turns", "20"])
    assert len(capsys.readouterr().out.strip().split("\n")) <= 3


@pytest.mark.slow
def test_several_games_are_tallied(capsys):
    cli.main(["--agents", "greedy", "random", "--seed", "1", "--games", "3", "--quiet"])
    out = capsys.readouterr().out
    assert "P1 (greedy)" in out and "P2 (random)" in out


@pytest.mark.parametrize("rules_flag", ["ranked1v1", "base"])
def test_both_rulesets_are_selectable(rules_flag, capsys):
    assert cli.main(["--agents", "random", "random", "--seed", "2", "--max-turns", "20",
                     "--rules", rules_flag, "--quiet"]) == 0


def test_an_unknown_agent_is_rejected():
    with pytest.raises(SystemExit):
        cli.main(["--agents", "greedy", "genius"])


@pytest.mark.parametrize("seats", [["greedy"], ["a", "b", "c", "d", "e"]])
def test_impossible_player_counts_are_rejected(seats):
    with pytest.raises(SystemExit):
        cli.main(["--agents", *seats])


def test_a_human_cannot_be_asked_to_play_many_games_at_once():
    with pytest.raises(SystemExit):
        cli.main(["--agents", "human", "greedy", "--games", "5"])


@pytest.mark.slow
def test_a_seed_makes_a_watched_game_reproducible(capsys):
    cli.main(["--agents", "greedy", "random", "--seed", "9", "--quiet"])
    first = capsys.readouterr().out
    cli.main(["--agents", "greedy", "random", "--seed", "9", "--quiet"])
    assert capsys.readouterr().out == first


def test_rendering_to_a_directory_writes_frames(tmp_path, capsys):
    pytest.importorskip("PIL")
    target = tmp_path / "frames"
    # capped, because a whole game would write hundreds of PNGs
    cli.main(["--agents", "random", "random", "--seed", "4", "--quiet",
              "--max-turns", "12", "--render", str(target)])
    frames = sorted(target.glob("turn_*.png"))
    assert len(frames) > 10
    assert all(frame.stat().st_size > 0 for frame in frames)

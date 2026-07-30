"""The web interface: the framework-free API layer, and the HTTP shim over it.

The load-bearing test is :func:`test_no_response_ever_leaks_the_opponents_cards`. Anything
in the JSON is in the browser, and someone will read it.
"""

import json
import random
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

import catan.topology as T
from catan import action_space, rules
from catan.dev_cards import DevCard
from catan.resources import Resource
from catan.state import Phase

api = pytest.importorskip("interfaces.web.api")
from interfaces.web.server import Handler, safe_path  # noqa: E402


def fresh_game(seed=5, **kwargs):
    return api.Game(seed=seed, **kwargs)


def play_out(game, limit=4000, rng=None):
    """Click through a game the way the client would, returning the final view."""
    rng = rng or random.Random(0)
    view = game.view()
    for _ in range(limit):
        if view["done"]:
            break
        board = [i for targets in view["actions"]["board"].values() for i in targets.values()]
        panel = [entry["index"] for entry in view["actions"]["panel"]]
        options = board + panel
        assert options, f"nothing to do in {view['phase']}"
        view = game.play(rng.choice(options))
    return view


# =========================================================================== #
# HIDDEN INFORMATION — the leak test                                          #
# =========================================================================== #

def test_no_response_ever_leaks_the_opponents_cards():
    """Walk a whole game and inspect every response the human would receive."""
    game = fresh_game(seed=7)
    rng = random.Random(7)
    view = game.view()

    for _ in range(1500):
        opponent = next(p for p in view["players"] if not p["you"])
        if not view["done"]:
            assert "hand" not in opponent, "the opponent's hand composition leaked"
            assert "dev" not in opponent, "the opponent's development cards leaked"
            assert "victoryPoints" not in opponent, "hidden victory points leaked"
            assert "tradeRates" not in opponent
        # counts are public — cards are countable at a real table
        assert "handCount" in opponent and "devCount" in opponent

        # neither deck may appear anywhere in the payload
        blob = json.dumps(view)
        assert "devDeckOrder" not in blob and "diceDeck" not in blob

        if view["done"]:
            break
        board = [i for t in view["actions"]["board"].values() for i in t.values()]
        panel = [entry["index"] for entry in view["actions"]["panel"]]
        view = game.play(rng.choice(board + panel))


def test_only_the_public_score_is_shown_while_the_game_runs():
    game = fresh_game()
    game.state.dev_cards[2][DevCard.VICTORY_POINT] = 3

    opponent = next(p for p in game.view()["players"] if not p["you"])
    assert opponent["publicVictoryPoints"] == rules.public_victory_points(game.state, 2)
    assert "victoryPoints" not in opponent


def test_the_opponents_cards_are_revealed_once_the_game_ends():
    """Hidden during play, shown at the end so the player can see what beat them."""
    game = fresh_game(seed=11)
    view = play_out(game)
    if not view["done"]:
        pytest.skip("game did not finish inside the step budget")
    opponent = next(p for p in view["players"] if not p["you"])
    assert "hand" in opponent and "dev" in opponent


def test_you_always_see_your_own_cards():
    game = fresh_game()
    you = next(p for p in game.view()["players"] if p["you"])
    assert set(you["hand"]) == {r.name.lower() for r in Resource}
    assert "victoryPoints" in you and "tradeRates" in you


# =========================================================================== #
# THE VIEW                                                                    #
# =========================================================================== #

def test_a_new_game_waits_for_the_human_to_place_a_settlement():
    view = fresh_game().view()
    assert view["you"] == api.HUMAN == 1
    assert view["yourTurn"] is True
    assert view["phase"] == Phase.SETUP_SETTLEMENT.name
    assert "settlement" in view["phaseHint"].lower()


def test_setup_offers_every_free_vertex_as_a_click_target():
    view = fresh_game().view()
    targets = view["actions"]["board"]["BUILD_SETTLEMENT"]
    assert len(targets) == T.NUM_VERTICES
    assert set(targets) == {str(v) for v in range(1, T.NUM_VERTICES + 1)}


def test_every_offered_index_is_actually_legal():
    """The client can only click what it is given, so what it is given must be right."""
    game = fresh_game(seed=3)
    rng = random.Random(3)
    view = game.view()
    for _ in range(200):
        if view["done"]:
            break
        legal = set(game.info["legal"])
        offered = [i for t in view["actions"]["board"].values() for i in t.values()]
        offered += [entry["index"] for entry in view["actions"]["panel"]]
        assert offered, f"nothing offered in {view['phase']}"
        assert set(offered) <= legal, "an illegal index was offered to the client"
        view = game.play(rng.choice(offered))


def test_board_targets_and_panel_actions_do_not_overlap():
    """A click target and a button for the same move would be two ways to do one thing."""
    game = fresh_game(seed=3)
    rng = random.Random(1)
    view = game.view()
    for _ in range(150):
        if view["done"]:
            break
        board = {i for t in view["actions"]["board"].values() for i in t.values()}
        panel = {entry["index"] for entry in view["actions"]["panel"]}
        assert not (board & panel)
        view = game.play(rng.choice(sorted(board | panel)))


def test_the_view_carries_what_a_player_needs_to_see():
    view = fresh_game().view()
    for key in ("lastRoll", "turn", "phase", "phaseHint", "bank", "devDeck",
                "robber", "board", "pieces", "players", "actions", "log",
                "victoryTarget", "handLimit", "winner", "done"):
        assert key in view, f"missing {key}"
    assert len(view["board"]["tiles"]) == T.NUM_TILES
    assert len(view["board"]["harbours"]) == 9


def test_the_dice_are_reported_every_turn():
    game = fresh_game(seed=4)
    assert game.view()["lastRoll"] is None       # nothing rolled yet
    view = play_out(game, limit=60)
    assert view["lastRoll"] is not None
    assert 2 <= view["lastRoll"] <= 12


def test_the_robber_and_tile_numbers_are_visible():
    view = fresh_game().view()
    flagged = [t for t in view["board"]["tiles"] if t["robber"]]
    assert len(flagged) == 1 and flagged[0]["id"] == view["robber"]
    desert = [t for t in view["board"]["tiles"] if t["resource"] == "desert"]
    assert len(desert) == 1 and desert[0]["number"] is None
    assert all(t["number"] is not None
               for t in view["board"]["tiles"] if t["resource"] != "desert")


def test_pieces_appear_once_they_are_built():
    game = fresh_game()
    view = game.view()
    assert view["pieces"]["buildings"] == []

    vertex = next(iter(view["actions"]["board"]["BUILD_SETTLEMENT"]))
    view = game.play(view["actions"]["board"]["BUILD_SETTLEMENT"][vertex])
    mine = [b for b in view["pieces"]["buildings"] if b["player"] == api.HUMAN]
    assert [b["vertex"] for b in mine] == [int(vertex)]
    assert mine[0]["kind"] == "settlement"


# =========================================================================== #
# THE LOG                                                                     #
# =========================================================================== #

def test_the_log_says_what_happened():
    game = fresh_game(seed=4)
    view = play_out(game, limit=120)
    assert view["log"], "nothing was logged"
    text = " ".join(view["log"])
    assert "rolled" in text
    assert "You" in text and "Opponent" in text


def test_the_log_reports_the_opponents_moves_not_just_yours():
    game = fresh_game(seed=4)
    view = play_out(game, limit=200)
    assert any(line.startswith("Opponent") for line in view["log"])


def test_production_is_narrated():
    game = fresh_game(seed=4)
    view = play_out(game, limit=400)
    assert any(" got " in line for line in view["log"]), "no payout was reported"


# =========================================================================== #
# PLAYING                                                                     #
# =========================================================================== #

def test_playing_advances_the_game_and_lets_the_opponent_reply():
    game = fresh_game()
    before = game.view()
    index = next(iter(before["actions"]["board"]["BUILD_SETTLEMENT"].values()))
    after = game.play(index)
    assert after["turn"] >= before["turn"]
    assert after["pieces"]["buildings"]


def test_the_human_is_never_asked_to_move_for_the_opponent():
    game = fresh_game(seed=6)
    rng = random.Random(6)
    view = game.view()
    for _ in range(300):
        if view["done"]:
            break
        assert view["yourTurn"] is True
        assert view["currentPlayer"] == api.HUMAN
        board = [i for t in view["actions"]["board"].values() for i in t.values()]
        panel = [entry["index"] for entry in view["actions"]["panel"]]
        view = game.play(rng.choice(board + panel))


@pytest.mark.parametrize("bad", [-1, 9999])
def test_an_out_of_range_index_is_refused(bad):
    game = fresh_game()
    with pytest.raises(ValueError):
        game.play(bad)


def test_an_illegal_index_is_refused():
    game = fresh_game()
    legal = set(game.info["legal"])
    illegal = next(i for i in range(action_space.NUM_ACTIONS) if i not in legal)
    with pytest.raises(ValueError):
        game.play(illegal)


def test_playing_after_the_game_is_over_is_refused():
    game = fresh_game(seed=11)
    view = play_out(game)
    if not view["done"]:
        pytest.skip("game did not finish inside the step budget")
    with pytest.raises(ValueError):
        game.play(0)


def test_a_seed_reproduces_the_whole_game():
    first = play_out(fresh_game(seed=21), rng=random.Random(1))
    second = play_out(fresh_game(seed=21), rng=random.Random(1))
    assert first["log"] == second["log"]
    assert first["winner"] == second["winner"]


@pytest.mark.parametrize("opponent", sorted(api.OPPONENTS))
@pytest.mark.parametrize("rules_name", sorted(api.RULESETS))
def test_every_opponent_and_ruleset_can_be_played(opponent, rules_name):
    game = fresh_game(seed=2, opponent=opponent, rules_name=rules_name)
    view = play_out(game, limit=300)
    assert view["log"]


def test_an_unknown_opponent_or_ruleset_is_refused():
    with pytest.raises(ValueError):
        api.Game(opponent="genius")
    with pytest.raises(ValueError):
        api.Game(rules_name="chess")


# =========================================================================== #
# GEOMETRY                                                                    #
# =========================================================================== #

def test_geometry_covers_the_whole_board():
    geometry = api.geometry()
    assert len(geometry["tiles"]) == T.NUM_TILES
    assert len(geometry["vertices"]) == T.NUM_VERTICES
    assert len(geometry["roads"]) == T.NUM_ROADS
    assert {t["id"] for t in geometry["tiles"]} == set(range(1, T.NUM_TILES + 1))


def test_geometry_matches_the_png_renderer():
    """The browser and the image must agree, which they do by sharing one lattice."""
    from interfaces.render import Geometry

    plan = Geometry(hex_width=110)
    geometry = api.geometry(hex_width=110)
    assert (geometry["width"], geometry["height"]) == plan.size
    for spot in geometry["vertices"]:
        assert (spot["x"], spot["y"]) == plan.vertex(spot["id"])


def test_road_geometry_names_its_endpoints():
    geometry = api.geometry()
    from interfaces.render import Geometry
    plan = Geometry(hex_width=110)
    for road in geometry["roads"]:
        first, second = (plan.vertex(v) for v in T.ROAD_VERTICES[road["id"]])
        assert (road["x1"], road["y1"]) == first
        assert (road["x2"], road["y2"]) == second


def test_geometry_is_json_serialisable():
    json.dumps(api.geometry())


# =========================================================================== #
# THE HTTP SHIM                                                               #
# =========================================================================== #

@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def request(base, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as response:
        return response.status, response.read(), response.headers


@pytest.mark.parametrize("path,content_type", [
    ("/", "text/html"),
    ("/app/app.js", "text/javascript"),
    ("/app/app.css", "text/css"),
    ("/images/tiles/wood.png", "image/png"),
])
def test_static_files_are_served(server, path, content_type):
    status, body, headers = request(server, path)
    assert status == 200
    assert headers["Content-Type"].startswith(content_type)
    assert len(body) > 0


def test_a_whole_game_can_be_played_over_http(server):
    _, body, _ = request(server, "/api/game", {"seed": 8})
    view = json.loads(body)
    game_id = view["gameId"]
    rng = random.Random(8)

    for _ in range(150):
        if view["done"]:
            break
        board = [i for t in view["actions"]["board"].values() for i in t.values()]
        panel = [entry["index"] for entry in view["actions"]["panel"]]
        _, body, _ = request(server, f"/api/game/{game_id}/action",
                             {"index": rng.choice(board + panel)})
        view = json.loads(body)

    assert view["log"]
    _, body, _ = request(server, f"/api/game/{game_id}")
    assert json.loads(body)["gameId"] == game_id


def test_the_geometry_endpoint_works(server):
    _, body, _ = request(server, "/api/geometry")
    assert len(json.loads(body)["vertices"]) == T.NUM_VERTICES


def test_an_illegal_action_is_a_400(server):
    _, body, _ = request(server, "/api/game", {"seed": 1})
    game_id = json.loads(body)["gameId"]
    with pytest.raises(urllib.error.HTTPError) as caught:
        request(server, f"/api/game/{game_id}/action", {"index": 0})
    assert caught.value.code == 400
    assert "legal" in json.loads(caught.value.read())["error"]


def test_a_missing_game_is_a_404(server):
    with pytest.raises(urllib.error.HTTPError) as caught:
        request(server, "/api/game/999999")
    assert caught.value.code == 404


def test_an_unknown_route_is_a_404(server):
    with pytest.raises(urllib.error.HTTPError) as caught:
        request(server, "/nope")
    assert caught.value.code == 404


def test_a_non_integer_action_is_a_400(server):
    _, body, _ = request(server, "/api/game", {"seed": 1})
    game_id = json.loads(body)["gameId"]
    with pytest.raises(urllib.error.HTTPError) as caught:
        request(server, f"/api/game/{game_id}/action", {"index": "one"})
    assert caught.value.code == 400


@pytest.mark.parametrize("path", [
    "/images/../../../catan/rules.py",
    "/app/../../../catan/rules.py",
    "/images/....//....//rules.py",
])
def test_path_traversal_is_refused(server, path):
    """A served path comes from the network, so `..` must be refused, not trusted."""
    with pytest.raises(urllib.error.HTTPError) as caught:
        request(server, path)
    assert caught.value.code == 404


def test_safe_path_rejects_escapes(tmp_path):
    (tmp_path / "inside.txt").write_text("ok")
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret")

    assert safe_path(tmp_path, "inside.txt") is not None
    assert safe_path(tmp_path, "../outside.txt") is None
    assert safe_path(tmp_path, "missing.txt") is None

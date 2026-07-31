"""Recording games, and the property that makes a recording worth keeping.

A record is only useful if it can be turned back into the game. The engine is
deterministic, so a seed plus the move indices reconstructs everything — every board, every
shuffled deck, every roll, and so every observation the network saw. Most of these tests
are about that reconstruction actually holding.
"""

import pathlib
import random

import pytest

from catan import action_space
from interfaces.web import api
from interfaces.web.recorder import Recorder, find, load, replay, summarise, verify


def played_game(tmp_path, seed=5, moves=2000, opponent="hard"):
    """Play a whole game through the web layer, recording it."""
    game = api.Game(opponent=opponent, seed=seed, record_game=True)
    game.recorder.path = tmp_path / "game.json"
    rng = random.Random(seed)
    for _ in range(moves):
        view = game.view()
        if view["done"]:
            break
        options = [i for t in view["actions"]["board"].values() for i in t.values()]
        options += [entry["index"] for entry in view["actions"]["panel"]]
        if not options:
            break
        game.play(rng.choice(options))
    game.recorder.save()
    return game, game.recorder.path


# =========================================================================== #
# THE POINT: IT REPLAYS                                                       #
# =========================================================================== #

def test_a_recorded_game_replays_exactly(tmp_path):
    """If this holds, the file implies every observation, roll and shuffled card in the
    game — which is what makes it usable as training data rather than as a summary."""
    game, path = played_game(tmp_path)
    if not game.info["done"]:
        pytest.skip("game did not finish in the move budget")

    assert verify(path) is True
    _, info = replay(path)
    assert info["winner"] == game.info["winner"]
    assert info["turn"] == game.info["turn"]
    assert info["scores"] == game.info["scores"]


def test_the_seed_is_recorded_even_when_the_caller_gives_none(tmp_path):
    """``reset(seed=None)`` seeds from the OS and the value is then unknowable, which would
    leave a recording that cannot be replayed — a summary, not a record."""
    game = api.Game(opponent="hard", seed=None, record_game=True)
    assert isinstance(game.seed, int)
    assert game.recorder.metadata["seed"] == game.seed

    game.recorder.path = tmp_path / "g.json"
    game.recorder.save()
    assert load(tmp_path / "g.json")["seed"] == game.seed


def test_replay_needs_nothing_but_the_file(tmp_path):
    game, path = played_game(tmp_path, seed=8)
    first = replay(path)[1]
    second = replay(path)[1]
    assert first["winner"] == second["winner"] and first["turn"] == second["turn"]


# =========================================================================== #
# WHAT IT KEEPS                                                               #
# =========================================================================== #

def test_every_decision_keeps_what_else_was_available(tmp_path):
    """The move alone does not say what was passed over, and that is exactly the question
    a lopsided game needs answered."""
    _, path = played_game(tmp_path, seed=3)
    decisions = load(path)["decisions"]
    assert decisions

    for decision in decisions:
        assert decision["chose"] in decision["legal"]
        assert decision["options"] == len(decision["legal"])
        assert action_space.describe(decision["chose"])


def test_both_sides_are_recorded(tmp_path):
    """The bot's choices are the ones being studied."""
    _, path = played_game(tmp_path, seed=3)
    decisions = load(path)["decisions"]
    assert any(d["human"] for d in decisions)
    assert any(not d["human"] for d in decisions)


def test_the_result_records_the_margin(tmp_path):
    game, path = played_game(tmp_path, seed=5)
    if not game.info["done"]:
        pytest.skip("game did not finish")
    result = load(path)["result"]
    assert result["winner"] == game.info["winner"]
    assert isinstance(result["margin"], int)
    assert result["human_won"] == (game.info["winner"] == api.HUMAN)


def test_an_unfinished_game_is_still_written(tmp_path):
    """Quitting half way leaves a record of how it was going."""
    game = api.Game(opponent="hard", seed=4, record_game=True)
    game.recorder.path = tmp_path / "part.json"
    view = game.view()
    options = [i for t in view["actions"]["board"].values() for i in t.values()]
    game.play(options[0])

    stored = load(tmp_path / "part.json")
    assert stored["result"] is None
    assert stored["moves"]
    assert verify(tmp_path / "part.json") is None      # nothing to check against yet


# =========================================================================== #
# FINDING THE INTERESTING ONES                                                #
# =========================================================================== #

def _stub(path, winner, margin, seed):
    recorder = Recorder(path=path, metadata={"seed": seed})
    recorder.moves = [0]
    recorder.result = {
        "winner": winner, "human_won": winner == 1, "turns": 50,
        "scores": {}, "margin": margin if winner == 1 else -margin,
    }
    recorder._dirty = True
    recorder.save()


def test_games_can_be_filtered_by_who_won_and_by_how_much(tmp_path):
    _stub(tmp_path / "a.json", winner=1, margin=11, seed=1)
    _stub(tmp_path / "b.json", winner=2, margin=3, seed=2)
    _stub(tmp_path / "c.json", winner=1, margin=2, seed=3)

    assert len(find(tmp_path)) == 3
    assert len(find(tmp_path, human_won=True)) == 2
    assert len(find(tmp_path, min_margin=10)) == 1, "the lopsided one"


def test_a_summary_reads_as_prose(tmp_path):
    game, path = played_game(tmp_path, seed=5)
    text = summarise(path)
    assert "opponent" in text and "decisions" in text
    if game.info["done"]:
        assert "won by" in text


def test_a_game_is_not_recorded_unless_a_person_is_playing():
    """The default. Tests, benchmarks and scripts all drive api.Game, and if they recorded
    too the handful of real games would be buried under hundreds of synthetic ones."""
    game = api.Game(opponent="hard", seed=1)
    assert game.recorder is None
    view = game.view()
    options = [i for t in view["actions"]["board"].values() for i in t.values()]
    game.play(options[0])                     # must not fall over without a recorder


# =========================================================================== #
# THE CHAMPION                                                                #
# =========================================================================== #

def test_the_interfaces_never_read_the_training_directory():
    """A run owns ``checkpoints/`` and rewrites it constantly. A game in progress must not
    be affected by a fine-tune in progress, and must never be handed a model a run later
    turned out to have made worse."""
    # Looks for a path being *built*, not for the word. Both files explain in prose why
    # they do not read that directory, and a test that forbade saying so would be a test
    # about the comments.
    import ast

    for name in ("interfaces/web/api.py", "interfaces/cli.py"):
        tree = ast.parse(pathlib.Path(name).read_text(encoding="utf-8"))
        literals = [node.value for node in ast.walk(tree)
                    if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        offenders = [text for text in literals if "checkpoints" in text and len(text) < 200]
        assert not offenders, f"{name} names the training directory in code: {offenders}"


def test_a_model_for_a_different_engine_is_declined():
    """A checkpoint from before an encoder or action-space change loads fine and then
    fails on the first move."""
    torch = pytest.importorskip("torch")
    from catan import action_space as space
    from training import champion
    from training.net import PolicyValueNet

    stale = PolicyValueNet(obs_size=64, num_actions=space.NUM_ACTIONS, hidden=(8,))
    path = pathlib.Path(__file__).parent / "_stale.pt"
    try:
        torch.save({"config": stale.config(), "weights": stale.state_dict()}, path)
        assert champion.load(path) is None
    finally:
        path.unlink(missing_ok=True)


def test_a_missing_champion_is_not_an_error():
    from training import champion

    assert champion.load("does/not/exist.pt") is None
    assert "no champion" in champion.describe() or champion.load() is not None


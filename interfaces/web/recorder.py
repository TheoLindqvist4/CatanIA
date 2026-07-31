"""Recording the games a person plays, so they can be studied and learned from.

A game you won by ten points is the most informative thing in the project: it is a worked
example of something the bot does badly, and unlike self-play it is not the bot's own
opinion of what a game looks like.

**The record is the seed and the moves, and that is enough.** The engine is deterministic —
the same seed produces the same board, the same shuffled decks and the same dice — so a
seed plus the sequence of action indices reconstructs the game *exactly*, down to every
observation the network saw. Storing observations instead would be 1,868 floats per
decision for something already implied. :func:`replay` is the proof: it rebuilds the game
and checks it lands where the recording says it did.

Alongside that, each decision keeps a small human-readable note — whose turn, the phase,
what was chosen, and **what else was on offer**. That is what makes a lost game
diagnosable: "the bot had these fourteen options here and picked that one" is the question
worth asking, and it cannot be reconstructed from the move alone without re-deriving the
legal set.

Files land in ``games/`` as JSON, one per game, written as the game goes so that quitting
half way still leaves a usable record.
"""

import datetime
import json
import pathlib

from catan import action_space
from catan.rulesets import RANKED_1V1

GAMES = pathlib.Path("games")


class Recorder:
    """Follows one game and writes it down.

    Args:
        path: where to write. Defaults to a timestamped file under ``games/``.
        metadata: anything worth knowing that is not a move — the opponent, the ruleset.
    """

    def __init__(self, path=None, metadata=None, human=1):
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = pathlib.Path(path) if path else GAMES / f"{stamp}.json"
        self.human = human
        self.metadata = dict(metadata or {})
        self.moves = []
        self.decisions = []
        self.result = None
        self._dirty = True

    # ------------------------------------------------------------------ #

    def record(self, info, index):
        """Note one decision: who, in what phase, chose what, out of what.

        ``info`` must be the one the chooser saw — *before* the action is applied.
        """
        legal = list(info["legal"])
        self.moves.append(index)
        self.decisions.append({
            "turn": info["turn"],
            "phase": info["phase"].name,
            "player": info["player"],
            "human": info["player"] == self.human,
            "chose": index,
            "action": action_space.describe(index).split(" ", 1)[-1],
            "options": len(legal),
            # The whole point: what else was available. Indices rather than labels, since
            # `action_space.describe` turns any of them back into words.
            "legal": legal,
            "scores": dict(info["public_scores"]),
            "roll": info["last_roll"],
        })
        self._dirty = True

    def finish(self, info):
        """Close the record with how it ended."""
        self.result = {
            "winner": info["winner"],
            "human_won": info["winner"] == self.human,
            "turns": info["turn"],
            "scores": dict(info["scores"]),
            "margin": _margin(info, self.human),
        }
        self._dirty = True
        self.save()

    # ------------------------------------------------------------------ #

    def save(self):
        """Write it out. Cheap enough to call after every move, and worth it — a game
        abandoned half way is still a record of how it was going."""
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "recorded_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "human": self.human,
            **self.metadata,
            "result": self.result,
            "moves": self.moves,
            "decisions": self.decisions,
        }
        tmp = self.path.with_suffix(".part")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        tmp.replace(self.path)
        self._dirty = False


def _margin(info, human):
    scores = info["scores"]
    mine = scores.get(human, 0)
    best_other = max((v for p, v in scores.items() if p != human), default=0)
    return mine - best_other


# --------------------------------------------------------------------------- #
# Reading them back                                                           #
# --------------------------------------------------------------------------- #

def load(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def replay(path):
    """Rebuild the game from the seed and the moves.

    Returns ``(env, info)`` at the final position. This is what makes the recording a
    *complete* record rather than a summary: if this reproduces the recorded result, then
    every observation, every roll and every shuffled card along the way was reproduced too,
    and the file can be turned into training data whenever it is wanted.
    """
    from catan.env import CatanEnv
    from catan.rulesets import BASE_GAME

    game = load(path)
    rules = {"ranked1v1": RANKED_1V1, "base": BASE_GAME}[game.get("rules", "ranked1v1")]
    env = CatanEnv(num_players=2, ruleset=rules)
    _, info = env.reset(seed=game.get("seed"))
    for index in game["moves"]:
        if info["done"]:
            break
        _, _, _, _, info = env.step(index)
    return env, info


def verify(path):
    """Whether replaying the file lands where the file says it did."""
    game = load(path)
    if not game.get("result"):
        return None                              # unfinished; nothing to check against
    _, info = replay(path)
    return (info["winner"] == game["result"]["winner"]
            and info["turn"] == game["result"]["turns"])


def summarise(path):
    """A readable account of one game, for working out what went wrong."""
    game = load(path)
    result = game.get("result")
    lines = [
        f"{path}",
        f"  opponent {game.get('opponent', '?')}, rules {game.get('rules', '?')}, "
        f"seed {game.get('seed')}",
    ]
    if result:
        who = "you" if result["human_won"] else "the bot"
        lines.append(f"  {who} won by {abs(result['margin'])} "
                     f"({result['scores']}) in {result['turns']} turns")
    else:
        lines.append("  unfinished")

    decisions = game["decisions"]
    bot = [d for d in decisions if not d["human"]]
    human = [d for d in decisions if d["human"]]
    lines.append(f"  {len(decisions)} decisions — you {len(human)}, the bot {len(bot)}")

    # Where the bot had the most to choose between is where it had the most to get wrong.
    widest = sorted(bot, key=lambda d: -d["options"])[:5]
    if widest:
        lines.append("  the bot's widest choices:")
        for d in widest:
            lines.append(f"    turn {d['turn']:>3} {d['phase']:<18} "
                         f"chose {d['action']:<28} out of {d['options']}")
    return "\n".join(lines)


def find(directory=GAMES, human_won=None, min_margin=None):
    """Recorded games, newest first, optionally filtered.

    ``min_margin`` is the one worth reaching for: a game lost by ten points says more about
    what the bot cannot do than fifty close ones.
    """
    directory = pathlib.Path(directory)
    if not directory.is_dir():
        return []
    out = []
    for path in sorted(directory.glob("*.json"), reverse=True):
        try:
            game = load(path)
        except (OSError, json.JSONDecodeError):
            continue
        result = game.get("result")
        if result is None:
            continue
        if human_won is not None and result["human_won"] != human_won:
            continue
        if min_margin is not None and abs(result["margin"]) < min_margin:
            continue
        out.append((path, game))
    return out


# --------------------------------------------------------------------------- #

def main(argv=None):
    """python -m interfaces.web.recorder — look through what has been recorded."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Review recorded games",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("path", nargs="?", help="one game; omit to list them")
    parser.add_argument("--won", action="store_true", help="only games you won")
    parser.add_argument("--lost", action="store_true", help="only games you lost")
    parser.add_argument("--margin", type=int,
                        help="only games decided by at least this many points — the "
                             "lopsided ones say most about what the bot cannot do")
    parser.add_argument("--verify", action="store_true",
                        help="replay each game and check it lands where it says")
    arguments = parser.parse_args(argv)

    if arguments.path:
        print(summarise(arguments.path))
        if arguments.verify:
            print(f"  replays exactly: {verify(arguments.path)}")
        return 0

    won = True if arguments.won else (False if arguments.lost else None)
    found = find(human_won=won, min_margin=arguments.margin)
    if not found:
        print("no recorded games match")
        return 0
    for path, game in found:
        result = game["result"]
        who = "you" if result["human_won"] else "bot"
        line = (f"{path.name}  {who} +{abs(result['margin']):<3} "
                f"{result['turns']:>4} turns  vs {game.get('opponent', '?')}")
        if arguments.verify:
            line += f"  replay {'ok' if verify(path) else 'MISMATCH'}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

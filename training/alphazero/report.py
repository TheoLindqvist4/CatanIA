"""What a run did, read back from its own record.

    python -m training.alphazero.report
    python -m training.alphazero.report --run checkpoints/alphazero --full

Section 18 of the design guide asks for a metrics dashboard: simulation throughput, the
training losses, the evaluation win rate, and the replay fill level. :class:`Trainer` writes
all of it to ``metrics.jsonl``, one JSON object per iteration, flushed every time — this is
the reader.

A file rather than a live dashboard for the reason ``CLAUDE.md`` records: ``python -m``
through a pipe buffers, so a long run's stdout arrives in lumps or not at all, and piping it
through ``tail`` is worse still because ``tail`` prints nothing until the process exits. The
file is the honest record, it can be read while the run is going, and it survives the run.

**Read the win-rate column with the interval, not without it.** 150 games is +-8 points at
50%, so two consecutive evaluations differing by 5 points have not shown anything. The column
is printed with its Wilson bounds for exactly that reason.
"""

import argparse
import json
import pathlib


def load(directory="checkpoints/alphazero"):
    """Every iteration record, oldest first. Tolerates a half-written last line."""
    path = pathlib.Path(directory) / "metrics.jsonl"
    if not path.is_file():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue                      # the trainer may be mid-write
    return entries


def evaluations(entries):
    """``[(iteration, evaluation)]`` for the iterations that ran one."""
    return [(e["iteration"], e["evaluation"]) for e in entries if e.get("evaluation")]


def summarise(entries):
    """The headline numbers for a whole run."""
    if not entries:
        return {}
    trained = [e for e in entries if "policy_loss" in e]
    generated = sum(e.get("generate_seconds", 0) for e in entries)
    checks = evaluations(entries)
    return {
        "iterations": len(entries),
        "games": entries[-1].get("total_games", 0),
        "minutes": entries[-1].get("elapsed_minutes", 0),
        "generation_share": round(generated / max(1e-9, 60 * entries[-1]["elapsed_minutes"]), 2),
        "positions_per_second": round(
            sum(e.get("positions_per_second", 0) for e in entries[-10:])
            / max(1, len(entries[-10:])), 1),
        "buffer": entries[-1].get("buffer", 0),
        "policy_loss": trained[-1]["policy_loss"] if trained else None,
        "value_loss": trained[-1]["value_loss"] if trained else None,
        "entropy": trained[-1]["entropy"] if trained else None,
        "first_evaluation": checks[0][1]["win_rate"] if checks else None,
        "best_evaluation": max((c[1]["win_rate"] for c in checks), default=None),
        "last_evaluation": checks[-1][1]["win_rate"] if checks else None,
    }


def report(directory="checkpoints/alphazero", full=False, log=print):
    entries = load(directory)
    if not entries:
        log(f"no metrics in {directory}/ — has a run started?")
        return {}

    summary = summarise(entries)
    log(f"run: {directory}")
    log(f"  {summary['iterations']} iterations, {summary['games']:,} games, "
        f"{summary['minutes']:.0f} minutes")
    log(f"  {summary['positions_per_second']:.0f} positions/sec (last 10 iterations), "
        f"{100 * summary['generation_share']:.0f}% of the clock spent generating")
    log(f"  buffer {summary['buffer']:,};  policy {summary['policy_loss']}, "
        f"value {summary['value_loss']}, entropy {summary['entropy']}")

    log("\nagainst the fixed heuristic (the raw policy, no search):")
    for iteration, check in evaluations(entries):
        low, high = check["ci"]
        bar = "#" * int(round(40 * check["win_rate"]))
        log(f"  iter {iteration:>4}  {100 * check['win_rate']:5.1f}%  "
            f"[{100 * low:4.1f}, {100 * high:4.1f}]  trunc {check['truncated']:>3}  {bar}")

    if full:
        log("\nper iteration:")
        for e in entries:
            log(f"  {e['iteration']:>4}  games {e.get('games', 0):>4}  "
                f"buffer {e.get('buffer', 0):>7,}  "
                f"gen {e.get('generate_seconds', 0):>5.1f}s  "
                f"train {e.get('train_seconds', 0):>5.1f}s  "
                f"pi {e.get('policy_loss', '-')}  v {e.get('value_loss', '-')}  "
                f"turns {e.get('turns', '-')}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="What a run did",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run", default="checkpoints/alphazero")
    parser.add_argument("--full", action="store_true", help="every iteration, not a summary")
    arguments = parser.parse_args(argv)
    report(arguments.run, full=arguments.full)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Start an AlphaZero run.

    python -u -m training.alphazero.train --hours 3
    python -u -m training.alphazero.train --iterations 2 --workers 2   # a smoke test

``-u`` matters. ``python -m`` through a pipe buffers its output, so a long run appears to
produce nothing for minutes at a time; ``CLAUDE.md`` records this costing time already. The
authoritative progress record is ``checkpoints/alphazero/metrics.jsonl`` either way.

Nothing this script writes goes near ``models/``. A run owns ``checkpoints/alphazero/`` and
may ruin it; the champion changes only through ``python -m training.alphazero.champion
promote``, which plays a match first. That separation is what lets somebody play the game in
one window while this runs in another.
"""

import argparse
import multiprocessing
import time

import torch

from training.alphazero.config import load_config
from training.alphazero.network import load_for_alphazero, new_network
from training.alphazero.trainer import Trainer


def build_network(config, log=print):
    """The network the run starts from, warm-started when a source is configured.

    A cold start is supported and is what "learns entirely from self-play" means, but it is
    not the default here: see ``docs/decisions/0023-alphazero-self-play.md`` for why, and for
    what is given up by warm-starting.
    """
    source = config["warm_start"]
    if not source:
        log("cold start: a fresh network")
        return new_network(), {"warm_start": None}
    try:
        net, notes = load_for_alphazero(source)
    except Exception as error:
        log(f"could not warm-start from {source}: {error}")
        log("falling back to a fresh network")
        return new_network(), {"warm_start": None, "warm_start_error": str(error)}
    log(f"warm-started from {source}")
    if notes["grafted_columns"]:
        log(f"  grafted {notes['grafted_columns']} zero columns onto the context layer "
            f"(checkpoint was trained at obs_size={notes['trained_at_obs_size']})")
    if notes["reset"]:
        log(f"  re-initialised: {', '.join(notes['reset'])}")
    return net, notes


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="AlphaZero self-play for 1v1 Catan",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--hours", type=float, default=None,
                        help="wall-clock budget; the run stops cleanly and checkpoints")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None, dest="self_play_workers")
    parser.add_argument("--simulations", type=int, default=None, dest="mcts_simulations")
    parser.add_argument("--generate-seconds", type=int, default=None,
                        dest="generate_seconds",
                        help="self-play wall-clock per iteration")
    parser.add_argument("--positions", type=int, default=None,
                        dest="positions_per_iteration",
                        help="stop generation on a sample count instead of a clock; idles "
                             "most of the pool, so this is for benchmarks")
    parser.add_argument("--batches", type=int, default=None, dest="training_batches")
    parser.add_argument("--lr", type=float, default=None, dest="learning_rate")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cold", action="store_true",
                        help="ignore warm_start and begin from a fresh network")
    parser.add_argument("--run-directory", default=None, dest="run_directory")
    arguments = parser.parse_args(argv)

    overrides = {
        name: getattr(arguments, name)
        for name in ("self_play_workers", "mcts_simulations", "positions_per_iteration",
                     "generate_seconds", "training_batches", "learning_rate", "seed",
                     "run_directory")
    }
    if arguments.cold:
        overrides["warm_start"] = ""
    config = load_config(arguments.config, **overrides)

    # The parent keeps a wide thread pool for the gradient step; the workers are pinned to
    # one each by ParallelSelfPlay before the pool is created.
    torch.set_num_threads(max(2, multiprocessing.cpu_count() // 4))

    net, notes = build_network(config)
    print(f"network: {net!r}")
    print(f"settings that differ from the defaults: {config.describe()}")
    print(f"workers {config['self_play_workers']}, "
          f"{config['envs_per_worker']} games each, "
          f"{config['mcts_simulations']} simulations/move")

    trainer = Trainer(net, config)
    trainer.history.append({"warm_start": notes})
    trainer.history.clear()

    seconds = None if arguments.hours is None else arguments.hours * 3600
    started = time.perf_counter()
    trainer.run(seconds=seconds, iterations=arguments.iterations)
    elapsed = (time.perf_counter() - started) / 60

    print(f"\nran {trainer.iteration} iterations, {trainer.games:,} games, "
          f"{trainer.positions:,} positions in {elapsed:.1f} minutes")
    print(f"checkpoints in {trainer.directory}/ — nothing was written to models/")
    print("promote with:  python -m training.alphazero.champion promote "
          f"{trainer.directory}/best.pt")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())

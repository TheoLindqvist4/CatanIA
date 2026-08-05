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

    ⚠️ **A warm start builds the checkpoint's shape, not the configured one.**
    :func:`~training.alphazero.network.load_for_alphazero` reads ``checkpoint["config"]``,
    which is the whole point when the *observation* has changed — ``graft`` widens the
    affected layers and every weight keeps its meaning. But ``graft`` cannot help when
    ``width``, ``depth`` or ``trunk`` change: then every tensor has a different shape and
    there is no column-wise correspondence to preserve.

    So changing :func:`~training.alphazero.network.new_network`'s defaults while
    ``warm_start`` points at an older checkpoint does **nothing**, silently. That is not
    hypothetical — it is the state this function was in when record 0026 changed the defaults
    to 374,331 parameters: a run started then trained the reigning 200,379-parameter shape,
    with ``aux=False``, so the auxiliary targets self-play was computing were dropped on the
    floor by the loss. Nothing raised, and the metrics looked healthy.

    It is now reported. The mismatch is not an error — carrying a trained policy forward is
    usually worth more than the shape you asked for — but it has to be a decision somebody
    took, so it is printed next to what to do about it.
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

    difference = shape_difference(net)
    if difference:
        notes["shape_mismatch"] = difference
        log("")
        # ASCII on purpose: this prints to a Windows console, whose default cp1252 codec
        # raises UnicodeEncodeError on the warning sign this file uses freely in docstrings.
        # A startup banner that crashes the run it is warning about is worse than no banner.
        log("  !! this run will train the CHECKPOINT's shape, not the configured one:")
        for key, (was, wanted) in sorted(difference.items()):
            log(f"        {key:<16} {was!r:>8}  configured {wanted!r}")
        log(f"      {net.num_parameters():,} parameters, not "
            f"{new_network().num_parameters():,}. graft carries an *observation* change; it "
            f"cannot carry a width, depth or trunk change.")
        log(f"      To actually train the configured shape, either:")
        log(f"        python -m training.alphazero.distil --source {source} "
            f"--out checkpoints/distilled.pt   (then warm_start from that)")
        log(f"        python -u -m training.alphazero.train --cold"
            f"                       (throws the policy away)")
        log("")
    return net, notes


def shape_difference(net):
    """``{key: (checkpoint's value, configured value)}`` for every geometry key that differs.

    Compared against a freshly built network rather than against a written-down table, so
    this cannot go stale when :func:`~training.alphazero.network.new_network`'s signature
    changes — which is exactly the event it exists to notice.
    """
    wanted = new_network().config()
    have = net.config()
    return {
        key: (have.get(key), wanted[key])
        for key in ("width", "road_width", "context", "hops", "depth", "rounds", "trunk",
                    "aux")
        if key in wanted and have.get(key) != wanted[key]
    }


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
    parser.add_argument("--warm-start", default=None, dest="warm_start",
                        help="checkpoint to continue from, overriding the config. The usual "
                             "value is models/champion_az.pt: a follow-on run should start "
                             "from the champion, not from whatever the previous run's "
                             "config happened to name")
    parser.add_argument("--run-directory", default=None, dest="run_directory")
    arguments = parser.parse_args(argv)

    overrides = {
        name: getattr(arguments, name)
        for name in ("self_play_workers", "mcts_simulations", "positions_per_iteration",
                     "generate_seconds", "training_batches", "learning_rate", "seed",
                     "run_directory", "warm_start")
    }
    if arguments.cold:
        if arguments.warm_start:
            parser.error("--cold and --warm-start contradict each other")
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
    # Which candidate to offer the gate depends on whether the in-loop check ran. With it
    # off there is no `best.pt` and, more to the point, no opinion about which snapshot is
    # best — which is the honest state, because that opinion was the raw policy's and
    # CLAUDE.md records it moving opposite to search-ranked strength. Rank first, promote
    # second.
    snapshots = sorted((trainer.directory / "snapshots").glob("iter_*.pt"))
    if snapshots:
        print(f"\n{len(snapshots)} snapshots in {trainer.directory}/snapshots/. Rank them "
              f"*with search* before promoting anything:")
        print(f"  python -m training.alphazero.arena {trainer.directory}/snapshots/*.pt "
              f"{trainer.directory}/latest.pt")
        print("  python -m training.alphazero.champion promote <the winner>")
    else:
        print("promote with:  python -m training.alphazero.champion promote "
              f"{trainer.directory}/best.pt")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())

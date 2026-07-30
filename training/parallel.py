"""Collecting rollouts across processes.

The engine is pure Python, so a rollout is bound by the GIL and one process uses one core of
twenty. Measured on this machine: 3,327 environment steps/sec on one worker against 22,377
across sixteen. That is the cheapest remaining improvement in the training loop and it needs
no change to the algorithm at all.

Three things make this less mechanical than it looks.

**Windows spawns rather than forks.** Every child re-imports this module, so the worker
function must be module-level and everything crossing the boundary must be picklable
(verified: ``CatanEnv`` 8.7 KB, ``GameState`` 8.7 KB, ``Board`` 2.8 KB). Nothing is inherited
— each worker builds its own environments and its own copy of the network.

**Thread oversubscription is the classic way to make this slower.** Torch defaults to one
OpenMP thread per core — measured ``get_num_threads() == 14`` here — so eight workers would
ask for 112 threads on 20 cores and spend their time in the scheduler. Workers are pinned to
one thread each. The environment variables must be set *before* the child imports torch, and
a child inherits ``os.environ`` at spawn time, so :func:`collector` sets them in the parent
just before creating the pool. The parent has already initialised its own pool by then, so its
own thread count is unaffected — it keeps the wide setting for the PPO update, which is a
genuinely parallel matmul.

**Each worker keeps its own opponent pool.** The alternative is broadcasting the pool with
every task, which at ten frozen networks is 54 MB per worker per iteration. Instead every
worker snapshots the weights it receives on the same fixed schedule — and since they all
receive the same weights, their pools are identical in content without a byte crossing the
boundary. Only the learner's 5.4 MB is sent per iteration.

The collectors are **persistent inside each worker**, for the same reason as in the
single-process case: a rollout that discarded its in-flight games would waste most of them.
"""

import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from catan import action_space, encoder
from training.net import PolicyValueNet, build
from training.rollout import TRAINING_MAX_TURNS, Rollout, SelfPlayCollector

#: Per-worker state. Rebuilt once per process, then reused for the whole run.
_WORKER = {}


def _configure(settings):
    """Runs once in each child. Builds the network, the pool mirror and the collector."""
    torch.set_num_threads(1)
    torch.manual_seed(settings["seed"])

    net = build(settings["config"])
    net.eval()

    state = {
        "net": net,
        "settings": settings,
        "frozen": [],                       # [(iteration, state_dict)] — this worker's mirror
        "rng": np.random.default_rng(settings["seed"]),
        "iteration": 0,
    }

    from catan.agents import HeuristicAgent

    def sample_opponent():
        """Same policy as OpponentPool.sample, decided locally so nothing is transferred."""
        draw = float(state["rng"].random())
        if draw < settings["self_play"] or not state["frozen"]:
            return None, "self"
        if draw < settings["self_play"] + settings["heuristic"]:
            return HeuristicAgent(int(state["rng"].integers(1 << 30))), "heuristic"
        weights = np.arange(1, len(state["frozen"]) + 1, dtype=float)
        index = int(state["rng"].choice(len(state["frozen"]), p=weights / weights.sum()))
        iteration, stored = state["frozen"][index]
        frozen = build(settings["config"])
        frozen.load_state_dict(stored)
        frozen.eval()
        return frozen, f"frozen@{iteration}"

    state["collector"] = SelfPlayCollector(
        net,
        num_envs=settings["envs"],
        opponent=sample_opponent,
        gamma=settings["gamma"],
        lam=settings["lam"],
        shaping=settings["shaping"],
        max_turns=settings["max_turns"],
        seed=settings["seed"],
    )
    _WORKER.clear()
    _WORKER.update(state)


def _collect(payload):
    """One worker's share of an iteration. Returns pickled arrays, not tensors."""
    weights, steps, iteration, snapshot = pickle.loads(payload)
    state = _WORKER

    state["net"].load_state_dict(weights)
    state["iteration"] = iteration

    if snapshot:
        # mirror the master's pool schedule, from weights this worker already has
        state["frozen"].append(
            (iteration, {k: v.detach().clone() for k, v in state["net"].state_dict().items()})
        )
        if len(state["frozen"]) > state["settings"]["pool_size"]:
            state["frozen"].pop(len(state["frozen"]) // 2)

    rollout = state["collector"].collect(steps)
    return pickle.dumps((
        rollout.obs.numpy(), rollout.mask.numpy(), rollout.action.numpy(),
        rollout.logp.numpy(), rollout.value.numpy(),
        rollout.advantage.numpy(), rollout.returns.numpy(),
        rollout.stats,
    ), protocol=pickle.HIGHEST_PROTOCOL)


class ParallelCollector:
    """Fans a rollout out over ``workers`` processes and stitches the results back.

    The interface matches :class:`~training.rollout.SelfPlayCollector` — ``collect(n)``
    returns a :class:`~training.rollout.Rollout` — so the trainer does not care which one it
    has.

    Args:
        net: the policy being trained. Only its weights cross the process boundary.
        workers: processes. Sensible default is ``cores - 4``, leaving room for the parent's
            PPO update.
        envs: environments *per worker*.
    """

    def __init__(self, net, workers=8, envs=32, gamma=1.0, lam=0.95, shaping=0.3,
                 max_turns=TRAINING_MAX_TURNS, self_play=0.6, heuristic=0.15,
                 pool_size=10, seed=0):
        self.net = net
        self.workers = workers
        self.pool_size = pool_size

        settings = {
            "config": net.config(),
            "envs": envs,
            "gamma": gamma, "lam": lam, "shaping": shaping,
            "max_turns": max_turns,
            "self_play": self_play, "heuristic": heuristic,
            "pool_size": pool_size,
            "seed": seed,
        }

        # Children inherit os.environ at spawn. The parent has already initialised its own
        # OpenMP pool, so setting these now pins the workers without narrowing the parent.
        previous = {}
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            previous[name] = os.environ.get(name)
            os.environ[name] = "1"

        self.pool = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_configure,
            initargs=({**settings, "seed": seed},),
        )
        # Give each worker a distinct stream by re-seeding on first use
        self._seeds = [seed * 104_729 + i for i in range(workers)]
        self._started = False

        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    # ------------------------------------------------------------------ #

    def collect(self, num_steps, iteration=0, snapshot=False):
        """Gather at least ``num_steps`` transitions, split across the workers."""
        share = max(256, num_steps // self.workers)
        weights = {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
        payload = pickle.dumps((weights, share, iteration, snapshot),
                               protocol=pickle.HIGHEST_PROTOCOL)

        results = list(self.pool.map(_collect, [payload] * self.workers))
        return _stitch([pickle.loads(blob) for blob in results])

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _stitch(parts):
    """One :class:`Rollout` from the workers' pieces, with the statistics summed."""
    stats = {"games": 0, "wins": 0, "losses": 0, "truncated": 0, "opponents": {},
             "lengths": [], "turns": [], "forced": 0, "decisions": 0}
    for part in parts:
        piece = part[-1]
        for key in ("games", "wins", "losses", "truncated", "forced", "decisions"):
            stats[key] += piece[key]
        for key in ("lengths", "turns"):
            stats[key].extend(piece[key])
        for label, count in piece["opponents"].items():
            stats["opponents"][label] = stats["opponents"].get(label, 0) + count

    def join(index, dtype=None):
        arrays = [part[index] for part in parts]
        joined = np.concatenate(arrays, axis=0)
        return torch.as_tensor(joined if dtype is None else joined.astype(dtype))

    return Rollout(
        obs=join(0), mask=join(1), action=join(2), logp=join(3),
        value=join(4), advantage=join(5), ret=join(6), stats=stats,
    )


def benchmark(workers=(1, 2, 4, 8, 12), envs=32, steps=4_000, seed=0):
    """Throughput against worker count. Run it before believing any speed-up."""
    import time

    net = PolicyValueNet(encoder.SIZE, action_space.NUM_ACTIONS)
    results = {}
    for count in workers:
        collector = ParallelCollector(net, workers=count, envs=envs, seed=seed)
        try:
            collector.collect(steps)                       # fill the pipelines
            started = time.perf_counter()
            total = sum(len(collector.collect(steps)) for _ in range(3))
            elapsed = time.perf_counter() - started
        finally:
            collector.close()
        results[count] = total / elapsed
        print(f"  {count:>3} workers  {total:>7,} transitions / {elapsed:5.1f}s "
              f"= {results[count]:>7,.0f}/sec", flush=True)
    return results


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    print(f"cores: {multiprocessing.cpu_count()}, torch threads: {torch.get_num_threads()}")
    benchmark()

"""The continuous loop: generate, learn, measure, repeat.

The guide is emphatic that this must *not* be "collect millions of games, then train". It is
one loop, and the two halves overlap in wall-clock because self-play runs in worker processes
while the gradient step runs in the parent.

    generate positions_per_iteration samples across the pool
    push them into the replay buffer
    take training_batches gradient steps
    every evaluate_every iterations, play the candidate against the ladder
    checkpoint on a timer

**The loss.** AlphaZero's, unchanged:

    L  =  -sum(pi * log p)  +  value_weight * (z - v)^2  +  weight_decay * ||theta||^2

with the policy cross-entropy taken over *legal* actions only. Masking during the update is
not an optimisation: the mask is applied at play time, so an unmasked update trains a
distribution the agent never samples from, and the two drift.

**The value target is the game result**, ``+1``/``-1`` from the mover's own point of view, or
0 for a game that hit the turn cap. Not the search's root value, and not a bootstrap: the
engine's three silent invariants recorded in ``CLAUDE.md`` — the winner is always the player
who just acted, the terminal observation is the winner's view, the terminal mask is all
zero — make every bootstrapped variant of this a sign error waiting to happen, and the game
is short enough that the plain outcome is a fine target.

**Progress is written to a file, not to stdout.** ``python -m`` through a pipe buffers, so a
long run's output arrives in 8 KB lumps or not at all. ``metrics.jsonl`` in the run directory
is the honest record, one JSON object per iteration, flushed every time.
"""

import json
import pathlib
import time

import numpy as np
import torch
from torch import nn

from training.alphazero.replay_buffer import ReplayBuffer
from training.alphazero.workers import ParallelSelfPlay


class Trainer:
    """One AlphaZero run.

    Args:
        net: the network to train. Already warm-started, if it is going to be.
        config: a :class:`~training.alphazero.config.Config`.
        log: where the human-readable line per iteration goes.
    """

    def __init__(self, net, config, log=print):
        self.net = net
        self.config = config
        self.log = log
        self.directory = pathlib.Path(config["run_directory"])
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.directory / "metrics.jsonl"

        self.buffer = ReplayBuffer(capacity=config["replay_buffer_size"])
        self.optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        self.rng = np.random.default_rng(config["seed"])
        self.iteration = 0
        self.games = 0
        self.positions = 0
        self.started = time.perf_counter()
        self.last_checkpoint = self.started
        self.best_win_rate = -1.0
        self.history = []

    # ------------------------------------------------------------------ #
    # The loop                                                            #
    # ------------------------------------------------------------------ #

    def run(self, seconds=None, iterations=None):
        """Train until a wall-clock budget or an iteration count is reached.

        A budget rather than an epoch count because the thing actually being spent is time:
        an iteration's length depends on how long the games in flight take to finish, which
        depends on the policy, which is what is changing.
        """
        deadline = None if seconds is None else self.started + seconds
        pool = ParallelSelfPlay(self.net, self.config)
        try:
            while True:
                if deadline is not None and time.perf_counter() >= deadline:
                    self.log(f"budget spent after {self.iteration} iterations")
                    break
                if iterations is not None and self.iteration >= iterations:
                    break
                self.iterate(pool, deadline=deadline)
        except KeyboardInterrupt:
            self.log("interrupted — writing a checkpoint before stopping")
        finally:
            self.checkpoint("latest")
            pool.close()
        return self.history

    def iterate(self, pool, deadline=None):
        """One turn of the loop."""
        self.iteration += 1
        started = time.perf_counter()

        arrays, results = pool.generate(
            positions=self.config["positions_per_iteration"] or None,
            seconds=None if self.config["positions_per_iteration"] else
                    self.config["generate_seconds"],
        )
        added = self.buffer.add(*arrays)
        generated = time.perf_counter()

        self.games += len(results)
        self.positions += added
        finished = [r for r in results if r["winner"] is not None]
        entry = {
            "iteration": self.iteration,
            "games": len(results),
            "total_games": self.games,
            "positions": added,
            "buffer": len(self.buffer),
            "generate_seconds": round(generated - started, 2),
            "turns": round(float(np.mean([r["turns"] for r in results])), 1) if results else None,
            "searched_per_game": (
                round(float(np.mean([r["searched"] for r in results])), 1) if results else None
            ),
            "truncated": len(results) - len(finished),
        }

        if len(self.buffer) >= self.config["min_buffer"]:
            entry.update(self.learn(self.config["training_batches"]))
        else:
            entry["skipped"] = f"buffer below min_buffer ({self.config['min_buffer']})"

        entry["train_seconds"] = round(time.perf_counter() - generated, 2)
        entry["elapsed_minutes"] = round((time.perf_counter() - self.started) / 60, 2)
        entry["games_per_second"] = (
            round(len(results) / max(1e-9, generated - started), 2) if results else 0.0
        )
        entry["positions_per_second"] = round(added / max(1e-9, generated - started), 1)

        if self.iteration % self.config["evaluate_every"] == 0:
            # Skipped when the budget is nearly gone: an evaluation that does not finish
            # before the deadline is time taken from generation for nothing.
            room = deadline is None or (deadline - time.perf_counter()) > 180
            if room:
                entry["evaluation"] = self.evaluate()

        self._record(entry)
        if (time.perf_counter() - self.last_checkpoint
                > 60 * self.config["checkpoint_interval_minutes"]):
            self.checkpoint("latest")
            self.last_checkpoint = time.perf_counter()
        return entry

    # ------------------------------------------------------------------ #
    # Learning                                                            #
    # ------------------------------------------------------------------ #

    def learn(self, batches):
        """``batches`` gradient steps on samples from the replay buffer."""
        self.net.train()
        policy_total = value_total = norm_total = entropy_total = 0.0
        size = self.config["batch_size"]

        for _ in range(batches):
            obs, target, mask, outcome = self.buffer.sample(size, self.rng)
            obs = torch.from_numpy(obs)
            target = torch.from_numpy(target)
            mask = torch.from_numpy(mask)
            outcome = torch.from_numpy(outcome)

            logits, value = self.net(obs)
            logits = self.net._apply_mask(logits, mask)
            log_probabilities = torch.log_softmax(logits, dim=-1)

            # Cross-entropy against the visit distribution. Rows whose target is all zero —
            # which cannot happen from a completed search, but can from a truncated one —
            # contribute nothing rather than a NaN.
            policy_loss = -(target * log_probabilities).sum(dim=-1).mean()
            value_loss = nn.functional.mse_loss(value, outcome)
            loss = policy_loss + self.config["value_weight"] * value_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.config["grad_clip"])
            self.optimizer.step()

            with torch.no_grad():
                probabilities = log_probabilities.exp()
                entropy = -(probabilities * log_probabilities).nan_to_num().sum(-1).mean()
                policy_total += float(policy_loss.detach())
                value_total += float(value_loss.detach())
            norm_total += float(norm)
            entropy_total += float(entropy)

        self.net.eval()
        return {
            "batches": batches,
            "policy_loss": round(policy_total / batches, 4),
            "value_loss": round(value_total / batches, 4),
            "entropy": round(entropy_total / batches, 4),
            "grad_norm": round(norm_total / batches, 3),
            "learning_rate": self.config["learning_rate"],
        }

    # ------------------------------------------------------------------ #
    # Measurement and checkpoints                                         #
    # ------------------------------------------------------------------ #

    def evaluate(self):
        """Play the current network against the fixed heuristic.

        Only the heuristic, and only ``evaluation_games`` of them: this runs *inside* the
        training loop, so it is a smoke alarm, not the promotion gate. The gate is
        :func:`training.alphazero.champion.promote`, and it plays a longer match against more
        opponents once the run is over.
        """
        from catan.agents import HeuristicAgent
        from training.alphazero.agent import MCTSAgent
        from training.alphazero.evaluator import evaluate_agent, format_result

        agent = MCTSAgent(self.net, simulations=self.config["eval_simulations"],
                          temperature=0.0, seed=self.iteration)
        result = evaluate_agent(agent, HeuristicAgent(0),
                                games=self.config["evaluation_games"],
                                seed=10_000 + self.iteration)
        self.log("  " + format_result("heuristic", result))

        # "Best so far" is best *within this run* and nothing more — it is not a champion and
        # is never copied into models/ by hand. It exists so a run that ends on a bad
        # iteration still has its good one on disk to offer the promotion gate.
        #
        # And it is best by the *policy-only* score, because that is what this loop can afford
        # to measure. Search is worth about ten points on top and does not have to be worth
        # the same ten points for every checkpoint, so "best policy" is not necessarily "best
        # player". Re-measure the candidates with search before promoting — see
        # `training/alphazero/arena.py`, and D17 in the decision record.
        if result["win_rate"] >= self.best_win_rate:
            self.best_win_rate = result["win_rate"]
            self.checkpoint("best")
        return {
            "win_rate": result["win_rate"],
            "ci": list(result["ci"]),
            "truncated": result["truncated"],
            "games": result["games"],
            "simulations": self.config["eval_simulations"],
        }

    def checkpoint(self, name):
        """Write the network under ``run_directory``. Never under ``models/``."""
        path = self.directory / f"{name}.pt"
        staging = path.with_suffix(".incoming")
        torch.save({
            "config": self.net.config(),
            "weights": {k: v.detach().cpu() for k, v in self.net.state_dict().items()},
            "optimizer": self.optimizer.state_dict(),
            "iteration": self.iteration,
            "games": self.games,
            "positions": self.positions,
            "settings": dict(self.config),
            "lineage": "alphazero",
        }, staging)
        staging.replace(path)
        return path

    def _record(self, entry):
        self.history.append(entry)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        parts = [
            f"iter {entry['iteration']:>3}",
            f"{entry['total_games']:>6,} games",
            f"buffer {entry['buffer']:>7,}",
            f"gen {entry['generate_seconds']:>5.1f}s",
            f"train {entry['train_seconds']:>5.1f}s",
        ]
        if "policy_loss" in entry:
            parts += [f"pi {entry['policy_loss']:.3f}", f"v {entry['value_loss']:.3f}",
                      f"H {entry['entropy']:.2f}"]
        if entry.get("turns"):
            parts.append(f"turns {entry['turns']:.0f}")
        if entry.get("evaluation"):
            parts.append(f"vs heuristic {100 * entry['evaluation']['win_rate']:.1f}%")
        parts.append(f"[{entry['elapsed_minutes']:.1f} min]")
        self.log("  ".join(parts))

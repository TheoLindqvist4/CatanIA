"""Collecting self-play experience.

Three things about this environment make the usual PPO rollout loop wrong, and each is
handled here rather than left to the trainer:

**The loser never receives its reward.** ``CatanEnv.step`` attributes reward to the player
who *acted*, and the terminal step is by definition the winner's. So transitions are
accumulated into a **per-seat trajectory** and the outcome is written onto both when the game
ends. Reading ``reward`` out of ``step`` and treating it as the signal would train the agent
on the winner's moves alone.

**The turn order is not alternating.** During a discard the decision belongs to whoever is
over the hand limit. ``info["player"]`` is the authority; assuming alternation is the classic
multi-agent environment bug and here it would silently mislabel whole trajectories.

**12.3% of decisions have exactly one legal action.** They carry no policy gradient — the
ratio is 1 by construction — so they are played without querying the network and never enter
the buffer. That is measured, not assumed (see ``docs/decisions/0017``). With ``gamma = 1.0``
skipping them is exact rather than approximate, which is one reason for that default.

Environments are stepped **in lockstep, many per process**, so one batched forward pass
serves all of them. The engine is pure Python at ~3,700 steps/sec, so a batch-of-one forward
pass per step would spend most of its time in PyTorch dispatch overhead.
"""

import numpy as np
import torch

from catan import action_space
from catan.env import CatanEnv
from catan.rulesets import RANKED_1V1

#: Games are cut off far sooner than the engine's 5,000-turn default. An untrained policy
#: plays close to randomly, and random play needs a median of 435 turns to reach 15 points —
#: so without this the first iterations are almost entirely spent on games that teach
#: nothing. Truncated games are adjudicated on victory points rather than thrown away.
TRAINING_MAX_TURNS = 400

#: What an unfinished game is worth, as a fraction of a real win. Scored by the victory-point
#: difference, the same way an adjourned chess game is adjudicated on material: a position
#: three points ahead is not a win, but it is not nothing either, and it is the only gradient
#: available early on when almost every game is truncated.
TRUNCATION_WEIGHT = 0.5


def callable_factory(opponent):
    """Whether ``opponent`` is a *factory* to be asked for a new opponent each game.

    A factory is a zero-argument callable; an agent is a two-argument one. Told apart by
    signature rather than by a flag, so a caller cannot pass one and mean the other.
    """
    if opponent is None or isinstance(opponent, torch.nn.Module):
        return False
    if not callable(opponent):
        return False
    import inspect
    try:
        return len(inspect.signature(opponent).parameters) == 0
    except (TypeError, ValueError):
        return False


class SeatTrajectory:
    """One player's decisions in one game, in order.

    Kept separately per seat because a seat's transitions are contiguous *for that seat*
    even though they are interleaved with the opponent's in the environment.
    """

    __slots__ = ("obs", "mask", "action", "logp", "value", "potential")

    def __init__(self):
        self.obs, self.mask, self.action = [], [], []
        self.logp, self.value, self.potential = [], [], []

    def __len__(self):
        return len(self.obs)

    def add(self, obs, mask, action, logp, value, potential):
        self.obs.append(obs)
        self.mask.append(mask)
        self.action.append(action)
        self.logp.append(logp)
        self.value.append(value)
        self.potential.append(potential)

    def clear(self):
        for name in self.__slots__:
            getattr(self, name).clear()


def potential(info, player, target):
    """Φ(s) — the position's victory-point lead, normalised.

    Used for **potential-based** shaping: the reward added to a transition is
    ``Φ(s') - Φ(s)``, which telescopes over an episode and therefore cannot change which
    policy is optimal (Ng, Harada & Russell 1999), while giving the agent something to learn
    from long before it ever completes a game. The alternative — per-step rewards for
    building things — silently rewrites the objective.

    Public points only. Victory-point development cards are hidden, and a shaping term that
    read them would leak information the policy is not allowed to have.
    """
    public = info["public_scores"]
    mine = public[player]
    best_other = max((v for p, v in public.items() if p != player), default=0)
    return (mine - best_other) / target


def outcome(info, player, target):
    """The terminal reward for ``player``: ±1 for a decided game, adjudicated otherwise."""
    winner = info["winner"]
    if winner is not None:
        return 1.0 if winner == player else -1.0
    # truncated: nobody reached the target, so score the position
    return TRUNCATION_WEIGHT * float(np.clip(potential(info, player, target), -1.0, 1.0))


class Rollout:
    """A batch of finished transitions, ready for the PPO update."""

    def __init__(self, obs, mask, action, logp, value, advantage, ret, stats):
        self.obs = obs
        self.mask = mask
        self.action = action
        self.logp = logp
        self.value = value
        self.advantage = advantage
        self.returns = ret
        self.stats = stats

    def __len__(self):
        return self.obs.shape[0]


def compute_gae(rewards, values, gamma, lam):
    """Generalised advantage estimation over **one seat's own decision sequence**.

    The timeline is the seat's consecutive decisions, not the environment's global step
    count — between two of this player's moves the opponent has moved, dice have been rolled
    and cards drawn, all of which are environment dynamics from this seat's point of view.
    Treating the global sequence as the timeline would discount a player's own future by the
    opponent's activity, which is not a quantity anyone wants to discount by.

    The trajectory always ends terminally (a decided or adjudicated game), so there is no
    bootstrap value at the end.
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    running = 0.0
    for t in reversed(range(n)):
        next_value = values[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        running = delta + gamma * lam * running
        advantages[t] = running
    return advantages, advantages + values[:n]


class SelfPlayCollector:
    """Runs ``num_envs`` games in lockstep and returns learner transitions.

    Args:
        net: the policy being trained.
        num_envs: games stepped together. Higher amortises PyTorch overhead across the
            batched forward pass; the environment itself is the throughput ceiling.
        opponent: ``None`` for pure self-play — the same network plays both seats and
            **both** seats' transitions are learned from, which doubles the data per game.
            A second network, or any ``(observation, info) -> index`` callable, plays the
            other seat and only the learner's transitions are kept.
        gamma: 1.0 by default. The game pays out once, at the end; discounting a terminal
            reward only biases it toward whoever moved last.
        shaping: coefficient on potential-based victory-point shaping. 0 disables it.
    """

    def __init__(self, net, num_envs=32, opponent=None, ruleset=RANKED_1V1,
                 gamma=1.0, lam=0.95, shaping=0.3, max_turns=TRAINING_MAX_TURNS,
                 seed=0, device="cpu"):
        self.net = net
        self.opponent_factory = opponent if callable_factory(opponent) else (lambda: (opponent, "fixed"))
        self.gamma, self.lam, self.shaping = gamma, lam, shaping
        self.device = device
        self.target = ruleset.victory_points_to_win
        self.rng = np.random.default_rng(seed)

        self.envs = [
            CatanEnv(num_players=2, ruleset=ruleset, max_turns=max_turns)
            for _ in range(num_envs)
        ]
        self.obs = [None] * num_envs
        self.info = [None] * num_envs
        self.learner_seat = [1] * num_envs
        self.opponents = [None] * num_envs
        self.labels = ["self"] * num_envs
        self.trajectories = [{1: SeatTrajectory(), 2: SeatTrajectory()} for _ in range(num_envs)]
        self._seed_counter = seed * 1_000_003
        for i in range(num_envs):
            self._reset(i)

    # ------------------------------------------------------------------ #

    def _reset(self, i):
        self._seed_counter += 1
        self.obs[i], self.info[i] = self.envs[i].reset(seed=self._seed_counter)
        # Alternate seats so the learner sees both sides. Measured at 20/40 for identical
        # agents, so first-player advantage is small here — but it costs nothing to be sure.
        self.learner_seat[i] = 1 + (self._seed_counter % 2)
        # The opponent is drawn per *game*, not per iteration, so a rollout is a mixture of
        # the pool rather than one opponent at a time — and so that a persistent collector
        # never has to switch a game's opponent halfway through.
        self.opponents[i], self.labels[i] = self.opponent_factory()
        for seat in (1, 2):
            self.trajectories[i][seat].clear()

    def _learner_controls(self, i, player):
        return self.opponents[i] is None or player == self.learner_seat[i]

    # ------------------------------------------------------------------ #

    def collect(self, num_steps):
        """Step until at least ``num_steps`` learner transitions are banked.

        Games in progress when the count is reached are **not** cut short — a half-episode
        has no terminal reward, and inventing one would be worse than waiting. They stay in
        their environments and continue on the next call, which is why **this object must
        outlive one iteration**. Rebuilding it per iteration throws away every unfinished
        game: at 128 environments that measured as 7.5x the necessary work, and made *more*
        environments slower rather than faster.
        """
        banked = {"obs": [], "mask": [], "action": [], "logp": [],
                  "value": [], "advantage": [], "returns": []}
        stats = {"games": 0, "wins": 0, "losses": 0, "truncated": 0, "opponents": {},
                 "lengths": [], "turns": [], "forced": 0, "decisions": 0}

        while len(banked["obs"]) < num_steps:
            self._tick(banked, stats)

        return Rollout(
            obs=torch.as_tensor(np.asarray(banked["obs"], dtype=np.float32)),
            mask=torch.as_tensor(np.asarray(banked["mask"], dtype=bool)),
            action=torch.as_tensor(np.asarray(banked["action"], dtype=np.int64)),
            logp=torch.as_tensor(np.asarray(banked["logp"], dtype=np.float32)),
            value=torch.as_tensor(np.asarray(banked["value"], dtype=np.float32)),
            advantage=torch.as_tensor(np.asarray(banked["advantage"], dtype=np.float32)),
            ret=torch.as_tensor(np.asarray(banked["returns"], dtype=np.float32)),
            stats=stats,
        )

    # ------------------------------------------------------------------ #

    def _tick(self, banked, stats):
        """One decision in every live game, with the network calls batched.

        Environments are grouped by *which network* is deciding, so each group costs one
        forward pass however many games are in it. With a mixed opponent pool that is
        usually two groups — the learner and one frozen self — plus whatever the heuristic
        games need, which is ordinary Python.
        """
        groups = {}
        chosen, record = {}, {}

        for i, info in enumerate(self.info):
            legal = info["legal"]
            if len(legal) == 1:
                chosen[i] = legal[0]                 # no choice: no gradient, no record
                stats["forced"] += 1
                continue

            opponent = self.opponents[i]
            if self._learner_controls(i, info["player"]):
                groups.setdefault(("net", id(self.net)), (self.net, []))[1].append(i)
            elif isinstance(opponent, torch.nn.Module):
                groups.setdefault(("net", id(opponent)), (opponent, []))[1].append(i)
            else:
                chosen[i] = opponent(self.obs[i], self.info[i])

        for (_, net_id), (net, indices) in groups.items():
            learning = net_id == id(self.net)
            for i, action, logp, value, obs_row, mask_row in self._query(net, indices):
                chosen[i] = action
                if learning:
                    record[i] = (logp, value, obs_row, mask_row)

        for i, action in chosen.items():
            self._step(i, action, record.get(i), banked, stats)

    def _query(self, net, indices):
        """One batched forward pass for every game this network is deciding.

        The batch arrays are handed back row by row and stored as-is. Observations arrive
        from the encoder as Python lists of `encoder.SIZE` floats; keeping them in that form
        costs roughly 32 bytes a number, which at 128 environments is about a gigabyte of
        live trajectory. The float32 rows are ~7 KB each and are being built here anyway.
        """
        obs_batch = np.asarray([self.obs[i] for i in indices], dtype=np.float32)
        mask_batch = np.asarray([self.info[i]["mask"] for i in indices], dtype=bool)
        obs = torch.as_tensor(obs_batch, device=self.device)
        mask = torch.as_tensor(mask_batch, device=self.device)

        actions, logps, values = net.act(obs, mask)
        actions = actions.numpy()
        logps = logps.numpy()
        values = values.numpy()
        return [
            (i, int(actions[k]), float(logps[k]), float(values[k]),
             obs_batch[k], mask_batch[k])
            for k, i in enumerate(indices)
        ]

    def _step(self, i, action, record, banked, stats):
        info = self.info[i]
        player = info["player"]
        stats["decisions"] += 1

        if record is not None:
            logp, value, obs_row, mask_row = record
            self.trajectories[i][player].add(
                obs=obs_row,
                mask=mask_row,
                action=action,
                logp=logp,
                value=value,
                potential=potential(info, player, self.target),
            )

        self.obs[i], _, terminated, truncated, self.info[i] = self.envs[i].step(action)

        if terminated or truncated:
            self._finish(i, banked, stats)

    # ------------------------------------------------------------------ #

    def _finish(self, i, banked, stats):
        """Write the outcome onto every seat we were learning from, then bank it."""
        info = self.info[i]
        seats = (1, 2) if self.opponents[i] is None else (self.learner_seat[i],)

        stats["games"] += 1
        stats["opponents"][self.labels[i]] = stats["opponents"].get(self.labels[i], 0) + 1
        stats["turns"].append(info["turn"])
        if info["winner"] is None:
            stats["truncated"] += 1
        elif info["winner"] == self.learner_seat[i]:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

        for seat in seats:
            trajectory = self.trajectories[i][seat]
            if not len(trajectory):
                continue
            stats["lengths"].append(len(trajectory))

            values = np.asarray(trajectory.value, dtype=np.float32)
            phi = np.asarray(trajectory.potential, dtype=np.float32)
            final_phi = potential(info, seat, self.target)

            # Potential-based shaping across this seat's own consecutive decisions, then the
            # real outcome on the last one.
            rewards = np.zeros(len(trajectory), dtype=np.float32)
            if self.shaping:
                nxt = np.concatenate([phi[1:], [final_phi]])
                rewards += self.shaping * (self.gamma * nxt - phi)
            rewards[-1] += outcome(info, seat, self.target)

            advantages, returns = compute_gae(rewards, values, self.gamma, self.lam)

            banked["obs"].extend(trajectory.obs)
            banked["mask"].extend(trajectory.mask)
            banked["action"].extend(trajectory.action)
            banked["logp"].extend(trajectory.logp)
            banked["value"].extend(trajectory.value)
            banked["advantage"].extend(advantages.tolist())
            banked["returns"].extend(returns.tolist())

        self._reset(i)

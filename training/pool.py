"""The pool of opponents a policy trains against.

Pure self-play — current policy against a copy of itself — is the obvious thing and it has a
specific, well-documented way of going wrong: the policy improves against *its present self*
and forgets how to handle what it used to do. In a game with rock-paper-scissors structure
this produces cycling, where a checkpoint from iteration 300 loses to one from iteration 100.
The symptom from the outside is a training run whose self-play win rate hovers at exactly 50%
— which is what self-play always shows, whether it is improving or going in circles.

So the opponent is sampled each iteration:

- **the current policy** most of the time — the actual self-play signal, and the only one
  that keeps improving once the agent is past everything else;
- **a frozen past checkpoint** sometimes — pressure not to forget;
- **the heuristic** occasionally — an *external* anchor. Without one, the only measure of
  progress is relative, and relative progress is exactly what cycling fakes.

The heuristic's share is deliberately small. It is a floor, not a target: an agent that
trains mostly against it learns to beat *it*, and inherits its blind spots.
"""

import copy
import random

from catan.agents import HeuristicAgent
from training.net import PolicyValueNet


class OpponentPool:
    """Frozen past selves, plus the fixed baseline.

    Args:
        capacity: how many checkpoints to keep. Older ones are dropped from the middle
            rather than the front, so the pool keeps a spread of ages instead of becoming
            "the last N iterations", which would defeat the point.
        self_play: probability of playing the live policy.
        heuristic: probability of playing :class:`~catan.agents.HeuristicAgent`.
            The remainder goes to frozen checkpoints.
    """

    def __init__(self, capacity=10, self_play=0.6, heuristic=0.15, seed=0):
        self.capacity = capacity
        self.self_play = self_play
        self.heuristic_share = heuristic
        self.rng = random.Random(seed)
        self.frozen = []            # list of (iteration, state_dict)
        self.config = None

    # ------------------------------------------------------------------ #

    def add(self, net, iteration):
        """Freeze a copy of the current policy."""
        self.config = net.config()
        weights = {k: v.detach().clone() for k, v in net.state_dict().items()}
        self.frozen.append((iteration, weights))
        if len(self.frozen) > self.capacity:
            # drop from the middle: keep the oldest (the widest test) and the newest
            # (the hardest), and thin out what is between them
            self.frozen.pop(len(self.frozen) // 2)

    def __len__(self):
        return len(self.frozen)

    # ------------------------------------------------------------------ #

    def sample(self, net, iteration):
        """Pick this iteration's opponent.

        Returns:
            ``(opponent, label)``. ``opponent`` is ``None`` for self-play — the collector
            reads that as "one network plays both seats", which also doubles the
            transitions banked per game.
        """
        draw = self.rng.random()

        if draw < self.self_play or not self.frozen:
            return None, "self"

        if draw < self.self_play + self.heuristic_share:
            return HeuristicAgent(self.rng.randrange(1 << 30)), "heuristic"

        # Sample a frozen self, biased toward recent ones: an ancient checkpoint is easy and
        # spending a whole iteration beating it teaches little.
        weights = [i + 1 for i in range(len(self.frozen))]
        index = self.rng.choices(range(len(self.frozen)), weights=weights)[0]
        stored_iteration, state = self.frozen[index]
        frozen_net = PolicyValueNet.from_config(self.config)
        frozen_net.load_state_dict(state)
        frozen_net.eval()
        return frozen_net, f"frozen@{stored_iteration}"

    def snapshot(self):
        """Serialisable form, so a run can resume with its pool intact."""
        return {"config": self.config,
                "frozen": [(i, {k: v.clone() for k, v in w.items()}) for i, w in self.frozen]}

    def restore(self, snapshot):
        if not snapshot:
            return
        self.config = snapshot["config"]
        self.frozen = list(snapshot["frozen"])

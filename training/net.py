"""The policy and value network.

A plain MLP over the observation (``encoder.SIZE`` floats), with two heads: one logit per
action and one value. Shared trunk, because the features that say "this position is winning"
are the same
features that say "build here" — and on CPU a second trunk doubles the cost of every
forward pass in the rollout, which is where the time goes.

**Masking is part of the network, not the caller.** Every method that produces a
distribution takes the legality mask and applies it, because the single most expensive bug
available here is a mask applied during the rollout but not during the update: the
recomputed log-probabilities would then be from a different distribution, the PPO ratio
would be meaningless, and nothing would crash.

The mask is applied by *adding* a large negative number rather than ``-inf``. With ``-inf``,
a state where one action is legal gives ``0 * inf = nan`` in the entropy, which poisons the
gradient silently.
"""

import torch
from torch import nn

#: Added to illegal logits. Large enough that ``softmax`` gives exactly 0 in float32,
#: finite so that entropy stays a number.
MASK_FILL = -1e8


def _layer(in_size, out_size, gain):
    """Orthogonal init, which for PPO is worth more than it looks.

    The policy head gets a tiny gain so the initial distribution is near-uniform over the
    legal moves: a policy that starts out confident explores nothing and PPO's clipping
    cannot rescue it.
    """
    layer = nn.Linear(in_size, out_size)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class PolicyValueNet(nn.Module):
    """``observation -> (one logit per action, state value)``.

    Args:
        obs_size: length of an observation, ``catan.encoder.SIZE``.
        num_actions: ``catan.action_space.NUM_ACTIONS``.
        hidden: trunk widths.
    """

    def __init__(self, obs_size, num_actions, hidden=(512, 512)):
        super().__init__()
        self.obs_size = obs_size
        self.num_actions = num_actions
        self.hidden = tuple(hidden)

        layers, last = [], obs_size
        for width in hidden:
            layers += [_layer(last, width, gain=2 ** 0.5), nn.Tanh()]
            last = width
        self.trunk = nn.Sequential(*layers)
        self.policy_head = _layer(last, num_actions, gain=0.01)
        self.value_head = _layer(last, 1, gain=1.0)

    # ------------------------------------------------------------------ #

    def forward(self, obs):
        features = self.trunk(obs)
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    def masked_logits(self, obs):
        raise NotImplementedError("call forward() and mask, or use act()/evaluate()")

    @staticmethod
    def _apply_mask(logits, mask):
        """``mask`` is a bool tensor of the same shape; ``True`` means legal."""
        return logits.masked_fill(~mask, MASK_FILL)

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def act(self, obs, mask, deterministic=False, generator=None):
        """Sample one action per row.

        Returns:
            ``(action, log_prob, value)`` — all detached, shaped ``(batch,)``.
        """
        logits, value = self.forward(obs)
        logits = self._apply_mask(logits, mask)
        distribution = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        elif generator is None:
            action = distribution.sample()
        else:
            # torch.distributions ignores generators, so sample by hand when a run has to
            # be reproducible.
            probabilities = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
        return action, distribution.log_prob(action), value

    def evaluate(self, obs, mask, action):
        """Re-score stored actions during the update. Differentiable.

        The mask must be the one stored at rollout time, not one recomputed from the state.
        """
        logits, value = self.forward(obs)
        logits = self._apply_mask(logits, mask)
        distribution = torch.distributions.Categorical(logits=logits)
        return distribution.log_prob(action), distribution.entropy(), value

    # ------------------------------------------------------------------ #

    def config(self):
        """Everything needed to rebuild this network, for the checkpoint."""
        return {
            "obs_size": self.obs_size,
            "num_actions": self.num_actions,
            "hidden": self.hidden,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def __repr__(self):
        return (f"PolicyValueNet({self.obs_size} -> {list(self.hidden)} -> "
                f"{self.num_actions}, {self.num_parameters():,} params)")


#: Networks that can appear in a checkpoint, keyed by the ``kind`` in their config.
#:
#: A checkpoint has to say which class to rebuild. The flat network predates the field, so
#: its absence means "flat" — old checkpoints keep loading.
def build(config):
    """Rebuild whichever network a checkpoint's config describes."""
    kind = config.get("kind", "flat")
    if kind == "flat":
        return PolicyValueNet.from_config(
            {k: v for k, v in config.items() if k != "kind"}
        )
    if kind == "structured":
        from training.structured_net import StructuredPolicyValueNet

        return StructuredPolicyValueNet.from_config(
            {k: v for k, v in config.items() if k != "kind"}
        )
    raise ValueError(f"unknown network kind {kind!r} in checkpoint")

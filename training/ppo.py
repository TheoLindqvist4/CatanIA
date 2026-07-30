"""The PPO update.

Nothing exotic — clipped surrogate objective, clipped value loss, entropy bonus, gradient
clipping. What matters is that every diagnostic a failed run needs is computed here and
returned, because the failure modes are silent: a PPO run that is quietly broken looks
exactly like a PPO run that is learning slowly, and the difference is only visible in the
KL, the clip fraction and the explained variance.

Two decisions worth naming:

**The mask comes from the buffer, never from the state.** If the mask used to re-score an
action differs by one bit from the mask used to choose it, the ratio is between two different
distributions and the whole objective is nonsense. Recomputing it would be a second source of
truth, so the rollout's mask is carried through.

**Advantages are normalised per minibatch, not per batch.** Standard, and it matters more
here than usual: episode returns are ±1 with a long tail of adjudicated draws near 0, so a
whole-batch normalisation is dominated by the win/loss split rather than by within-position
differences.
"""

import numpy as np
import torch
from torch import nn


class PPO:
    """Args are the hyperparameters; :meth:`update` consumes a :class:`~training.rollout.Rollout`.

    Args:
        net: the :class:`~training.net.PolicyValueNet` being trained.
        lr: Adam step size. ``3e-4`` is the usual starting point; the schedule is the
            trainer's business, not this class's.
        clip: PPO's ε. 0.2 standard.
        value_clip: clip range for the value function, or ``None`` to leave it unclipped.
        epochs: passes over each rollout. More is more sample-efficient and more likely to
            walk off the trust region; ``target_kl`` is the guard.
        minibatch: rows per gradient step.
        entropy_coef: exploration pressure. Catan's legal-action count swings from 2 to 54,
            so raw entropy is not comparable between states — see the note in
            :meth:`update` on why it is still the right thing to add unnormalised.
        value_coef: weight on the value loss (shared trunk, so this trades off against the
            policy).
        max_grad_norm: global gradient clip.
        target_kl: stop the epoch loop early if the policy has moved this far. The single
            most useful safety valve in PPO.
    """

    def __init__(self, net, lr=3e-4, clip=0.2, value_clip=0.2, epochs=4, minibatch=512,
                 entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5, target_kl=0.03):
        self.net = net
        self.optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)
        self.clip = clip
        self.value_clip = value_clip
        self.epochs = epochs
        self.minibatch = minibatch
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

    # ------------------------------------------------------------------ #

    def set_lr(self, lr):
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    @property
    def lr(self):
        return self.optimizer.param_groups[0]["lr"]

    # ------------------------------------------------------------------ #

    def update(self, rollout):
        """One PPO update over ``rollout``. Returns the diagnostics.

        The entropy bonus is added unnormalised. Normalising by ``log(legal count)`` would
        make it comparable across states, but it would also weight the bonus *against*
        exactly the states with the most choices — the ones where exploration is worth
        something. The measured spread (median 5 legal actions, p95 26) is narrow enough
        that the unnormalised version does not distort much.
        """
        n = len(rollout)
        indices = np.arange(n)
        diagnostics = {k: [] for k in
                       ("policy_loss", "value_loss", "entropy", "kl", "clip_fraction")}
        stopped_early = False

        for epoch in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.minibatch):
                batch = indices[start:start + self.minibatch]
                if len(batch) < 2:
                    continue
                stats = self._step(rollout, batch)
                for key, value in stats.items():
                    diagnostics[key].append(value)

            if self.target_kl is not None and diagnostics["kl"]:
                recent = float(np.mean(diagnostics["kl"][-max(1, n // self.minibatch):]))
                if recent > self.target_kl:
                    stopped_early = True
                    break

        summary = {key: float(np.mean(values)) if values else 0.0
                   for key, values in diagnostics.items()}
        summary["epochs_run"] = epoch + 1
        summary["stopped_early"] = stopped_early
        summary["explained_variance"] = explained_variance(
            rollout.value.numpy(), rollout.returns.numpy()
        )
        summary["lr"] = self.lr
        return summary

    # ------------------------------------------------------------------ #

    def _step(self, rollout, batch):
        obs = rollout.obs[batch]
        mask = rollout.mask[batch]
        action = rollout.action[batch]
        old_logp = rollout.logp[batch]
        old_value = rollout.value[batch]
        advantage = rollout.advantage[batch]
        returns = rollout.returns[batch]

        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        logp, entropy, value = self.net.evaluate(obs, mask, action)
        ratio = torch.exp(logp - old_logp)

        unclipped = ratio * advantage
        clipped = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantage
        policy_loss = -torch.min(unclipped, clipped).mean()

        if self.value_clip is None:
            value_loss = 0.5 * (value - returns).pow(2).mean()
        else:
            bounded = old_value + torch.clamp(
                value - old_value, -self.value_clip, self.value_clip
            )
            value_loss = 0.5 * torch.max(
                (value - returns).pow(2), (bounded - returns).pow(2)
            ).mean()

        entropy_mean = entropy.mean()
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_mean

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            # Schulman's low-variance KL estimator; the naive (old_logp - logp).mean() is
            # unbiased but noisy enough to trigger the early stop at random.
            log_ratio = logp - old_logp
            kl = ((ratio - 1) - log_ratio).mean()
            clip_fraction = ((ratio - 1).abs() > self.clip).float().mean()

        return {
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy_mean.detach()),
            "kl": float(kl),
            "clip_fraction": float(clip_fraction),
        }


def explained_variance(predicted, actual):
    """``1 - Var(actual - predicted) / Var(actual)``.

    The value head's report card, and the first number to look at when a run is not
    improving. Near 0 means the critic knows nothing and every advantage is noise; near 1
    means it predicts the outcome from the position. Negative means it is worse than
    guessing the mean, which is a bug rather than slow learning.
    """
    variance = np.var(actual)
    if variance < 1e-12:
        return 0.0
    return float(1 - np.var(actual - predicted) / variance)

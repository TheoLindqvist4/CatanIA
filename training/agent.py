"""A trained policy, wearing the same interface as every other agent.

    from training.agent import PolicyAgent
    agent = PolicyAgent.load("checkpoints/latest.pt")
    play_match({1: agent, 2: HeuristicAgent(0)}, games=100)

This is the whole payoff of having kept the agent interface to
``(observation, info) -> index``: a network drops into ``interfaces.web.api.OPPONENTS`` and
``interfaces.cli.AGENTS`` as one more entry, and neither interface changes.
"""

import numpy as np
import torch

from training.net import PolicyValueNet


class PolicyAgent:
    """Wraps a :class:`~training.net.PolicyValueNet` as a callable agent.

    Args:
        net: the network.
        temperature: 0 plays the argmax — strongest, and completely predictable, which makes
            it a poor sparring partner for a human. Above 0 samples, so the same position
            does not always get the same reply.
        seed: for sampling.
    """

    def __init__(self, net, temperature=0.0, seed=None):
        self.net = net.eval()
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, path, temperature=0.0, seed=None, map_location="cpu"):
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        net = PolicyValueNet.from_config(checkpoint["config"])
        net.load_state_dict(checkpoint["weights"])
        agent = cls(net, temperature=temperature, seed=seed)
        agent.metadata = {k: v for k, v in checkpoint.items()
                          if k not in ("weights", "optimizer")}
        return agent

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def __call__(self, observation, info):
        mask = info["mask"]
        legal = info["legal"]
        if len(legal) == 1:
            return legal[0]

        obs = torch.as_tensor(np.asarray([observation], dtype=np.float32))
        mask_t = torch.as_tensor(np.asarray([mask], dtype=bool))
        logits, _ = self.net(obs)
        logits = self.net._apply_mask(logits, mask_t)[0].numpy()

        if self.temperature <= 0:
            return int(np.argmax(logits))

        scaled = logits / self.temperature
        scaled -= scaled.max()
        probabilities = np.exp(scaled)
        probabilities /= probabilities.sum()
        return int(self.rng.choice(len(probabilities), p=probabilities))

    @torch.no_grad()
    def value(self, observation):
        """The critic's opinion of a position, in [-1, 1]. Useful for a UI, and for debugging."""
        obs = torch.as_tensor(np.asarray([observation], dtype=np.float32))
        _, value = self.net(obs)
        return float(value[0])

    def __repr__(self):
        return f"PolicyAgent(temperature={self.temperature}, {self.net!r})"


def export(checkpoint, path):
    """Write a slim checkpoint: the network, and nothing else.

    A training checkpoint carries the optimiser state, the frozen opponent pool and the
    full per-iteration history, which is 70 MB and exactly the right thing to keep for
    resuming a run. None of it is needed to *play*, and shipping it would mean the web
    interface loading fourteen networks to use one.
    """
    import pathlib
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": source["config"],
        "weights": source["weights"],
        "iteration": source.get("iteration"),
        "trained_at": source.get("history", [])[-1] if source.get("history") else None,
    }, path)
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export a slim, playable checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("out")
    arguments = parser.parse_args()
    written = export(arguments.checkpoint, arguments.out)
    size = written.stat().st_size / 1e6
    print(f"wrote {written} ({size:.1f} MB)")

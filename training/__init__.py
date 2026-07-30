"""Self-play training for the Catan engine.

The engine itself stays dependency-free; this package is the only thing that needs PyTorch.

    python -m training.train --smoke          # 60-second pipeline check
    python -m training.train --iterations 400 # a real run

A finished checkpoint drops into the interfaces as one more opponent, because an agent is
only a ``(observation, info) -> index`` callable:

    from training.agent import PolicyAgent
    OPPONENTS["learned"] = lambda seed: PolicyAgent.load("checkpoints/best.pt", seed=seed)
"""

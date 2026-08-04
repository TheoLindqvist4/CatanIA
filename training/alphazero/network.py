"""The network AlphaZero trains, and how it is started.

Not a new architecture. :class:`~training.structured_net.StructuredPolicyValueNet` already
produces exactly what AlphaZero needs — one logit per action laid out by board element, and
one value — and it is already the structure-aware model the guide asks for: 19 tiles, 54
vertices and 72 roads embedded by shared per-entity MLPs, with neighbourhood information
carried by constant incidence matrices from :mod:`catan.topology`. That *is* a graph network
over the Catan graph; it simply exploits the graph being the same in every game to do the
message passing as a matmul against a constant instead of a gather.

Two things are different here.

**The value head is bounded.** AlphaZero's target is exactly ``+1``/``-1`` and the loss is a
mean square error, so ``tanh`` is applied — see ``value_activation`` on the network.

**The run is warm-started, not started.** A cold AlphaZero net plays randomly, and at the
simulation counts a 20-core CPU can afford, random self-play produces a policy target barely
better than its own prior. The repository already contains a policy that beats the heuristic
about 71% of the time. :func:`graft` carries those weights onto the current observation, so
search begins as *policy improvement over something that already plays*, which is the regime
AlphaZero is strong in. Recorded as a deliberate departure from "learns entirely from
self-play" in ``docs/decisions/0023-alphazero-self-play.md``.

**Why a graft is needed at all.** ``models/champion.pt`` was promoted when
``encoder.SIZE`` was 1868. The affordability block added 16 floats at offset 1773, so the
encoder is now 1884 and the champion no longer loads — ``training.champion.load()`` returns
``None`` today, which is why the interfaces currently offer no learned opponent. Only one
tensor is the wrong shape: the first layer of ``context_mlp``, whose input is the
un-positional tail of the observation. Every other parameter is untouched by the change,
because the tiles/vertices/roads encoders read fixed-width rows and the affordability block
is not one of them.

The 16 new columns are inserted **as zeros**. That is the whole reason this is safe rather
than a guess: a zero column contributes nothing, so the grafted network computes the same
function as the original on the same position, and the new features start neutral and are
learned. :func:`graft` returns the number of columns it inserted so a caller can refuse a
checkpoint it does not understand instead of silently mangling one.
"""

import pathlib

import torch

from catan import action_space, encoder
from training.net import build
from training.structured_net import CONTEXT_START, StructuredPolicyValueNet

#: The layer whose input width follows ``encoder.SIZE``. The only one.
CONTEXT_LAYER = "context_mlp.0.weight"


def new_network(width=64, road_width=32, context=128, hops=1, depth=2, rounds=0,
                trunk=256):
    """A fresh AlphaZero network at the current observation and action space."""
    return StructuredPolicyValueNet(
        obs_size=encoder.SIZE,
        num_actions=action_space.NUM_ACTIONS,
        width=width, road_width=road_width, context=context,
        hops=hops, depth=depth, rounds=rounds, trunk=trunk,
        value_activation="tanh",
    )


# --------------------------------------------------------------------------- #
# Carrying a checkpoint across an observation change                          #
# --------------------------------------------------------------------------- #

def insertion_point(old_size, new_size=None):
    """Where the new observation columns were inserted, as an index into the context vector.

    Derived from :data:`catan.encoder.LAYOUT` rather than written down: the growth is
    located by finding the block whose length accounts for the difference. Returns
    ``(offset, count)`` in context-vector coordinates, or ``None`` if the difference cannot
    be explained by exactly one block — in which case the caller must not graft.
    """
    new_size = encoder.SIZE if new_size is None else new_size
    grew_by = new_size - old_size
    if grew_by <= 0:
        return None
    for name, span in encoder.LAYOUT.items():
        if span.stop - span.start != grew_by:
            continue
        if span.start < CONTEXT_START:
            return None                       # a positional block grew; rows change shape
        return span.start - CONTEXT_START, grew_by
    return None


def graft(state_dict, config):
    """Widen a checkpoint's context layer to the current observation, with zero columns.

    Args:
        state_dict: the stored weights.
        config: the stored config, which carries the ``obs_size`` they were trained at.

    Returns:
        ``(state_dict, inserted)`` — a new dict safe to load into a network built at the
        current :data:`catan.encoder.SIZE`, and how many columns were added. ``inserted``
        is 0 when the checkpoint already matches.

    Raises:
        ValueError: when the shapes cannot be reconciled by inserting one block. Loud
            rather than approximate: a checkpoint whose layout is not understood should be
            retrained, not bent into shape.
    """
    old_size = config.get("obs_size")
    if old_size == encoder.SIZE:
        return dict(state_dict), 0
    if config.get("kind") != "structured":
        raise ValueError(f"cannot graft a {config.get('kind', 'flat')!r} checkpoint: only "
                         f"the structured network has a single observation-width layer")

    where = insertion_point(old_size)
    if where is None:
        raise ValueError(
            f"checkpoint was trained at obs_size={old_size} and the encoder is now "
            f"{encoder.SIZE}; the difference of {encoder.SIZE - old_size} does not match "
            f"exactly one non-positional block, so where the new floats belong is unknown"
        )
    offset, count = where

    grafted = dict(state_dict)
    weight = grafted[CONTEXT_LAYER]
    if weight.shape[1] + count != encoder.SIZE - CONTEXT_START:
        raise ValueError(
            f"{CONTEXT_LAYER} is {tuple(weight.shape)}, which does not become the expected "
            f"{encoder.SIZE - CONTEXT_START} inputs by inserting {count} columns"
        )
    zeros = torch.zeros(weight.shape[0], count, dtype=weight.dtype, device=weight.device)
    grafted[CONTEXT_LAYER] = torch.cat(
        [weight[:, :offset], zeros, weight[:, offset:]], dim=1
    )
    return grafted, count


def load_for_alphazero(path, value_activation="tanh"):
    """Load any structured checkpoint as an AlphaZero network at the current observation.

    Grafts if needed, and switches the value head to ``tanh``. Switching the activation
    changes what the *existing* value head means — an unbounded head trained by PPO produces
    numbers outside ``[-1, 1]`` that ``tanh`` will saturate — so the head is re-initialised
    small rather than carried over. The policy, which is what warm-starting is for, is kept
    in full.

    Returns:
        ``(net, notes)`` where ``notes`` is a dict describing what was done, for the record.
    """
    path = pathlib.Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    weights, inserted = graft(checkpoint["weights"], config)

    config["obs_size"] = encoder.SIZE
    config["value_activation"] = value_activation
    net = build(config)

    if value_activation != checkpoint["config"].get("value_activation", "linear"):
        # A PPO critic is a return predictor on a different scale. Keeping it would start
        # the run with a confident, wrong, saturated value head, and AlphaZero's search is
        # steered by the value. Reset it; it is 257 numbers.
        torch.nn.init.orthogonal_(net.value_head.weight, gain=0.01)
        torch.nn.init.constant_(net.value_head.bias, 0.0)
        weights = {k: v for k, v in weights.items() if not k.startswith("value_head.")}

    missing, unexpected = net.load_state_dict(weights, strict=False)
    if unexpected:
        raise ValueError(f"checkpoint has parameters this network does not: {unexpected}")
    return net, {
        "source": str(path),
        "trained_at_obs_size": checkpoint["config"].get("obs_size"),
        "grafted_columns": inserted,
        "reset": sorted(missing),
    }

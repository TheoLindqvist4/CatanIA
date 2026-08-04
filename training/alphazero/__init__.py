"""AlphaZero-style self-play for 1v1 Catan.

The pipeline the design guide asks for — self-play, replay buffer, trainer, evaluator,
gated promotion — built on top of the engine and observation that already exist rather
than beside a second copy of them.

    engine + environment   catan/            unchanged
    search                 mcts.py           PUCT over determinized information sets
    generation             self_play.py      + workers.py for the process pool
    storage                replay_buffer.py
    learning               trainer.py
    measurement            evaluator.py
    what you play against  champion.py       models/champion_az.pt

The one rule that shapes everything here: **search may not see what the player may not
see**. `GameState.clone` copies the hidden state verbatim, so a tree built on the true
state would be reading the opponent's hand and the top of the development deck. Every
search in this package therefore runs on a state produced by :mod:`determinize`, which
resamples every hidden quantity from what is public. See
``docs/decisions/0023-alphazero-self-play.md``.
"""

from training.alphazero.config import Config, load_config

__all__ = ["Config", "load_config"]

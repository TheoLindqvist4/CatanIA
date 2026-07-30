"""CatanIA engine.

Layers, innermost first:

* :mod:`catan.topology` — immutable geometry, generated from the row structure.
* :mod:`catan.resources` — the five resources and what things cost.
* :mod:`catan.board` — one board layout: numbers, resources, production index.
  **Immutable after construction**, so clones share it.
* :mod:`catan.state` — :class:`~catan.state.GameState`: everything mutable, and
  nothing else. Cheap :meth:`~catan.state.GameState.clone`.
* :mod:`catan.actions` — what a player can do, as data.
* :mod:`catan.rules` — the only legality authority: ``legal_actions`` and ``apply``.

Nothing in this package performs I/O or touches the global ``random`` module.
"""

from catan.actions import Action, ActionType
from catan.board import Board, Production
from catan.resources import (
    CITY_COST,
    DEV_CARD_COST,
    NUM_RESOURCES,
    ROAD_COST,
    SETTLEMENT_COST,
    Resource,
)
from catan.state import GameState, Phase, Piece

__all__ = [
    "Action",
    "ActionType",
    "Board",
    "Production",
    "Resource",
    "NUM_RESOURCES",
    "ROAD_COST",
    "SETTLEMENT_COST",
    "CITY_COST",
    "DEV_CARD_COST",
    "GameState",
    "Phase",
    "Piece",
]

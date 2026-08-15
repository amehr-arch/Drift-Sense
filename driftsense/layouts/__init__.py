"""Die-architecture layout models and their registry.

New architectures are added by implementing :class:`~driftsense.layouts.base.LayoutModel`
and registering the class here. Nothing outside this package needs to change --
in particular the generator, the geometry core and the localiser are all
architecture-agnostic by construction.
"""

from __future__ import annotations

from typing import Dict, Type

from .base import LayoutModel, LayoutRender, RenderWindow
from .dram import CELL_ARCHITECTURES, DramLayout, DramParams
from .finfet import FinfetLayout, FinfetParams

__all__ = [
    "LayoutModel",
    "LayoutRender",
    "RenderWindow",
    "DramLayout",
    "DramParams",
    "FinfetLayout",
    "FinfetParams",
    "CELL_ARCHITECTURES",
    "LAYOUT_REGISTRY",
    "available_architectures",
    "get_layout_class",
]

#: Architecture name -> layout class.
LAYOUT_REGISTRY: Dict[str, Type[LayoutModel]] = {
    DramLayout.name: DramLayout,
    FinfetLayout.name: FinfetLayout,
}


def available_architectures() -> tuple:
    """Names accepted by ``--architecture`` on the command line."""
    return tuple(sorted(LAYOUT_REGISTRY))


def get_layout_class(name: str) -> Type[LayoutModel]:
    """Look up a layout class by architecture name, with a helpful error."""
    try:
        return LAYOUT_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown architecture {name!r}; available: {', '.join(available_architectures())}"
        ) from None

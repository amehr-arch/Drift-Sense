"""Abstract interface every die-architecture layout model implements.

DESIGN NOTE -- WHY A LAYER DICTIONARY RATHER THAN A GREYSCALE IMAGE
--------------------------------------------------------------------
``LayoutModel.render`` returns per-material *coverage fields* rather than a
finished greyscale picture. Coverage is the fraction of each pixel occupied by a
given material, in ``[0, 1]``.

This is the seam between geometry and physics. A scanning electron microscope
does not image "grey"; it images secondary-electron yield, which depends on
material and on local topography, and it brightens edges where the yield rises on
sidewalls. Keeping coverage per material means the imaging model can
assign a yield per material and compute edge terms from the coverage gradients,
without the geometry code having to know anything about electron optics.

THE LAYERS ARE DISJOINT, AND THAT MATTERS
-----------------------------------------
Each layer reports the coverage of the material *visible* in that pixel -- the
topmost one, after occlusion has been resolved. The layers are therefore mutually
exclusive and sum to at most one, with the remainder being exposed substrate.

The earlier design stored each layer's full extent and composited them with a
painter's algorithm using coverage as an alpha channel. That is exact only when
the sets involved are independent within the pixel, which fails wherever one
layer is nested inside another that is itself only partially covering -- in
practice, on mat-boundary pixels. It left a residual of up to 0.79 of an 8-bit
grey level between a fine render and a coarse one of the same region.

Resolving occlusion in the layout model instead, where the set relationships are
known exactly, makes the composite a plain area-weighted sum and therefore exact.
The residual is now bounded only by the sub-sample quantisation of the coverage
vectors themselves. It also gives the imaging model what it actually
needs: the fraction of each pixel from which each material is emitting.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, Sequence

import numpy as np

__all__ = [
    "RenderWindow",
    "LayoutRender",
    "LayoutModel",
    "COVERAGE_TOLERANCE",
    "periodic_block_edges",
]



def periodic_block_edges(
    lo_nm: float, hi_nm: float, period_nm: float, size_nm: float, phase_nm: float
) -> tuple:
    """Edges of a periodic run of blocks that fall within ``[lo_nm, hi_nm]``.

    Both shipped layouts are the same shape of thing: a lattice interrupted at
    regular intervals by a gap. DRAM calls the blocks mats and the gaps periphery
    strips; FinFET calls them cell blocks and diffusion breaks. The arithmetic
    that finds their edges is identical, and it lives here so the two cannot
    drift apart.

    Both edges of each block are reported -- the leading edge and the trailing
    one -- because either is an equally good anchor for a localiser.
    """
    if period_nm <= 0.0:
        raise ValueError(f"period_nm must be positive, got {period_nm}")
    first = int(np.floor((lo_nm - phase_nm) / period_nm)) - 1
    last = int(np.ceil((hi_nm - phase_nm) / period_nm)) + 1
    edges = []
    for k in range(first, last + 1):
        start = k * period_nm + phase_nm
        for edge in (start, start + size_nm):
            if lo_nm <= edge <= hi_nm:
                edges.append(float(edge))
    return tuple(sorted(edges))


@dataclass(frozen=True)
class RenderWindow:
    """A rectangular, axis-aligned request for rendered layout.

    Parameters
    ----------
    origin_x_nm, origin_y_nm:
        World coordinates of the top-left corner of the window.
    size_px:
        Side length of the output array in pixels.
    pixel_size_nm:
        Physical size of one output pixel. Coverage is integrated exactly over
        this footprint, so the same window can be requested at any resolution.
    """

    origin_x_nm: float
    origin_y_nm: float
    size_px: int
    pixel_size_nm: float

    def __post_init__(self) -> None:
        if self.size_px <= 0:
            raise ValueError("size_px must be positive")
        if self.pixel_size_nm <= 0:
            raise ValueError("pixel_size_nm must be positive")

    @property
    def extent_nm(self) -> float:
        return self.size_px * self.pixel_size_nm

    @property
    def bounds_nm(self) -> tuple:
        """``(x0, y0, x1, y1)`` world bounds of the window."""
        return (
            self.origin_x_nm,
            self.origin_y_nm,
            self.origin_x_nm + self.extent_nm,
            self.origin_y_nm + self.extent_nm,
        )


#: Tolerance on the "coverages sum to at most one" invariant, in coverage units.
#: Sized to absorb float32 accumulation over a handful of layers, and nothing more.
COVERAGE_TOLERANCE = 1e-4


@dataclass
class LayoutRender:
    """Per-material *visible* coverage fields for one rendered window.

    Attributes
    ----------
    layers:
        Mapping of material name to a ``(size, size)`` float32 array in ``[0, 1]``.
        The layers are mutually exclusive: each reports the fraction of the pixel
        from which that material is visible, after occlusion. They sum to at most
        one, the remainder being exposed substrate.
    intensities:
        Flat greyscale value in ``[0, 1]`` for each material, used by
        ``to_greyscale``. an earlier revision replaces this with a physical yield model.
    background:
        Greyscale value of exposed substrate.
    """

    layers: Dict[str, np.ndarray]
    intensities: Dict[str, float]
    background: float = 0.4
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        shapes = {name: arr.shape for name, arr in self.layers.items()}
        if len(set(shapes.values())) > 1:
            raise ValueError(f"layer shapes disagree: {shapes}")
        for name in self.layers:
            if name not in self.intensities:
                raise ValueError(f"no intensity defined for layer {name!r}")

    @property
    def shape(self) -> tuple:
        return next(iter(self.layers.values())).shape

    def total_coverage(self) -> np.ndarray:
        """Fraction of each pixel covered by any material."""
        total = np.zeros(self.shape, dtype=np.float32)
        for coverage in self.layers.values():
            total = total + coverage
        return total

    def check_disjoint(self, tolerance: float = COVERAGE_TOLERANCE) -> None:
        """Assert the disjointness invariant, raising with a useful message.

        Called by the test suite rather than on every render: the invariant is a
        property of a layout model's occlusion arithmetic, so it needs checking
        once per model, not once per image.
        """
        total = self.total_coverage()
        worst = float(total.max())
        if worst > 1.0 + tolerance:
            raise ValueError(
                f"layer coverages are not disjoint: they sum to {worst:.6f} at some "
                f"pixel, which exceeds 1. Occlusion is being double-counted."
            )
        least = min(float(c.min()) for c in self.layers.values())
        if least < -tolerance:
            raise ValueError(
                f"a layer has negative coverage ({least:.6f}); an occlusion "
                f"subtraction has gone the wrong way."
            )

    def to_greyscale(self) -> np.ndarray:
        """Composite into a float32 image in ``[0, 1]``.

        Because the layers are disjoint visible-material fractions, the composite
        is a plain area-weighted sum -- which is exactly the area average of the
        underlying piecewise-constant image over the pixel footprint. Partially
        covered edge pixels are therefore anti-aliased correctly rather than
        approximately.
        """
        image = np.zeros(self.shape, dtype=np.float32)
        covered = np.zeros(self.shape, dtype=np.float32)
        for name, coverage in self.layers.items():
            image += np.float32(self.intensities[name]) * coverage
            covered += coverage
        image += np.float32(self.background) * (1.0 - covered)
        return np.clip(image, 0.0, 1.0, out=image)

    def to_uint8(self) -> np.ndarray:
        """Composite and quantise to the 8-bit greyscale the dataset ships as."""
        return np.rint(self.to_greyscale() * 255.0).astype(np.uint8)


class LayoutModel(abc.ABC):
    """A parametric, procedurally rendered die layout.

    Implementations must be *stateless with respect to the window*: rendering the
    same world region at two different pixel sizes must describe the same
    physical structure. This is what makes the reference (1 nm/px) and search
    (10 nm/px) captures genuinely consistent, and it is what allows the ground
    truth to be computed arithmetically rather than estimated.
    """

    #: Short identifier used in filenames, metadata and the CLI.
    name: str = "base"

    @abc.abstractmethod
    def render(self, window: RenderWindow) -> LayoutRender:
        """Render coverage fields for the requested window."""

    @abc.abstractmethod
    def describe(self) -> Dict[str, object]:
        """Return a JSON-serialisable record of every structural parameter."""

    def boundary_coordinates_nm(self, axis: str, lo_nm: float, hi_nm: float) -> Sequence[float]:
        """World coordinates of structural discontinuities along one axis.

        These are the features that break translational periodicity -- array
        (mat) edges, pitch changes, alignment marks -- and are therefore the only
        places a localiser can find an unambiguous anchor. The placement sampler
        uses them to deliberately construct both easy cases (boundary inside the
        reference window) and hard cases (deep inside a uniform array).

        The default implementation reports no boundaries, which is the correct
        behaviour for a perfectly periodic layout.
        """
        return ()

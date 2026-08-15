"""Line-edge roughness: printed edges are not straight.

PURPOSE
-------
Every edge in the layout models is mathematically straight. Real printed edges
are not: photon shot noise in the exposure, the finite size of resist polymer
aggregates and stochastic acid diffusion leave an edge that wanders by a couple
of nanometres about its nominal position, with a characteristic correlation
length along the line. This is *line-edge roughness* (LER), and it is the largest
piece of structural realism the generator was missing.

It matters here for a specific reason. Every other departure from the ideal
layout in this project is a property of the *capture* -- beam spot, dose, noise,
charging -- so the reference and the wide-search image get independent draws.
Roughness is a property of the *specimen*. Both captures look at the same rough
edges, so the roughness is correlated between them, and a localiser can in
principle use it as signal rather than suffering it as noise. Modelling it as
capture noise would have been physically wrong and would have made the problem
look harder than it is.

HOW IT IS MODELLED
------------------
As a displacement field applied to the rendered coverage, not as a redrawing of
the geometry. Displacing a coverage field moves its edges; the interiors are
uniform, so moving them is invisible. The two are equivalent, and this way the
layout models stay exactly separable and exactly area-integrated.

The field is a sum of a few sinusoids with seeded random wavelengths, directions
and phases:

    d(x, y) = A * sum_k  a_k * sin(2*pi*(x/lx_k + y/ly_k) + phi_k)

evaluated at *world* coordinates in nanometres. Being an analytic function of
world position rather than a sampled noise array, it returns the same
displacement for the same physical point at any pixel size -- which is what keeps
the 1 nm/px reference and the 10 nm/px search image describing the same specimen.
A sampled noise field would have had to be interpolated differently at the two
scales and would have broken that.

ANISOTROPY
----------
Real LER is correlated *along* a line -- typical correlation lengths are tens of
nanometres -- and essentially independent *between* neighbouring lines, which are
separate stochastic events. An isotropic field would make adjacent lines wander
together, which reads as stage drift rather than roughness and is much easier to
correlate against.

So the two displacement components are drawn with different correlation lengths
on their two axes. ``dx`` displaces vertical lines: it varies slowly along y
(``along_length_nm``) and quickly along x (``across_lines_nm``), so neighbouring
vertical lines decorrelate. ``dy`` is the transpose of that.

EFFECT ON CROSS-SCALE EXACTNESS
-------------------------------
The rest of the generator renders a window at any pixel size to within 2e-05 grey
levels of any other. Roughness weakens that, and it should: warping an
area-averaged image is not the same as area-averaging a warped one, and the
difference is exactly the fine roughness detail that a 10 nm pixel cannot
resolve. That is the physical truth -- a coarse capture genuinely does not see
nanometre-scale edge wander -- rather than a numerical defect. The measured
residual is reported in the README and is checked by a test, so the number cannot
drift.

SOURCES
-------
See CITATIONS.md section 2.5. The amplitude and correlation-length defaults sit
in the range published for production lithography; 3-sigma LER of a few
nanometres with a correlation length of tens of nanometres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .layouts.base import LayoutRender, RenderWindow
from .resample import bilinear_sample

__all__ = ["RoughnessParams", "displacement_fields", "apply_roughness"]


@dataclass(frozen=True)
class RoughnessParams:
    """Line-edge roughness settings. All lengths in nanometres."""

    #: Standard deviation of edge displacement. Published 3-sigma LER for
    #: production lithography is a few nanometres, so a sigma near 1.2 nm sits in
    #: the middle of that range.
    amplitude_nm: float = 1.2
    #: Correlation length *along* a line. Tens of nanometres in the literature.
    along_length_nm: float = 40.0
    #: Correlation length *across* lines. Short, so neighbouring lines are
    #: independent -- see the module docstring on why isotropy would be wrong.
    across_lines_nm: float = 12.0
    #: Number of sinusoidal components. Enough for the sum to look like noise
    #: rather than a beat pattern; more costs time and buys nothing.
    n_components: int = 8
    #: Seeds the field. A property of the specimen, so the reference and search
    #: captures of one pair must share it.
    seed: int = 0

    def __post_init__(self) -> None:
        if self.amplitude_nm < 0.0:
            raise ValueError(f"amplitude_nm must be non-negative, got {self.amplitude_nm}")
        if self.along_length_nm <= 0.0 or self.across_lines_nm <= 0.0:
            raise ValueError("correlation lengths must be positive")
        if self.n_components < 1:
            raise ValueError(f"n_components must be at least 1, got {self.n_components}")

    @property
    def enabled(self) -> bool:
        return self.amplitude_nm > 0.0

    def as_dict(self) -> Dict[str, object]:
        return {
            "amplitude_nm": self.amplitude_nm,
            "along_length_nm": self.along_length_nm,
            "across_lines_nm": self.across_lines_nm,
            "n_components": self.n_components,
            "seed": self.seed,
        }


def _component_field(
    xs_nm: np.ndarray,
    ys_nm: np.ndarray,
    lambda_x_nm: float,
    lambda_y_nm: float,
    n_components: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Band-limited field with the given correlation lengths per axis.

    Wavelengths are jittered around the nominal by a factor in [0.6, 1.7] so the
    sum does not degenerate into a regular beat pattern.
    """
    total = np.zeros((ys_nm.size, xs_nm.size), dtype=np.float64)
    for _ in range(n_components):
        lx = lambda_x_nm * float(rng.uniform(0.6, 1.7))
        ly = lambda_y_nm * float(rng.uniform(0.6, 1.7))
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        sign = 1.0 if rng.random() < 0.5 else -1.0
        # sin(a + b) expanded so the two axes stay separable and the whole
        # component costs two 1-D evaluations plus one outer product.
        ax = 2.0 * math.pi * xs_nm / lx
        by = sign * 2.0 * math.pi * ys_nm / ly + phase
        total += np.sin(ax)[None, :] * np.cos(by)[:, None]
        total += np.cos(ax)[None, :] * np.sin(by)[:, None]
    # sin(ax)cos(by) + cos(ax)sin(by) is sin(ax + by): one unit-amplitude
    # sinusoid, so variance 1/2. n of them in random phase give variance n/2, and
    # dividing by sqrt(n/2) leaves the field at unit standard deviation -- which
    # is what makes ``amplitude_nm`` mean the sigma it says it means.
    return total / math.sqrt(max(n_components, 1) / 2.0)


def displacement_fields(
    window: RenderWindow, params: RoughnessParams
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-pixel edge displacement ``(dx_px, dy_px)`` for ``window``.

    Returned in *pixels* of the requested window, converted from the nanometre
    amplitude, because that is what the resampler consumes.
    """
    n, px = window.size_px, window.pixel_size_nm
    xs = window.origin_x_nm + (np.arange(n, dtype=np.float64) + 0.5) * px
    ys = window.origin_y_nm + (np.arange(n, dtype=np.float64) + 0.5) * px

    if not params.enabled:
        zero = np.zeros((n, n), dtype=np.float64)
        return zero, zero.copy()

    # Separate streams so dx and dy are independent, and both are a pure function
    # of the specimen seed -- never of the window, the pixel size or draw order.
    rng_x = np.random.default_rng((int(params.seed) & 0xFFFFFFFF, 0xA1))
    rng_y = np.random.default_rng((int(params.seed) & 0xFFFFFFFF, 0xB2))

    # dx displaces vertical lines: slow along y, fast along x.
    dx = _component_field(
        xs, ys, params.across_lines_nm, params.along_length_nm, params.n_components, rng_x
    )
    # dy displaces horizontal lines: slow along x, fast along y.
    dy = _component_field(
        xs, ys, params.along_length_nm, params.across_lines_nm, params.n_components, rng_y
    )
    scale = params.amplitude_nm / px
    return dx * scale, dy * scale


def apply_roughness(
    render: LayoutRender, window: RenderWindow, params: Optional[RoughnessParams]
) -> LayoutRender:
    """Return ``render`` with its coverage fields displaced by edge roughness.

    Every layer is warped with the *same* sampling coordinates. Bilinear weights
    sum to one, so the warped layers sum to the warp of the original sum: the
    disjointness invariant survives exactly, and the composite stays an area
    average rather than becoming an approximation to one.
    """
    if params is None or not params.enabled:
        return render

    dx, dy = displacement_fields(window, params)
    rows, cols = render.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(cols, dtype=np.float64), np.arange(rows, dtype=np.float64)
    )
    sample_x = grid_x + dx[:rows, :cols]
    sample_y = grid_y + dy[:rows, :cols]

    warped = {
        name: bilinear_sample(coverage, sample_x, sample_y).astype(np.float32)
        for name, coverage in render.layers.items()
    }
    metadata = dict(render.metadata)
    metadata["roughness"] = params.as_dict()
    return LayoutRender(
        layers=warped,
        intensities=dict(render.intensities),
        background=render.background,
        metadata=metadata,
    )

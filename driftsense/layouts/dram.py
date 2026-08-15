"""DRAM-style die layout: word lines, bit lines, cell contacts, mat structure.

STRUCTURAL MODEL
----------------
A DRAM array is organised hierarchically. At the bottom is the cell array: word
lines running horizontally (gating the access transistors) crossed by bit lines
running vertically, with a storage-node contact at every crossing. The array is
not continuous across the die -- it is broken into *mats* (sub-arrays), separated
by strips of periphery carrying sense amplifiers and sub-word-line drivers.

Cell dimensions are expressed in the lithographic feature size ``F`` (the
half-pitch). The standard cell architectures are named for their area in units of
``F^2``:

    6F^2 : cell is 2F wide by 3F tall  -> bit-line pitch 2F, word-line pitch 3F
    8F^2 : cell is 2F wide by 4F tall  -> bit-line pitch 2F, word-line pitch 4F

Both are supported; ``6F2`` is the default as it is the dominant modern
arrangement. Explicit pitch overrides are available for cases where the cell
architecture is not the thing being varied.

Default ``mat_size_nm = 2600`` and ``strip_width_nm = 320`` are taken from the
metadata block released alongside the organisers' sample image pair, so that
generated data sits in the same structural regime as the evaluation set. At a
10 um field of view this produces roughly 3.4 mats per axis, which matches the
mat count visible in the released sample search image.

WHY THIS IS SEPARABLE (and therefore exact and cheap to render)
---------------------------------------------------------------
Stacking order is array field < word line < bit line < contact. Resolving
occlusion first, so that each layer reports only the material actually *visible*,
leaves four mutually exclusive regions -- and each is still a product set of a 1D
set in x and a 1D set in y:

    contact      =  C_x               x  C_y
    bit line     =  (BL_x x MatY)     \  contact
    word line    =  (MatX \ BL_x)     x  (WL_y and MatY)
    array field  =  (MatX \ BL_x)     x  (MatY \ WL_y)

so each 2D coverage field is an outer product of two 1D coverage vectors (see
``driftsense.raster``), and the greyscale composite is a plain area-weighted sum
that is exactly the area average over the pixel footprint.

Note that the 1D set algebra -- clipping each mat's lattice to that mat -- happens
*before* integration to coverage. Intersecting afterwards, by multiplying
coverages, would be wrong at the mat edges.

PER-MAT VARIATION
-----------------
If every mat is rendered identically, a reference window taken from one mat is
*pixel-identical* to the same window taken from any other. The correlation surface
then has several exactly-tied maxima and no algorithm can recover which one was
intended -- the pair is unsolvable by construction rather than merely hard.
Measured on the first an earlier revision dataset, the ground truth was not the global
correlation maximum on 8 of 12 pairs, and on one pair six locations tied at
exactly 1.0000.

Mats therefore carry a per-mat cell pitch, drawn deterministically from the mat
index. This is both what makes the problem well posed and what the organisers'
own sample search image shows: its mats visibly differ in pitch and contrast
rather than being carbon copies.

Crucially the variation is applied per mat *column* for bit lines and per mat
*row* for word lines, which keeps every layer a product set and preserves the
exact separable rendering. Setting ``mat_pitch_variation`` to zero restores the
perfectly periodic -- and genuinely unsolvable -- layout, which is retained
deliberately as the ambiguity case for failure analysis.

Note what this does *not* fix. Inside a single mat the array remains perfectly
periodic, so a window lying wholly within one mat is still ambiguous up to
translation by one cell pitch. That is a true property of the problem, not an
artefact: it is why the problem statement specifies a nearest-to-centre tiebreak,
and why the generator labels each pair with whether its reference window contains
a mat boundary.

Structural realism deferred to later stages (deliberately, to keep the geometry
layer free of physics): corner rounding of contacts, line-edge roughness, and
periphery circuitry detail inside the strips.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..raster import outer_coverage, periodic_stripe_coverage
from .base import LayoutModel, LayoutRender, RenderWindow, periodic_block_edges

__all__ = ["DramParams", "DramLayout", "CELL_ARCHITECTURES"]

#: Word-line pitch as a multiple of F, keyed by cell architecture name.
CELL_ARCHITECTURES: Dict[str, float] = {"6F2": 3.0, "8F2": 4.0}


def mat_jitter(seed: int, axis: str, index: int) -> float:
    """Deterministic value in ``[-1, 1]`` keyed on a mat index.

    Used to give each mat column and each mat row its own cell pitch. Being a
    pure function of ``(seed, axis, index)`` rather than of draw order means the
    same mat renders identically no matter which window is requested, which is
    what keeps the reference and search captures consistent.
    """
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")
    key = (
        int(seed) & 0xFFFFFFFF,
        0 if axis == "x" else 1,
        (int(index) + (1 << 20)) & 0xFFFFFFFF,
    )
    return float(np.random.default_rng(key).uniform(-1.0, 1.0))


@dataclass(frozen=True)
class DramParams:
    """Structural parameters of a DRAM-style layout. All lengths in nanometres.

    Every field is exported verbatim into the per-pair metadata, so a failing
    case can always be reproduced exactly.
    """

    # -- cell geometry ------------------------------------------------------
    feature_size_nm: float = 35.0
    cell_architecture: str = "6F2"
    bitline_pitch_nm: float | None = None  # defaults to 2F
    wordline_pitch_nm: float | None = None  # defaults to CELL_ARCHITECTURES[arch] * F
    bitline_width_ratio: float = 0.5  # fraction of bit-line pitch that is metal
    wordline_width_ratio: float = 0.42  # fraction of word-line pitch that is gate
    contact_size_ratio: float = 0.55  # contact size as a fraction of feature size

    # -- etch / lithography bias -------------------------------------------
    linewidth_bias_nm: float = 0.0  # positive grows every drawn feature

    # -- mat (sub-array) structure -----------------------------------------
    mat_size_nm: float = 2600.0
    strip_width_nm: float = 320.0
    mat_phase_x_nm: float = 0.0
    mat_phase_y_nm: float = 0.0

    # -- per-mat variation --------------------------------------------------
    # Fractional spread of cell pitch between mats. Zero reproduces a perfectly
    # periodic layout in which distinct mats are indistinguishable, which makes
    # localisation ill-posed; see the module docstring.
    mat_pitch_variation: float = 0.22
    # Fractional spread of *line width* between mats, independent of pitch.
    #
    # This is critical-dimension non-uniformity across the die, and it is not a
    # duplicate of mat_pitch_variation. Line width is otherwise a fixed ratio of
    # pitch, so the duty cycle -- the fraction of the mat covered by metal -- is
    # identical in every mat however much the pitch varies. Duty cycle is the
    # only thing that survives heavy blur, so under an optical microscope, where
    # the cell grating is entirely unresolved, mats with pitch variation alone
    # are indistinguishable and localisation is ill-posed. Measured on the
    # optical path before this existed: the true location was the global
    # correlation maximum on 0 of 10 pairs.
    mat_width_variation: float = 0.10
    signature_seed: int = 0

    # -- flat greyscale levels (an earlier revision only; an earlier revision replaces with SE yield)
    intensity_substrate: float = 0.45
    intensity_array_field: float = 0.38
    intensity_wordline: float = 0.60
    intensity_bitline: float = 0.66
    intensity_contact: float = 0.90

    def __post_init__(self) -> None:
        if self.feature_size_nm <= 0:
            raise ValueError("feature_size_nm must be positive")
        if self.cell_architecture not in CELL_ARCHITECTURES:
            raise ValueError(
                f"unknown cell_architecture {self.cell_architecture!r}; "
                f"expected one of {sorted(CELL_ARCHITECTURES)}"
            )
        if self.mat_size_nm <= 0:
            raise ValueError("mat_size_nm must be positive")
        if self.strip_width_nm < 0:
            raise ValueError("strip_width_nm must be non-negative")
        if not 0.0 <= self.mat_pitch_variation < 1.0:
            raise ValueError("mat_pitch_variation must lie in [0, 1)")
        if not 0.0 <= self.mat_width_variation < 1.0:
            raise ValueError("mat_width_variation must lie in [0, 1)")
        if self.resolved_bitline_pitch_nm <= 0 or self.resolved_wordline_pitch_nm <= 0:
            raise ValueError("resolved pitches must be positive")

    # -- resolved geometry --------------------------------------------------

    @property
    def resolved_bitline_pitch_nm(self) -> float:
        if self.bitline_pitch_nm is not None:
            return float(self.bitline_pitch_nm)
        return 2.0 * self.feature_size_nm

    @property
    def resolved_wordline_pitch_nm(self) -> float:
        if self.wordline_pitch_nm is not None:
            return float(self.wordline_pitch_nm)
        return CELL_ARCHITECTURES[self.cell_architecture] * self.feature_size_nm

    @property
    def mat_period_nm(self) -> float:
        return self.mat_size_nm + self.strip_width_nm

    def _biased(self, width: float, pitch: float) -> float:
        """Apply etch bias and clamp so a feature can never swallow its pitch."""
        return float(np.clip(width + self.linewidth_bias_nm, 0.0, pitch))

    @property
    def bitline_width_nm(self) -> float:
        """Nominal bit-line width. Actual width varies per mat column."""
        pitch = self.resolved_bitline_pitch_nm
        return self._biased(self.bitline_width_ratio * pitch, pitch)

    @property
    def wordline_width_nm(self) -> float:
        """Nominal word-line width. Actual width varies per mat row."""
        pitch = self.resolved_wordline_pitch_nm
        return self._biased(self.wordline_width_ratio * pitch, pitch)

    @property
    def contact_size_nm(self) -> float:
        nominal = self.contact_size_ratio * self.feature_size_nm
        smallest_pitch = min(self.resolved_bitline_pitch_nm, self.resolved_wordline_pitch_nm)
        return self._biased(nominal, smallest_pitch)

    def mat_pitch_nm(self, axis: str, index: int) -> float:
        """Cell pitch of the mat at ``index`` along ``axis``."""
        base = self.resolved_bitline_pitch_nm if axis == "x" else self.resolved_wordline_pitch_nm
        jitter = mat_jitter(self.signature_seed, axis, index)
        return float(base * (1.0 + self.mat_pitch_variation * jitter))

    def mat_width_ratio(self, axis: str, index: int) -> float:
        """Line-width-to-pitch ratio of the mat at ``index``.

        Drawn from a stream offset from the pitch jitter so the two are
        independent: a mat with a wide pitch is not thereby given a wide line.
        """
        base = self.bitline_width_ratio if axis == "x" else self.wordline_width_ratio
        jitter = mat_jitter(self.signature_seed + 0x5EED, axis, index)
        return float(np.clip(base * (1.0 + self.mat_width_variation * jitter), 0.02, 0.95))

    def as_dict(self) -> Dict[str, object]:
        return {
            "feature_size_nm": self.feature_size_nm,
            "cell_architecture": self.cell_architecture,
            "bitline_pitch_nm": self.resolved_bitline_pitch_nm,
            "wordline_pitch_nm": self.resolved_wordline_pitch_nm,
            "bitline_width_nm": self.bitline_width_nm,
            "wordline_width_nm": self.wordline_width_nm,
            "contact_size_nm": self.contact_size_nm,
            "linewidth_bias_nm": self.linewidth_bias_nm,
            "mat_size_nm": self.mat_size_nm,
            "strip_width_nm": self.strip_width_nm,
            "mat_phase_x_nm": self.mat_phase_x_nm,
            "mat_phase_y_nm": self.mat_phase_y_nm,
            "mat_pitch_variation": self.mat_pitch_variation,
            "mat_width_variation": self.mat_width_variation,
            "signature_seed": self.signature_seed,
        }


class DramLayout(LayoutModel):
    """Procedural DRAM-style layout renderable at any pixel size."""

    name = "dram"

    #: Stacking order, lowest first. Used only for documentation and for the
    #: occlusion arithmetic in ``render``; the rendered layers are disjoint.
    LAYER_ORDER: List[str] = ["array_field", "wordline", "bitline", "contact"]

    def __init__(self, params: DramParams | None = None):
        self.params = params or DramParams()

    # -- rendering ----------------------------------------------------------

    def _axis_coverages(
        self, size_px: int, pixel_size_nm: float, origin_nm: float, axis: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Exact per-pixel coverage for one axis: ``(lines, contacts, mat_region)``.

        Each mat contributing to the window is rendered with its own cell pitch,
        with the lattice phased from that mat's leading edge -- which is the
        physical arrangement, since an array starts at its boundary rather than at
        an arbitrary global origin.

        Contributions from different mats are clipped to disjoint intervals, so
        summing their coverages is exact rather than an approximation.
        """
        p = self.params
        phase = p.mat_phase_x_nm if axis == "x" else p.mat_phase_y_nm
        width_ratio = p.bitline_width_ratio if axis == "x" else p.wordline_width_ratio
        period, size = p.mat_period_nm, p.mat_size_nm

        mat = periodic_stripe_coverage(
            size_px, pixel_size_nm, origin_nm, period, size, phase
        )
        lines = np.zeros(size_px, dtype=np.float32)
        contacts = np.zeros(size_px, dtype=np.float32)

        window_lo = origin_nm
        window_hi = origin_nm + size_px * pixel_size_nm
        first = int(np.floor((window_lo - phase) / period))
        last = int(np.floor((window_hi - phase) / period))
        for index in range(first, last + 1):
            start = index * period + phase
            end = start + size
            if end <= window_lo or start >= window_hi:
                continue

            pitch = p.mat_pitch_nm(axis, index)
            ratio = p.mat_width_ratio(axis, index)
            line_width = float(np.clip(ratio * pitch + p.linewidth_bias_nm, 0.0, pitch))
            # The contact lands *on* the line, so it can never be wider than the
            # line it sits on. Clamping here is what guarantees the containment
            # the occlusion arithmetic in ``render`` relies on; without it a
            # narrow line under a wide contact would drive the visible-coverage
            # subtraction negative.
            contact = float(
                np.clip(
                    p.contact_size_ratio * p.feature_size_nm + p.linewidth_bias_nm,
                    0.0,
                    line_width,
                )
            )

            lines += periodic_stripe_coverage(
                size_px, pixel_size_nm, origin_nm, pitch, line_width, start, start, end
            )
            contacts += periodic_stripe_coverage(
                size_px,
                pixel_size_nm,
                origin_nm,
                pitch,
                contact,
                start + (line_width - contact) / 2.0,
                start,
                end,
            )

        return lines, contacts, mat

    def render(self, window: RenderWindow) -> LayoutRender:
        p = self.params
        n, px = window.size_px, window.pixel_size_nm

        # The per-axis helper clips each mat's lattice to that mat before
        # integrating, so the 1D set algebra is already complete and the
        # coverages below need no further intersection.
        cov_bl_x, cov_c_x, cov_mat_x = self._axis_coverages(n, px, window.origin_x_nm, "x")
        cov_wl_y, cov_c_y, cov_mat_y = self._axis_coverages(n, px, window.origin_y_nm, "y")

        # --- occlusion, resolved exactly ----------------------------------
        # Stacking order is array field < word line < bit line < contact. Each
        # layer below reports only the part of the pixel from which it is
        # actually *visible*, so the four are disjoint and a plain weighted sum
        # reproduces the area average exactly.
        #
        # Every region below is still a product set (or a difference of two),
        # which is why the outer-product identity continues to hold:
        #
        #   contact     = C_x            x  C_y
        #   bit line    = (BL_x x MatY)  \  contact
        #   word line   = (MatX \ BL_x)  x  (WL_y and MatY)
        #   array field = (MatX \ BL_x)  x  (MatY \ WL_y)
        #
        # Containment C_x subset BL_x and C_y subset WL_y is guaranteed by the
        # contact clamp in ``_axis_indicators``, so no term can go negative.
        exposed_mat_x = cov_mat_x - cov_bl_x
        exposed_mat_y = cov_mat_y - cov_wl_y

        contact = outer_coverage(cov_c_y, cov_c_x)
        layers = {
            "array_field": outer_coverage(exposed_mat_y, exposed_mat_x),
            "wordline": outer_coverage(cov_wl_y, exposed_mat_x),
            "bitline": outer_coverage(cov_mat_y, cov_bl_x) - contact,
            "contact": contact,
        }

        intensities = {
            "array_field": p.intensity_array_field,
            "wordline": p.intensity_wordline,
            "bitline": p.intensity_bitline,
            "contact": p.intensity_contact,
        }

        return LayoutRender(
            layers=layers,
            intensities=intensities,
            background=p.intensity_substrate,
            metadata={"architecture": self.name, "window": window.bounds_nm},
        )

    # -- introspection ------------------------------------------------------

    def describe(self) -> Dict[str, object]:
        return {"architecture": self.name, **self.params.as_dict()}

    def boundary_coordinates_nm(self, axis: str, lo_nm: float, hi_nm: float) -> Sequence[float]:
        """Mat edges within ``[lo_nm, hi_nm]`` -- the layout's uniqueness anchors."""
        if axis not in ("x", "y"):
            raise ValueError("axis must be 'x' or 'y'")
        p = self.params
        return periodic_block_edges(
            lo_nm,
            hi_nm,
            p.mat_period_nm,
            p.mat_size_nm,
            p.mat_phase_x_nm if axis == "x" else p.mat_phase_y_nm,
        )

    def with_params(self, **changes) -> "DramLayout":
        """Return a copy of this layout with selected parameters replaced."""
        return DramLayout(replace(self.params, **changes))

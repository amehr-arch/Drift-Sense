"""Randomised sampling of layout parameters and reference-crop placements.

TWO KINDS OF DIVERSITY
----------------------
1. *Structural* diversity -- each pair uses a different feature size, cell
   architecture, mat size and lattice phase, so the locator can never latch onto
   one particular pitch.
2. *Positional* diversity -- where in the search field the reference was taken
   from, which controls how hard the pair is.

The positional axis is the one that matters most. A reference window that
contains a mat boundary carries an unambiguous anchor and is easy; a window taken
from deep inside a uniform array is genuinely ambiguous and is exactly the
failure mode the organisers say they will test. ``boundary_bias`` is the
probability that a placement is deliberately steered onto a boundary; the
remainder fall wherever they land, which for the default mat geometry is
predominantly array interior.

The organisers' sample metadata carries ``boundary_bias = 0.35``, adopted here as
the default. Their exact semantics are not published; the interpretation used
here (probability of steering the crop onto a structural boundary) is stated
explicitly so that it can be revised if a definition is released.

DETERMINISM
-----------
Every sampler takes an explicit ``numpy.random.Generator``. Given the same seed
the entire dataset -- parameters, placements and pixels -- is bit-for-bit
reproducible, which is asserted in the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np

from .geometry import ImagingGeometry, Placement
from .imaging import CaptureParams
from .layouts import DramLayout, DramParams, FinfetLayout, FinfetParams, LayoutModel

__all__ = [
    "DramParamRanges",
    "sample_dram_layout",
    "FinfetParamRanges",
    "sample_finfet_layout",
    "PlacementSampler",
    "CaptureRanges",
    "sample_captures",
]


@dataclass(frozen=True)
class DramParamRanges:
    """Inclusive sampling ranges for DRAM structural parameters.

    Defaults span a deliberately wide regime -- roughly a 2x spread in feature
    size and a 1.6x spread in mat size. Over-narrow generator ranges are the
    standard route to a locator that scores well on its own data and collapses on
    the organisers' test set, so the ranges here are wider than strictly needed.
    """

    feature_size_nm: Tuple[float, float] = (24.0, 48.0)
    cell_architectures: Sequence[str] = ("6F2", "8F2")
    bitline_width_ratio: Tuple[float, float] = (0.40, 0.58)
    wordline_width_ratio: Tuple[float, float] = (0.34, 0.50)
    contact_size_ratio: Tuple[float, float] = (0.45, 0.70)
    linewidth_bias_nm: Tuple[float, float] = (-2.0, 2.0)
    mat_size_nm: Tuple[float, float] = (2000.0, 3200.0)
    strip_width_nm: Tuple[float, float] = (240.0, 420.0)
    intensity_jitter: float = 0.04
    #: Spread of cell pitch between mats. Values near zero make distinct mats
    #: indistinguishable and localisation ill-posed; see driftsense.layouts.dram.
    mat_pitch_variation: Tuple[float, float] = (0.15, 0.30)

    def __post_init__(self) -> None:
        for name in (
            "feature_size_nm",
            "bitline_width_ratio",
            "wordline_width_ratio",
            "contact_size_ratio",
            "linewidth_bias_nm",
            "mat_size_nm",
            "strip_width_nm",
            "mat_pitch_variation",
        ):
            lo, hi = getattr(self, name)
            if lo > hi:
                raise ValueError(f"{name} range is inverted: ({lo}, {hi})")
        if not self.cell_architectures:
            raise ValueError("cell_architectures must not be empty")


def sample_dram_layout(
    rng: np.random.Generator,
    ranges: DramParamRanges | None = None,
    geometry: ImagingGeometry | None = None,
) -> DramLayout:
    """Draw one randomised DRAM layout.

    Lattice and mat phases are drawn over a full period so that no absolute
    position in the world is privileged; without this, every generated pair would
    share the same alignment to the pixel grid and the locator would be tuned to
    an artefact of the generator.
    """
    ranges = ranges or DramParamRanges()
    geometry = geometry or ImagingGeometry()

    def uniform(bounds: Tuple[float, float]) -> float:
        return float(rng.uniform(bounds[0], bounds[1]))

    feature = uniform(ranges.feature_size_nm)
    architecture = str(rng.choice(list(ranges.cell_architectures)))
    mat_size = uniform(ranges.mat_size_nm)
    strip = uniform(ranges.strip_width_nm)
    mat_period = mat_size + strip

    jitter = ranges.intensity_jitter

    def shade(value: float) -> float:
        return float(np.clip(value + rng.uniform(-jitter, jitter), 0.02, 0.98))

    params = DramParams(
        feature_size_nm=feature,
        cell_architecture=architecture,
        bitline_width_ratio=uniform(ranges.bitline_width_ratio),
        wordline_width_ratio=uniform(ranges.wordline_width_ratio),
        contact_size_ratio=uniform(ranges.contact_size_ratio),
        linewidth_bias_nm=uniform(ranges.linewidth_bias_nm),
        mat_size_nm=mat_size,
        strip_width_nm=strip,
        mat_phase_x_nm=float(rng.uniform(0.0, mat_period)),
        mat_phase_y_nm=float(rng.uniform(0.0, mat_period)),
        mat_pitch_variation=uniform(ranges.mat_pitch_variation),
        signature_seed=int(rng.integers(0, 2**31 - 1)),
        intensity_substrate=shade(0.45),
        intensity_array_field=shade(0.38),
        intensity_wordline=shade(0.60),
        intensity_bitline=shade(0.66),
        intensity_contact=shade(0.90),
    )
    return DramLayout(params)


@dataclass(frozen=True)
class FinfetParamRanges:
    """Inclusive sampling ranges for FinFET structural parameters.

    Wide for the same reason the DRAM ranges are wide: a generator that samples
    narrowly produces a locator tuned to its own data.

    The fin-pitch range starts at 32 nm rather than the 24 nm published ground
    rules reach. At the 10 nm wide-search pixel size a 24 nm pitch is 2.4 pixels,
    under Nyquist, so the grating is not sampled at all -- such a pair is
    unsolvable for reasons of optics rather than of algorithm. Generating them by
    default would depress the accuracy figure while telling nobody anything. They
    remain one parameter away; see ``driftsense.layouts.finfet``.
    """

    fin_pitch_nm: Tuple[float, float] = (32.0, 54.0)
    #: Fin width as a fraction of fin pitch.
    fin_width_ratio: Tuple[float, float] = (0.22, 0.36)
    #: Gate pitch as a multiple of fin pitch, the standard-cell ratio.
    gate_pitch_ratio: Tuple[float, float] = (2.4, 3.6)
    #: Gate length as a fraction of gate pitch.
    gate_length_ratio: Tuple[float, float] = (0.22, 0.34)
    #: Contact extent as a fraction of gate pitch, before clamping.
    contact_length_ratio: Tuple[float, float] = (0.30, 0.44)
    linewidth_bias_nm: Tuple[float, float] = (-2.0, 2.0)
    block_size_nm: Tuple[float, float] = (1400.0, 2400.0)
    break_width_nm: Tuple[float, float] = (180.0, 340.0)
    intensity_jitter: float = 0.04
    #: Spread of grating pitch between blocks. Near zero makes distinct blocks
    #: indistinguishable and localisation ill-posed.
    block_pitch_variation: Tuple[float, float] = (0.12, 0.26)

    def __post_init__(self) -> None:
        for name in (
            "fin_pitch_nm",
            "fin_width_ratio",
            "gate_pitch_ratio",
            "gate_length_ratio",
            "contact_length_ratio",
            "linewidth_bias_nm",
            "block_size_nm",
            "break_width_nm",
            "block_pitch_variation",
        ):
            lo, hi = getattr(self, name)
            if lo > hi:
                raise ValueError(f"{name} range is inverted: ({lo}, {hi})")


def sample_finfet_layout(
    rng: np.random.Generator,
    ranges: "FinfetParamRanges | None" = None,
    geometry: ImagingGeometry | None = None,
) -> FinfetLayout:
    """Draw one randomised FinFET layout.

    Block phases are drawn over a full period for the same reason the DRAM
    sampler draws mat phases: otherwise every pair shares one alignment to the
    pixel grid and the locator learns an artefact of the generator.
    """
    ranges = ranges or FinfetParamRanges()
    geometry = geometry or ImagingGeometry()

    def uniform(bounds: Tuple[float, float]) -> float:
        return float(rng.uniform(bounds[0], bounds[1]))

    fin_pitch = uniform(ranges.fin_pitch_nm)
    gate_pitch = fin_pitch * uniform(ranges.gate_pitch_ratio)
    block_size = uniform(ranges.block_size_nm)
    break_width = uniform(ranges.break_width_nm)
    block_period = block_size + break_width

    jitter = ranges.intensity_jitter

    def shade(value: float) -> float:
        return float(np.clip(value + rng.uniform(-jitter, jitter), 0.02, 0.98))

    params = FinfetParams(
        fin_pitch_nm=fin_pitch,
        fin_width_nm=fin_pitch * uniform(ranges.fin_width_ratio),
        gate_pitch_nm=gate_pitch,
        gate_length_nm=gate_pitch * uniform(ranges.gate_length_ratio),
        contact_length_nm=gate_pitch * uniform(ranges.contact_length_ratio),
        linewidth_bias_nm=uniform(ranges.linewidth_bias_nm),
        block_size_nm=block_size,
        break_width_nm=break_width,
        block_phase_x_nm=float(rng.uniform(0.0, block_period)),
        block_phase_y_nm=float(rng.uniform(0.0, block_period)),
        block_pitch_variation=uniform(ranges.block_pitch_variation),
        signature_seed=int(rng.integers(0, 2**31 - 1)),
        intensity_substrate=shade(0.42),
        intensity_field=shade(0.34),
        intensity_fin=shade(0.58),
        intensity_contact=shade(0.78),
        intensity_gate=shade(0.88),
    )
    return FinfetLayout(params)



@dataclass
class PlacementSampler:
    """Chooses where in the search field the reference crop is taken from.

    Parameters
    ----------
    geometry:
        Imaging geometry, used for the valid origin range.
    boundary_bias:
        Probability, per axis, of steering the crop so that a structural
        boundary falls inside the reference window.
    subpixel:
        When ``False`` (the default) origins are snapped to whole nanometres,
        which makes the ground truth land on exact multiples of 0.1 search
        pixels. This matches the organisers' released sample, whose origins are
        integral in nanometres. Setting ``True`` produces arbitrary sub-pixel
        offsets and is the harder regime used to stress sub-pixel refinement.
    interior_margin:
        Fraction of the reference window kept clear of the boundary when
        steering, so the anchor never sits exactly on the window edge where it
        would be half cut off.
    """

    geometry: ImagingGeometry = field(default_factory=ImagingGeometry)
    boundary_bias: float = 0.35
    subpixel: bool = False
    interior_margin: float = 0.2

    def __post_init__(self) -> None:
        if not 0.0 <= self.boundary_bias <= 1.0:
            raise ValueError("boundary_bias must lie in [0, 1]")
        if not 0.0 <= self.interior_margin < 0.5:
            raise ValueError("interior_margin must lie in [0, 0.5)")

    def sample(self, layout: LayoutModel, rng: np.random.Generator) -> Placement:
        """Draw one placement, returning the world origin of the crop."""
        return Placement(
            origin_x_nm=self._sample_axis(layout, rng, "x"),
            origin_y_nm=self._sample_axis(layout, rng, "y"),
        )

    # -- internals ----------------------------------------------------------

    def _sample_axis(self, layout: LayoutModel, rng: np.random.Generator, axis: str) -> float:
        geom = self.geometry
        limit = geom.max_origin_nm
        origin: float | None = None

        if rng.random() < self.boundary_bias:
            origin = self._steer_to_boundary(layout, rng, axis, limit)
        if origin is None:
            origin = float(rng.uniform(0.0, limit))

        origin = float(np.clip(origin, 0.0, limit))
        if not self.subpixel:
            origin = float(np.clip(np.rint(origin), 0.0, np.floor(limit)))
        return origin

    def _steer_to_boundary(
        self, layout: LayoutModel, rng: np.random.Generator, axis: str, limit: float
    ) -> float | None:
        """Place the window so a structural boundary lands inside it, or give up."""
        fov = self.geometry.reference_fov_nm
        candidates = [
            edge
            for edge in layout.boundary_coordinates_nm(axis, 0.0, self.geometry.search_fov_nm)
            # Reachable only if some legal origin puts the edge inside the window.
            if edge - fov * (1.0 - self.interior_margin) <= limit
            and edge >= fov * self.interior_margin
        ]
        if not candidates:
            return None
        edge = float(rng.choice(candidates))
        offset = float(rng.uniform(self.interior_margin, 1.0 - self.interior_margin)) * fov
        return edge - offset


@dataclass(frozen=True)
class CaptureRanges:
    """Sampling ranges for acquisition settings, per capture.

    The reference and wide-search captures are drawn from separate ranges because
    they are physically different acquisitions: the wide-field image uses a larger
    spot and roughly a tenth of the dose, which is what makes it softer and
    noisier. Those two figures follow the organisers' released sample metadata.

    Ranges are deliberately wider than the nominal operating point. A locator
    tuned to one noise level is a locator that fails on the evaluation set, whose
    stated design is to be noisier than anything participants train on.
    """

    # -- shared -------------------------------------------------------------
    edge_gain: Tuple[float, float] = (0.35, 0.80)
    astigmatism_ratio: Tuple[float, float] = (0.9, 1.15)
    vignette_strength: Tuple[float, float] = (0.0, 0.18)
    gamma: Tuple[float, float] = (0.85, 1.20)

    # -- reference capture --------------------------------------------------
    reference_spot_nm: Tuple[float, float] = (3.0, 8.0)
    reference_dose: Tuple[float, float] = (1200.0, 3000.0)
    reference_detector_sigma: Tuple[float, float] = (1.0, 3.0)

    # -- wide-search capture ------------------------------------------------
    search_spot_nm: Tuple[float, float] = (8.0, 18.0)
    search_dose: Tuple[float, float] = (90.0, 320.0)
    search_detector_sigma: Tuple[float, float] = (3.0, 8.0)
    search_charging_prob: Tuple[float, float] = (0.0, 0.02)
    search_charging_intensity: Tuple[float, float] = (0.0, 0.25)

    # -- inter-visit stage error (applied to the reference capture) ---------
    rotation_deg: Tuple[float, float] = (-1.0, 1.0)
    scale_error: Tuple[float, float] = (-0.01, 0.01)
    shear_px: Tuple[float, float] = (-1.5, 1.5)
    drift_jitter_px: Tuple[float, float] = (0.0, 0.5)

    #: Fraction of the nominal geometric error actually applied. an earlier revision held this
    #: at 0.25 because the locator searched neither rotation nor scale, so
    #: full stage error would have measured nothing but that gap. a later revision added
    #: both searches, so it is now at full strength -- which is what the problem
    #: statement's 1-3 degree figure actually describes.
    geometric_scale: float = 1.0


def sample_captures(
    rng: np.random.Generator, ranges: CaptureRanges | None = None
) -> Tuple[CaptureParams, CaptureParams]:
    """Draw acquisition settings for one reference/search pair."""
    ranges = ranges or CaptureRanges()

    def uniform(bounds: Tuple[float, float]) -> float:
        return float(rng.uniform(bounds[0], bounds[1]))

    geometric = ranges.geometric_scale
    reference = CaptureParams(
        spot_size_nm=uniform(ranges.reference_spot_nm),
        astigmatism_ratio=uniform(ranges.astigmatism_ratio),
        edge_gain=uniform(ranges.edge_gain),
        rotation_deg=uniform(ranges.rotation_deg) * geometric,
        scale_error=uniform(ranges.scale_error) * geometric,
        shear_px=uniform(ranges.shear_px) * geometric,
        drift_jitter_px=uniform(ranges.drift_jitter_px) * geometric,
        dose=uniform(ranges.reference_dose),
        detector_noise_sigma=uniform(ranges.reference_detector_sigma),
        vignette_strength=uniform(ranges.vignette_strength),
        gamma=uniform(ranges.gamma),
    )
    search = CaptureParams(
        spot_size_nm=uniform(ranges.search_spot_nm),
        astigmatism_ratio=uniform(ranges.astigmatism_ratio),
        edge_gain=uniform(ranges.edge_gain),
        charging_streak_prob=uniform(ranges.search_charging_prob),
        charging_streak_intensity=uniform(ranges.search_charging_intensity),
        dose=uniform(ranges.search_dose),
        detector_noise_sigma=uniform(ranges.search_detector_sigma),
        vignette_strength=uniform(ranges.vignette_strength),
        gamma=uniform(ranges.gamma),
    )
    return reference, search

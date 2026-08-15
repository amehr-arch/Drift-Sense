"""SEM image formation: turning a layout render into a simulated micrograph.

THE CHAIN, AND WHY THE ORDER IS WHAT IT IS
------------------------------------------
A scanning electron microscope does not photograph a layout. It rasters a focused
electron beam across the specimen and counts secondary electrons. Each physical
effect enters at a specific point in that process, and applying them in the wrong
order gives the wrong answer -- most obviously, noise added before blur comes out
smoothed, which is not what a detector does.

    1. material contrast     secondary-electron yield per material
    2. edge brightening      yield rises where the beam meets a sidewall
    3. beam point spread     finite spot size, possibly astigmatic
    4. geometric error       inter-visit stage rotation, scale and shear
    5. charging              local field distortion from trapped charge
    6. vignetting            collection efficiency falls off the axis
    7. shot noise            finite dose; Poisson in the electron count
    8. detector noise        additive, from the amplifier chain
    9. gamma                 display transfer
   10. sensor defects        speckle, hot and dead pixels
   11. quantisation          to the 8-bit greyscale the dataset ships as

Steps 1 and 2 concern the specimen. Steps 3 to 6 concern the column and the scan.
Steps 7 onward concern detection and readout, and are therefore the only ones that
may introduce noise.

WHY NO OVERSAMPLING IS NEEDED FOR THE BLUR
------------------------------------------
The physically correct model is: specimen, convolved with the beam PSF, then
integrated over each pixel footprint. The layout renderer already delivers the
specimen integrated over the pixel footprint, so applying the PSF afterwards
computes ``(specimen * box) * psf`` where the correct quantity is
``(specimen * psf) * box``. Convolution commutes, so these are identical. The blur
can be applied at the target resolution with no approximation and no 10x
intermediate raster.

That argument covers the *linear* steps only. Edge brightening is a non-linear
function of the specimen, so integrating it after the fact is an approximation --
see ``edge_density`` for what is actually computed and why it is the right
first-order quantity.

INDEPENDENCE OF THE TWO CAPTURES
--------------------------------
The reference and the wide-search image are separate physical acquisitions, taken
at different times and different magnification. Every stochastic term is drawn
from its own generator, and the dose differs by design -- the wide-field capture
collects roughly a tenth of the electrons per pixel, which is the physical reason
it is noisier rather than an arbitrary decision to add more noise.

GEOMETRIC ERROR IS ATTRIBUTED TO THE REFERENCE
----------------------------------------------
The answer is expressed in search-image coordinates, so the search image defines
the frame. Any relative misalignment between the two captures -- the stage
rotation, scale and shear error that the whole problem is about -- is therefore
applied to the reference capture. This is not a physical claim about which tool
moved; it is the choice that keeps the ground truth exactly computable while
still presenting the locator with the full relative distortion it must overcome.

REFERENCES
----------
See CITATIONS.md at the repository root. Every parameter below is annotated with
the source that justifies its form and its plausible range.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Mapping, Optional

import numpy as np

from .layouts.base import LayoutRender
from .resample import bilinear_sample

__all__ = [
    "MaterialYields",
    "CaptureParams",
    "REFERENCE_CAPTURE",
    "SEARCH_CAPTURE",
    "secondary_electron_signal",
    "edge_density",
    "gaussian_blur",
    "apply_geometric_error",
    "apply_charging",
    "apply_vignette",
    "apply_dose_noise",
    "form_image",
]


#: Relative secondary-electron yield per material, in arbitrary units where the
#: exposed substrate is 1.0. Metals and heavily doped features emit more strongly
#: than dielectric or bare silicon; the ordering matters far more than the exact
#: values, which is why these are exposed as parameters.
DEFAULT_YIELDS: Dict[str, float] = {
    "array_field": 0.86,
    "wordline": 1.32,
    "bitline": 1.46,
    "contact": 1.95,
}


@dataclass(frozen=True)
class MaterialYields:
    """Secondary-electron yield per material, plus the exposed substrate."""

    substrate: float = 1.0
    materials: Mapping[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.materials is None:
            object.__setattr__(self, "materials", dict(DEFAULT_YIELDS))
        if self.substrate <= 0:
            raise ValueError("substrate yield must be positive")
        for name, value in self.materials.items():
            if value < 0:
                raise ValueError(f"yield for {name!r} must be non-negative")

    def for_layer(self, name: str) -> float:
        return float(self.materials.get(name, self.substrate))

    def as_dict(self) -> Dict[str, float]:
        return {"substrate": self.substrate, **dict(self.materials)}


@dataclass(frozen=True)
class CaptureParams:
    """Everything about how one image is acquired.

    Lengths are in nanometres so that a parameter means the same physical thing
    at both magnifications; conversion to pixels happens inside the model using
    the window's pixel size.
    """

    # -- column ------------------------------------------------------------
    #: 1/e beam spot radius. Sets the Gaussian PSF width.
    spot_size_nm: float = 5.0
    #: Ratio of vertical to horizontal PSF width. 1.0 is a well-corrected column.
    astigmatism_ratio: float = 1.0

    # -- specimen response -------------------------------------------------
    #: Strength of the edge-brightening term relative to flat-surface yield.
    edge_gain: float = 0.55
    #: Width of the bright fringe along a feature edge, set by the escape depth
    #: of secondary electrons. Physical, so the fringe occupies a *smaller*
    #: fraction of a wide-field pixel than of a high-magnification one.
    edge_width_nm: float = 4.0

    # -- geometric error (applied to the reference capture only) -----------
    rotation_deg: float = 0.0
    scale_error: float = 0.0  # fractional; 0.01 means the capture is 1% larger
    shear_px: float = 0.0
    drift_jitter_px: float = 0.0  # per-scanline random displacement, std dev

    # -- charging ----------------------------------------------------------
    charging_streak_prob: float = 0.0
    charging_streak_intensity: float = 0.0

    # -- detection ---------------------------------------------------------
    #: Electrons collected per pixel at unit yield. Lower dose, noisier image.
    dose: float = 2000.0
    #: Additive readout noise, in 8-bit grey levels.
    detector_noise_sigma: float = 2.0
    vignette_strength: float = 0.0
    gamma: float = 1.0
    speckle_sigma: float = 0.0
    salt_pepper_prob: float = 0.0

    # -- output ------------------------------------------------------------
    #: Grey level the mean signal is mapped to before quantisation. Keeping this
    #: fixed across captures means a dose change alters noise, not brightness.
    target_level: float = 128.0

    def __post_init__(self) -> None:
        if self.spot_size_nm < 0:
            raise ValueError("spot_size_nm must be non-negative")
        if self.edge_width_nm < 0:
            raise ValueError("edge_width_nm must be non-negative")
        if self.astigmatism_ratio <= 0:
            raise ValueError("astigmatism_ratio must be positive")
        if self.dose <= 0:
            raise ValueError("dose must be positive")
        if self.detector_noise_sigma < 0:
            raise ValueError("detector_noise_sigma must be non-negative")
        if not -0.9 < self.scale_error < 0.9:
            raise ValueError("scale_error must lie in (-0.9, 0.9)")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if not 0.0 <= self.charging_streak_prob <= 1.0:
            raise ValueError("charging_streak_prob must lie in [0, 1]")
        if not 0.0 <= self.salt_pepper_prob <= 1.0:
            raise ValueError("salt_pepper_prob must lie in [0, 1]")

    def as_dict(self) -> Dict[str, float]:
        return {
            "spot_size_nm": self.spot_size_nm,
            "astigmatism_ratio": self.astigmatism_ratio,
            "edge_gain": self.edge_gain,
            "edge_width_nm": self.edge_width_nm,
            "rotation_deg": self.rotation_deg,
            "scale_error": self.scale_error,
            "shear_px": self.shear_px,
            "drift_jitter_px": self.drift_jitter_px,
            "charging_streak_prob": self.charging_streak_prob,
            "charging_streak_intensity": self.charging_streak_intensity,
            "dose": self.dose,
            "detector_noise_sigma": self.detector_noise_sigma,
            "vignette_strength": self.vignette_strength,
            "gamma": self.gamma,
            "speckle_sigma": self.speckle_sigma,
            "salt_pepper_prob": self.salt_pepper_prob,
        }

    def with_changes(self, **changes) -> "CaptureParams":
        return replace(self, **changes)


#: High-magnification capture: small spot, high dose, quiet detector.
REFERENCE_CAPTURE = CaptureParams(
    spot_size_nm=5.0, dose=2000.0, detector_noise_sigma=2.0
)

#: Wide-field capture. The larger spot and the tenfold lower dose are the physical
#: reasons this image is noisier and softer; both figures follow the organisers'
#: released sample metadata (dose 2000 against 200, sigma 2.0 against 5.0).
SEARCH_CAPTURE = CaptureParams(
    spot_size_nm=12.0, dose=200.0, detector_noise_sigma=5.0
)


# ---------------------------------------------------------------------------
# Individual physical steps
# ---------------------------------------------------------------------------


def secondary_electron_signal(render: LayoutRender, yields: MaterialYields) -> np.ndarray:
    """Flat-surface SE yield per pixel, before any topographic term.

    Because the render's layers are disjoint visible-material fractions, the
    per-pixel yield is exactly the area-weighted mean of the yields of the
    materials visible in that pixel.
    """
    signal = np.zeros(render.shape, dtype=np.float64)
    covered = np.zeros(render.shape, dtype=np.float64)
    for name, coverage in render.layers.items():
        signal += yields.for_layer(name) * coverage
        covered += coverage
    signal += yields.substrate * (1.0 - covered)
    return signal


def edge_density(render: LayoutRender) -> np.ndarray:
    """Edge-length density per pixel, as the driver of edge brightening.

    Secondary-electron yield rises where the beam strikes a sloped or vertical
    surface, because the interaction volume sits closer to more escape area. In a
    top-down image this appears as a bright fringe along every feature edge.

    The physically meaningful per-pixel quantity is the *length of material edge*
    inside that pixel. For a coverage field the gradient magnitude is exactly a
    density of boundary length, so summing ``|grad c|`` over the disjoint material
    layers estimates it directly, and does so consistently at any pixel size --
    which matters, because the same physical edge must brighten both captures by
    the same amount despite their tenfold difference in sampling.

    Each boundary is seen by the two materials that meet along it, so the sum is
    halved to avoid double counting. The result is a *per-pixel* edge density: a
    straight edge crossing a pixel completely gives 1.

    Note that this quantity is resolution-dependent by construction -- a coarse
    pixel contains more edge. Converting it into a brightness contribution
    therefore requires scaling by the physical fringe width over the pixel size,
    which ``form_image`` does. Omitting that scaling brightens a 10 nm/px capture
    roughly ten times too much.
    """
    total = np.zeros(render.shape, dtype=np.float64)
    for coverage in render.layers.values():
        grad_y, grad_x = np.gradient(coverage.astype(np.float64))
        total += np.hypot(grad_y, grad_x)
    return np.clip(0.5 * total, 0.0, 1.0)


def _gaussian_kernel(sigma_px: float, truncate: float = 4.0) -> np.ndarray:
    radius = int(np.ceil(truncate * sigma_px))
    if radius < 1:
        return np.array([1.0])
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_px) ** 2)
    return kernel / kernel.sum()


def _convolve_axis(image: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Separable 1-D convolution with reflecting edges, in pure NumPy."""
    if kernel.size == 1:
        return image
    radius = kernel.size // 2
    pad = [(0, 0)] * image.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(image, pad, mode="reflect")
    out = np.zeros_like(image, dtype=np.float64)
    for index, weight in enumerate(kernel):
        window = [slice(None)] * image.ndim
        window[axis] = slice(index, index + image.shape[axis])
        out += weight * padded[tuple(window)]
    return out


def gaussian_blur(image: np.ndarray, sigma_y_px: float, sigma_x_px: float) -> np.ndarray:
    """Anisotropic Gaussian blur modelling the beam point-spread function.

    Separate widths per axis represent astigmatism, which is the dominant
    residual aberration in a slightly mis-tuned column and which the problem
    statement's metadata exposes as ``astigmatism_ratio``.
    """
    if sigma_y_px < 0 or sigma_x_px < 0:
        raise ValueError("sigma must be non-negative")
    out = np.asarray(image, dtype=np.float64)
    if sigma_y_px > 0:
        out = _convolve_axis(out, _gaussian_kernel(sigma_y_px), axis=0)
    if sigma_x_px > 0:
        out = _convolve_axis(out, _gaussian_kernel(sigma_x_px), axis=1)
    return out


def apply_geometric_error(
    image: np.ndarray,
    rotation_deg: float,
    scale_error: float,
    shear_px: float,
    drift_jitter_px: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample the capture through the inter-visit stage error.

    Rotation and scale are taken about the image centre, so a pattern centred in
    the frame stays centred -- which is what keeps the ground truth exact when
    this is applied to the reference capture.

    Shear models a scan axis that is not quite orthogonal. Drift jitter models the
    stage creeping between scan lines: each raster line is displaced horizontally
    by an independent small amount, which is why it is applied per row rather than
    as a smooth field.
    """
    rows, cols = image.shape
    if rotation_deg == 0 and scale_error == 0 and shear_px == 0 and drift_jitter_px == 0:
        return np.asarray(image, dtype=np.float64)

    centre_y, centre_x = (rows - 1) / 2.0, (cols - 1) / 2.0
    grid_y, grid_x = np.mgrid[0:rows, 0:cols].astype(np.float64)
    dy, dx = grid_y - centre_y, grid_x - centre_x

    angle = np.deg2rad(rotation_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    scale = 1.0 + scale_error

    # Inverse map: where in the source does this output pixel come from?
    src_x = (cos_a * dx + sin_a * dy) / scale
    src_y = (-sin_a * dx + cos_a * dy) / scale

    if shear_px:
        src_x = src_x + shear_px * (src_y / max(rows - 1, 1))
    if drift_jitter_px:
        src_x = src_x + rng.normal(0.0, drift_jitter_px, size=(rows, 1))

    return bilinear_sample(image, src_x + centre_x, src_y + centre_y)


def apply_charging(
    image: np.ndarray,
    streak_prob: float,
    intensity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bright horizontal streaks from locally trapped charge.

    Charge accumulating on a poorly grounded feature deflects the beam and
    inflates the collected signal for the remainder of that scan line, decaying as
    the charge bleeds away. The artefact is therefore directional: it follows the
    raster, not the geometry.
    """
    out = np.array(image, dtype=np.float64, copy=True)
    if streak_prob <= 0 or intensity <= 0:
        return out

    rows, cols = out.shape
    struck = rng.random(rows) < streak_prob
    if not struck.any():
        return out

    positions = np.arange(cols, dtype=np.float64)
    for row in np.flatnonzero(struck):
        start = int(rng.integers(0, cols))
        decay = max(cols * float(rng.uniform(0.05, 0.35)), 1.0)
        profile = np.where(
            positions >= start, np.exp(-(positions - start) / decay), 0.0
        )
        out[row] += intensity * float(rng.uniform(0.5, 1.5)) * profile * out[row].mean()
    return out


def apply_vignette(image: np.ndarray, strength: float) -> np.ndarray:
    """Radial falloff in collection efficiency away from the optical axis."""
    if strength <= 0:
        return np.asarray(image, dtype=np.float64)
    rows, cols = image.shape
    grid_y, grid_x = np.mgrid[0:rows, 0:cols].astype(np.float64)
    ny = (grid_y - (rows - 1) / 2.0) / max((rows - 1) / 2.0, 1.0)
    nx = (grid_x - (cols - 1) / 2.0) / max((cols - 1) / 2.0, 1.0)
    radius_sq = np.clip(nx * nx + ny * ny, 0.0, 2.0) / 2.0
    return np.asarray(image, dtype=np.float64) * (1.0 - strength * radius_sq)


def apply_dose_noise(
    signal: np.ndarray, dose: float, rng: np.random.Generator
) -> np.ndarray:
    """Poisson shot noise from a finite electron count.

    The dominant noise source in SEM imaging. The number of secondary electrons
    detected per pixel is a Poisson variable whose mean is the dose times the local
    yield, so the signal-to-noise ratio goes as the square root of dose. Halving
    the dose does not halve the noise -- it worsens SNR by root two, which is why
    the tenfold dose difference between the two captures produces a roughly
    threefold difference in relative noise rather than a tenfold one.

    Returns the signal in yield units, so downstream stages are dose-independent.
    """
    if dose <= 0:
        raise ValueError("dose must be positive")
    mean_counts = np.clip(signal, 0.0, None) * dose
    return rng.poisson(mean_counts).astype(np.float64) / dose


def form_image(
    render: LayoutRender,
    params: CaptureParams,
    pixel_size_nm: float,
    rng: np.random.Generator,
    yields: Optional[MaterialYields] = None,
    apply_geometry: bool = False,
) -> np.ndarray:
    """Run the full imaging chain, returning an 8-bit greyscale image.

    Parameters
    ----------
    render:
        Disjoint material coverage fields from a layout model.
    params:
        Acquisition settings for this capture.
    pixel_size_nm:
        Physical size of one pixel, used to convert nanometre-valued parameters
        into pixels. This is what makes the same physical spot size produce the
        correct blur at both magnifications.
    rng:
        Generator for every stochastic term. Each capture must be given its own.
    apply_geometry:
        Whether to apply the inter-visit stage error. True for the reference
        capture only; see the module docstring.
    """
    yields = yields or MaterialYields()

    # 1-2. specimen: material contrast, then the topographic edge term.
    signal = secondary_electron_signal(render, yields)
    if params.edge_gain and params.edge_width_nm:
        # The bright fringe has a fixed physical width, so it fills a smaller
        # fraction of a large pixel. Scaling by fringe width over pixel size is
        # what makes the same physical edge produce a consistent contribution at
        # both magnifications -- and it is why the reference capture shows crisp
        # bright edges while the wide-field capture shows only a faint lift.
        fringe_fraction = np.clip(params.edge_width_nm / pixel_size_nm, 0.0, 1.0)
        signal = signal * (1.0 + params.edge_gain * fringe_fraction * edge_density(render))

    # 3. column: beam point spread, in pixels at this magnification.
    sigma_px = params.spot_size_nm / pixel_size_nm
    if sigma_px > 0:
        signal = gaussian_blur(
            signal, sigma_px * params.astigmatism_ratio, sigma_px
        )

    # 4. scan: inter-visit stage error.
    if apply_geometry:
        signal = apply_geometric_error(
            signal,
            params.rotation_deg,
            params.scale_error,
            params.shear_px,
            params.drift_jitter_px,
            rng,
        )

    # 5-6. charging, then collection efficiency.
    signal = apply_charging(
        signal, params.charging_streak_prob, params.charging_streak_intensity, rng
    )
    signal = apply_vignette(signal, params.vignette_strength)

    # Normalise before detection so that dose sets noise rather than brightness.
    mean_signal = float(np.mean(signal))
    if mean_signal > 0:
        signal = signal / mean_signal

    # 7-8. detection: shot noise, then readout noise.
    signal = apply_dose_noise(signal, params.dose, rng)
    levels = signal * params.target_level
    if params.detector_noise_sigma > 0:
        levels = levels + rng.normal(0.0, params.detector_noise_sigma, size=levels.shape)

    # 9-10. display transfer and sensor defects.
    if params.gamma != 1.0:
        levels = np.clip(levels, 0.0, None)
        peak = max(float(levels.max()), 1e-9)
        levels = peak * (levels / peak) ** params.gamma
    if params.speckle_sigma > 0:
        levels = levels * (1.0 + rng.normal(0.0, params.speckle_sigma, size=levels.shape))
    if params.salt_pepper_prob > 0:
        draw = rng.random(levels.shape)
        levels = np.where(draw < params.salt_pepper_prob / 2.0, 0.0, levels)
        levels = np.where(draw > 1.0 - params.salt_pepper_prob / 2.0, 255.0, levels)

    # 11. quantisation.
    return np.rint(np.clip(levels, 0.0, 255.0)).astype(np.uint8)

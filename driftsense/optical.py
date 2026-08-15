"""Optical-microscope image formation: three-channel, diffraction-limited.

PURPOSE
-------
The problem statement offers bonus credit for a solution that also generalises to
optical-microscope imagery -- RGB, three channels -- rather than only the SEM
greyscale case. This is that path. It reuses the same layout models, the same
coordinate convention and the same ground truth; only image formation changes.

WHY THE OPTICAL CASE DIFFERS
----------------------------
Not the colour. The resolution.

An optical microscope is diffraction-limited to roughly ``lambda / (2 NA)``. At a
550 nm green wavelength and a numerical aperture of 0.9 that is about 300 nm --
an order of magnitude coarser than the SEM spot sizes modelled elsewhere in this
project, and roughly *eight times* the DRAM bit-line pitch.

So the cell array does not merely blur. It disappears. What survives into an
optical image is the mat and block structure at 2000-3200 nm, and the contrast
difference between array regions and periphery. That inverts the difficulty of
the problem: under the SEM the fine grating is the signal and the mat boundaries
are the rare anchor, while under the optical microscope the grating is gone and
the boundaries are all there is. A pair with no structural anchor, which is merely
ambiguous under the SEM, is *featureless* under an optical microscope.

This is worth stating because it means an optical accuracy figure is not
comparable with the SEM one and should never be quoted alongside it as though it
were the same measurement.

COLOUR MODEL
------------
Colour in a real optical inspection image comes mostly from thin-film
interference in the dielectric stack: layer thickness modulates which wavelengths
reflect, which is why bare silicon, oxide and metal look different. Modelling that
properly needs a film stack this project does not have, so each material is given
a plausible flat RGB reflectance instead, with the consequence that the
three channels here are close to scaled copies of one another.

One physical effect is real and is modelled: **the blur differs per channel**,
because the diffraction limit scales with wavelength. Blue resolves finer than
red. That is genuine chromatic information, and it is why the channels are
convolved separately rather than a greyscale image being tinted.

MEASURED PERFORMANCE
--------------------
Measured over 10 optical pairs with a 30 nm/px reference (30 um field) and a
300 nm/px search (300 um field), against the DRAM layout:

    correlation at the true location     0.898
    correlation at the best location     0.938
    true location was the global maximum on 2 of 10 pairs

The signal is present -- 0.898 is a strong correlation -- but somewhere else
matches better. The optical regime is **aliasing-limited, not noise-limited**,
and the diagnosis is specific:

1. Blur destroys the cell grating entirely, so the only property of a mat that
   survives is its *duty cycle*, the fraction of area covered by metal.
2. Line width is drawn as a fixed ratio of pitch, so duty cycle was identical in
   every mat no matter how much the pitch varied. Every mat therefore imaged
   identically once blurred -- unsolvable by construction, exactly the an earlier revision
   defect in a new guise. ``DramParams.mat_width_variation`` now varies line
   width independently of pitch to break that, which is critical-dimension
   non-uniformity and is real.
3. Vignetting then dominates what little contrast remains. With it disabled the
   true location becomes the global maximum on 2 of 10 pairs rather than 0 --
   better, and still not solved.

The conclusion is that an optical tool cannot navigate by cell-array
texture, and would use die-level and block-level features tens of microns across.
Modelling those needs a floorplan layer this project does not have. The bonus
path is therefore **implemented, physically grounded and characterised, but not
solved**, and its accuracy figure must not be quoted alongside the SEM one.

LOCATOR HANDLING
----------------
``locate`` reduces any 3-channel input to luminance and proceeds unchanged. Given
the channels are near-copies, that is close to optimal here and is certainly the
right default -- a per-channel correlation would triple the cost to recover
information the model does not really contain. The measured figures are in the
README.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .imaging import gaussian_blur
from .layouts.base import LayoutRender

__all__ = ["OpticalParams", "MaterialColours", "rayleigh_resolution_nm", "form_rgb_image"]

#: Channel centre wavelengths in nanometres, in R, G, B order.
CHANNEL_WAVELENGTHS_NM: Tuple[float, float, float] = (620.0, 550.0, 460.0)


@dataclass(frozen=True)
class MaterialColours:
    """Flat RGB reflectance per material, each component in ``[0, 1]``.

    Values are plausible rather than derived: see the module docstring on why a
    proper film-stack model is out of scope. Keys match the layer names the
    layout models emit; anything unlisted falls back to ``default``.
    """

    colours: Dict[str, Tuple[float, float, float]] = field(
        default_factory=lambda: {
            # DRAM
            "array_field": (0.32, 0.34, 0.40),
            "wordline": (0.55, 0.52, 0.45),
            "bitline": (0.62, 0.60, 0.55),
            "contact": (0.80, 0.78, 0.70),
            # FinFET
            "field": (0.30, 0.33, 0.42),
            "fin": (0.52, 0.54, 0.56),
            "gate": (0.78, 0.74, 0.62),
        }
    )
    substrate: Tuple[float, float, float] = (0.38, 0.40, 0.46)
    default: Tuple[float, float, float] = (0.50, 0.50, 0.50)

    def for_layer(self, name: str) -> Tuple[float, float, float]:
        return self.colours.get(name, self.default)


@dataclass(frozen=True)
class OpticalParams:
    """Acquisition settings for one optical capture."""

    #: Numerical aperture. 0.9 is a dry objective at the high end; immersion
    #: reaches ~1.4 but is not typical for wafer inspection.
    numerical_aperture: float = 0.90
    #: Multiplies the diffraction-limited spot, standing in for aberrations and
    #: defocus. 1.0 is a perfect objective.
    aberration_factor: float = 1.15
    #: Mean photons per pixel at full reflectance. Optical detectors collect far
    #: more signal than an SEM at comparable speed, so this is large.
    exposure: float = 4000.0
    #: Read noise standard deviation, in the same units as ``exposure``.
    read_noise: float = 12.0
    #: Illumination falloff towards the frame corners.
    vignette_strength: float = 0.10
    #: Display gamma.
    gamma: float = 1.0
    #: Per-channel gain, standing in for white balance being slightly off.
    channel_gain: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if self.numerical_aperture <= 0.0:
            raise ValueError("numerical_aperture must be positive")
        if self.aberration_factor < 1.0:
            raise ValueError(
                "aberration_factor must be at least 1; a real objective cannot beat "
                f"its own diffraction limit, got {self.aberration_factor}"
            )
        if self.exposure <= 0.0:
            raise ValueError("exposure must be positive")
        if self.read_noise < 0.0:
            raise ValueError("read_noise must be non-negative")
        if not 0.0 <= self.vignette_strength < 1.0:
            raise ValueError("vignette_strength must lie in [0, 1)")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if len(self.channel_gain) != 3 or any(g <= 0.0 for g in self.channel_gain):
            raise ValueError("channel_gain must be three positive numbers")

    def resolution_nm(self, wavelength_nm: float) -> float:
        """Rayleigh resolution for one wavelength, including aberrations."""
        return self.aberration_factor * rayleigh_resolution_nm(
            wavelength_nm, self.numerical_aperture
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "numerical_aperture": self.numerical_aperture,
            "aberration_factor": self.aberration_factor,
            "exposure": self.exposure,
            "read_noise": self.read_noise,
            "vignette_strength": self.vignette_strength,
            "gamma": self.gamma,
            "channel_gain": list(self.channel_gain),
            "resolution_nm": {
                "red": round(self.resolution_nm(CHANNEL_WAVELENGTHS_NM[0]), 2),
                "green": round(self.resolution_nm(CHANNEL_WAVELENGTHS_NM[1]), 2),
                "blue": round(self.resolution_nm(CHANNEL_WAVELENGTHS_NM[2]), 2),
            },
        }


def rayleigh_resolution_nm(wavelength_nm: float, numerical_aperture: float) -> float:
    """Rayleigh criterion, ``0.61 * lambda / NA``.

    The smallest separation at which two points are still distinguishable. Blue
    light resolves finer than red, which is the one genuinely chromatic effect
    this model carries.
    """
    if wavelength_nm <= 0.0 or numerical_aperture <= 0.0:
        raise ValueError("wavelength and numerical aperture must be positive")
    return 0.61 * wavelength_nm / numerical_aperture


def _reflectance(render: LayoutRender, colours: MaterialColours) -> np.ndarray:
    """Composite the disjoint coverage layers into an ``(H, W, 3)`` reflectance map."""
    rows, cols = render.shape
    image = np.zeros((rows, cols, 3), dtype=np.float64)
    covered = np.zeros((rows, cols), dtype=np.float64)
    for name, coverage in render.layers.items():
        rgb = colours.for_layer(name)
        for channel in range(3):
            image[..., channel] += float(rgb[channel]) * coverage
        covered += coverage
    exposed = 1.0 - covered
    for channel in range(3):
        image[..., channel] += float(colours.substrate[channel]) * exposed
    return np.clip(image, 0.0, 1.0)


def form_rgb_image(
    render: LayoutRender,
    params: OpticalParams,
    pixel_size_nm: float,
    rng: np.random.Generator,
    colours: Optional[MaterialColours] = None,
) -> np.ndarray:
    """Run the optical imaging chain, returning an 8-bit ``(H, W, 3)`` array.

    The steps, in order: material reflectance, per-channel diffraction blur,
    vignetting, photon shot noise, read noise, gamma, quantisation.
    """
    colours = colours or MaterialColours()
    if pixel_size_nm <= 0.0:
        raise ValueError("pixel_size_nm must be positive")

    image = _reflectance(render, colours)
    rows, cols = image.shape[:2]

    # 1. Diffraction, per channel. The Rayleigh radius is converted to a Gaussian
    #    sigma by the usual approximation sigma ~ r / 2.3, which matches the
    #    central lobe of an Airy disc closely enough for a synthetic image.
    for channel, wavelength in enumerate(CHANNEL_WAVELENGTHS_NM):
        sigma_px = params.resolution_nm(wavelength) / 2.3 / pixel_size_nm
        if sigma_px > 0.05:
            image[..., channel] = gaussian_blur(image[..., channel], sigma_px, sigma_px)

    # 2. Vignetting: a smooth radial falloff, identical across channels.
    if params.vignette_strength > 0.0:
        yy, xx = np.mgrid[0:rows, 0:cols]
        cy, cx = (rows - 1) / 2.0, (cols - 1) / 2.0
        radius = np.hypot((yy - cy) / max(cy, 1.0), (xx - cx) / max(cx, 1.0))
        falloff = 1.0 - params.vignette_strength * np.clip(radius / math.sqrt(2.0), 0.0, 1.0) ** 2
        image *= falloff[..., None]

    # 3. Photon counting, then read noise. Shot noise is Poisson in the collected
    #    signal, so it scales as the square root of exposure exactly as in the SEM
    #    model -- the physics is the same, only the photon budget is larger.
    for channel in range(3):
        gain = float(params.channel_gain[channel])
        expected = np.clip(image[..., channel] * params.exposure * gain, 0.0, None)
        counted = rng.poisson(expected).astype(np.float64)
        if params.read_noise > 0.0:
            counted += rng.normal(0.0, params.read_noise, counted.shape)
        image[..., channel] = counted / (params.exposure * gain)

    image = np.clip(image, 0.0, 1.0)
    if params.gamma != 1.0:
        image = image ** (1.0 / params.gamma)
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)

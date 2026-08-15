"""Image quality: how much real structure does this capture actually carry?

PURPOSE
-------
Preprocessing was measured to improve a well-conditioned image three to five fold
and to destroy a blur-limited one. It ships disabled because there was no way to
tell the two apart *at inference time*: the locator is handed two images and
nothing else, with no dose, no spot size and no ground truth.

This module is the missing measurement. It estimates, from a single image, the
highest spatial frequency at which real structure still stands above the noise
floor. That is the quantity the preprocessing decision actually turns on: a
capture whose structure survives to high frequency has detail worth
band-passing, and one whose structure died at low frequency has only noise up
there, which band-passing then amplifies.

HOW IT WORKS
------------
The radially averaged power spectrum of a periodic layout has two components: the
lattice, appearing as power concentrated at the pitch frequency and its harmonics,
and a broadly flat noise floor from shot and detector noise. Blur is a low-pass
filter, so it attenuates the lattice terms while leaving the detector noise floor
untouched, so the two separate cleanly in the spectrum even though they are
inseparable in the image.

So:

    1. Window the image, to stop the edge discontinuity leaking a false
       high-frequency tail across every radial bin.
    2. Radially average the power spectrum into frequency bins.
    3. Take the noise floor as the median power in the highest-frequency bins,
       where a blurred layout contributes essentially nothing.
    4. Find the highest frequency whose power exceeds that floor by a factor.
    5. Report it as a fraction of Nyquist.

That fraction is the ``cutoff_fraction``, and it runs from near 0 for an image
that is pure noise to near 1 for one that is sharp to the pixel.

SCOPE
-----
Only the pixels. No layout parameters, no capture settings, no ground truth. This
runs inside ``locate`` on the evaluator's data, so it is held to exactly the same
standard as the rest of the locator, unlike ``failures``, which is an after-the-
fact analysis tool and may read anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "QualityConfig",
    "QualityEstimate",
    "radial_power_spectrum",
    "estimate_quality",
]


@dataclass(frozen=True)
class QualityConfig:
    """Settings for the spectral quality estimate."""

    #: Number of radial frequency bins. Enough to resolve where the lattice dies
    #: without making each bin so sparse that its average is noisy.
    n_bins: int = 48
    #: Bins above this fraction of Nyquist define the noise floor. A layout
    #: blurred by any realistic beam contributes nothing this high, so what
    #: remains is detector and shot noise.
    noise_band_start: float = 0.8
    #: Power must exceed the noise floor by this factor to count as structure.
    #: Two is a deliberately unambitious threshold: the question is whether
    #: anything is there at all, not how strong it is.
    detection_factor: float = 2.0
    #: Apply a Hann window before the transform. Without it the wrap-around
    #: discontinuity at the image edge contributes a broadband tail that raises
    #: every bin and makes a blurred image look sharp.
    window: bool = True

    def __post_init__(self) -> None:
        if self.n_bins < 8:
            raise ValueError(f"n_bins must be at least 8, got {self.n_bins}")
        if not 0.0 < self.noise_band_start < 1.0:
            raise ValueError(
                f"noise_band_start must lie in (0, 1), got {self.noise_band_start}"
            )
        if self.detection_factor <= 1.0:
            raise ValueError(
                "detection_factor must exceed 1, otherwise the noise floor detects "
                f"itself; got {self.detection_factor}"
            )


@dataclass(frozen=True)
class QualityEstimate:
    """What the spectrum says about one image."""

    #: Highest frequency carrying structure, as a fraction of Nyquist. Zero when
    #: no bin clears the floor.
    cutoff_fraction: float
    #: Median power in the noise band.
    noise_floor: float
    #: Ratio of peak structural power to the noise floor. A rough contrast
    #: measure, reported for diagnosis rather than used in the decision.
    peak_over_floor: float
    #: How many bins cleared the threshold.
    n_bins_above: int

    def as_dict(self) -> dict:
        return {
            "cutoff_fraction": round(float(self.cutoff_fraction), 5),
            "noise_floor": float(self.noise_floor),
            "peak_over_floor": round(float(self.peak_over_floor), 4),
            "n_bins_above": int(self.n_bins_above),
        }


def _hann_2d(rows: int, cols: int) -> np.ndarray:
    """Separable Hann window. Hand-written; NumPy's ``hanning`` is fine but this
    keeps the symmetry convention explicit and matches the raster module's habit
    of not relying on edge-case behaviour it has not checked."""
    def axis(n: int) -> np.ndarray:
        if n < 2:
            return np.ones(n, dtype=np.float64)
        k = np.arange(n, dtype=np.float64)
        return 0.5 - 0.5 * np.cos(2.0 * np.pi * k / (n - 1))

    return np.outer(axis(rows), axis(cols))


def radial_power_spectrum(
    image: np.ndarray, n_bins: int = 48, window: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Radially averaged power spectrum.

    Returns ``(frequencies, power)`` where frequencies are fractions of Nyquist
    in ``(0, 1]`` and power is the mean squared magnitude in each annulus.

    The DC term is excluded. It carries the mean brightness, which says nothing
    about resolution and would otherwise dominate every summary statistic by
    several orders of magnitude.
    """
    data = np.asarray(image, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {data.shape}")
    rows, cols = data.shape
    if rows < 4 or cols < 4:
        raise ValueError(f"image too small to have a spectrum: {rows}x{cols}")

    data = data - data.mean()
    if window:
        data = data * _hann_2d(rows, cols)

    spectrum = np.fft.fftshift(np.fft.fft2(data))
    power = np.abs(spectrum) ** 2

    # Radius of each bin centre, normalised so that 1.0 is Nyquist along the
    # shorter axis. Using the shorter axis keeps the measure comparable between
    # non-square images.
    fy = np.fft.fftshift(np.fft.fftfreq(rows)) * 2.0  # (-1, 1] in Nyquist units
    fx = np.fft.fftshift(np.fft.fftfreq(cols)) * 2.0
    radius = np.hypot(fy[:, None], fx[None, :])

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    inside = radius <= 1.0
    index = np.digitize(radius[inside], edges) - 1
    index = np.clip(index, 0, n_bins - 1)
    values = power[inside]

    totals = np.bincount(index, weights=values, minlength=n_bins)
    counts = np.bincount(index, minlength=n_bins).astype(np.float64)
    counts[counts == 0.0] = 1.0
    means = totals / counts

    centres = 0.5 * (edges[:-1] + edges[1:])
    # Drop the innermost bin: it contains DC and its immediate neighbourhood,
    # which is dominated by the residual mean and the window's own transform.
    return centres[1:], means[1:]


def estimate_quality(
    image: np.ndarray, config: Optional[QualityConfig] = None
) -> QualityEstimate:
    """Estimate how far real structure survives in ``image``.

    See the module docstring for the reasoning. The returned
    ``cutoff_fraction`` is the number the preprocessing gate is built on.
    """
    config = config or QualityConfig()
    freqs, power = radial_power_spectrum(image, config.n_bins, config.window)

    noise_band = power[freqs >= config.noise_band_start]
    if noise_band.size == 0:
        noise_band = power[-1:]
    floor = float(np.median(noise_band))

    if not math.isfinite(floor) or floor <= 0.0:
        # A perfectly flat or degenerate image has no noise floor to speak of.
        # Reporting zero says "no structure detected", which is correct.
        return QualityEstimate(0.0, 0.0, 0.0, 0)

    above = power >= floor * config.detection_factor
    n_above = int(np.count_nonzero(above))
    cutoff = float(freqs[above].max()) if n_above else 0.0
    peak_over_floor = float(power.max() / floor)

    return QualityEstimate(
        cutoff_fraction=cutoff,
        noise_floor=floor,
        peak_over_floor=peak_over_floor,
        n_bins_above=n_above,
    )

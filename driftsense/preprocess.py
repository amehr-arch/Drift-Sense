"""Preprocessing: band-pass filtering and local contrast normalisation.

PURPOSE
-------
Normalised cross-correlation already removes a global gain and offset, but two
effects remain that filtering addresses.

The normalisation is per window, so a slow illumination gradient across the search
image, such as vignetting or a charging streak, still biases the score, because
the gradient is not constant within a 100 px window. A high-pass removes structure
coarser than the template and suppresses it.

The two captures do not share a spatial frequency band. The wide-field image is
taken with a larger beam spot, so its high frequencies are attenuated relative to
the reference. A low-pass set near the coarser capture's spot size prevents the
reference contributing detail the search image cannot carry.

Local contrast normalisation divides out slowly varying contrast, so a locally
faint region matches as readily as a locally strong one.

HOW IT IS APPLIED
-----------------
Not unconditionally. ``locate`` runs the pipeline twice, once with these filters
and once without, and returns the answer whose runner-up margin is larger. See
``LocalisationConfig.arbitrate_preprocessing``.

Measured on the solvable subset of three datasets:

                     development     held out      validation
    filters off       0.344 / 100%   0.802 / 62%   0.703 / 54%
    filters on        0.078 / 100%   219.4 / 38%   0.358 / 65%
    arbitrated        0.096 / 100%   0.707 / 62%   0.354 / 69%

Arbitration is at least as good as leaving the filters off on every metric in
every regime, and avoids the degradation seen when they are always applied. The
cost is a second full pass.

SCOPE
-----
No denoising. The degradation study measured shot noise as a minor term:
correlation integrates over ten thousand pixels, so uncorrelated noise averages
down by about a hundred. Beam spot is the dominant term, and band-pass filtering
addresses it directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .correlate import window_sum
from .imaging import gaussian_blur

__all__ = [
    "PreprocessConfig",
    "box_blur",
    "bandpass",
    "local_contrast_normalise",
    "preprocess",
]

#: A box of half-width r has standard deviation sqrt(((2r+1)^2 - 1)/12), so this
#: is the radius whose box matches a given Gaussian sigma in second moment.
def _radius_for_sigma(sigma: float) -> int:
    return max(1, int(round((np.sqrt(12.0 * sigma * sigma + 1.0) - 1.0) / 2.0)))


def box_blur(image: np.ndarray, radius: int) -> np.ndarray:
    """Mean over a square window, in time independent of the window size.

    Used wherever the filter only has to be *smooth* rather than exactly
    Gaussian -- estimating an illumination gradient, or a local contrast level.
    A direct Gaussian at these scales costs a kernel tap per pixel: local
    contrast normalisation with sigma 25 needs a 201-tap kernel on two axes,
    which measured at roughly two seconds per pair and dominated the whole
    locator. Integral images give the same job in a few milliseconds.
    """
    radius = max(1, int(radius))
    data = np.asarray(image, dtype=np.float64)
    rows, cols = data.shape
    size = 2 * radius + 1
    padded = np.pad(data, radius, mode="reflect")
    if size > padded.shape[0] or size > padded.shape[1]:
        return np.full_like(data, data.mean())
    totals = window_sum(padded, size, size)
    return totals[:rows, :cols] / float(size * size)


@dataclass(frozen=True)
class PreprocessConfig:
    """Band-pass and contrast-normalisation settings, in pixels of the target grid.

    Both scales are expressed in *search-image* pixels and applied consistently to
    the template, which is already at search-image scale by the time preprocessing
    runs.
    """

    #: Removes structure coarser than this, killing illumination gradients,
    #: vignetting and charging. Zero disables the high-pass.
    highpass_sigma_px: float = 12.0
    #: Removes structure finer than this. Set near the coarser capture's spot size
    #: so the reference is not credited with detail the search image cannot carry.
    #: Zero disables the low-pass.
    lowpass_sigma_px: float = 0.8
    #: Divides out slowly varying contrast, so a locally faint region matches as
    #: readily as a locally strong one. Zero disables it.
    contrast_sigma_px: float = 25.0
    #: Floor on the local standard deviation, preventing division blow-up in flat
    #: regions where there is no structure to normalise.
    contrast_floor: float = 1e-3
    #: Template side the sigmas above are quoted for. The filter scales are
    #: meaningful relative to the object being matched, not in absolute pixels:
    #: a high-pass at sigma 12 removes a gentle gradient from a 100 px template
    #: but erases a 20 px one entirely. ``scaled_for`` rescales accordingly, so
    #: the same config behaves sensibly at any template size.
    reference_template_px: float = 100.0

    def __post_init__(self) -> None:
        for name in (
            "highpass_sigma_px",
            "lowpass_sigma_px",
            "contrast_sigma_px",
            "contrast_floor",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.lowpass_sigma_px and self.highpass_sigma_px:
            if self.lowpass_sigma_px >= self.highpass_sigma_px:
                raise ValueError(
                    "lowpass_sigma_px must be smaller than highpass_sigma_px, "
                    "otherwise the band is empty"
                )


    def scaled_for(self, template_px: float) -> "PreprocessConfig":
        """Return this config with its spatial scales matched to a template size.

        The low-pass is left alone: it represents the physical beam profile,
        which is a property of the microscope rather than of the template.
        """
        if template_px <= 0:
            raise ValueError("template_px must be positive")
        factor = float(template_px) / float(self.reference_template_px)

        # A zero sigma means the filter is deliberately switched off, so it must
        # stay off. Applying the minimum floor unconditionally turned a disabled
        # high-pass back on, which then failed the "band must not be empty"
        # validation and crashed -- a filter the caller had explicitly asked not
        # to run.
        def rescale(sigma: float, minimum: float) -> float:
            return 0.0 if sigma <= 0 else max(sigma * factor, minimum)

        return PreprocessConfig(
            highpass_sigma_px=rescale(self.highpass_sigma_px, 2.0),
            lowpass_sigma_px=self.lowpass_sigma_px,
            contrast_sigma_px=rescale(self.contrast_sigma_px, 4.0),
            contrast_floor=self.contrast_floor,
            reference_template_px=template_px,
        )


def bandpass(image: np.ndarray, highpass_sigma: float, lowpass_sigma: float) -> np.ndarray:
    """Keep only spatial frequencies both captures can carry.

    Implemented as a difference of Gaussians: subtracting a heavily blurred copy
    removes the low frequencies, and a light blur removes the high ones.
    """
    out = np.asarray(image, dtype=np.float64)
    if lowpass_sigma > 0:
        # Small sigma, so a true Gaussian is both cheap and worth having: this is
        # the filter that has to match the physical beam profile.
        out = gaussian_blur(out, lowpass_sigma, lowpass_sigma)
    if highpass_sigma > 0:
        # Large sigma, and only a smooth background estimate is needed, so a box
        # is indistinguishable in effect and vastly cheaper.
        out = out - box_blur(out, _radius_for_sigma(highpass_sigma))
    return out


def local_contrast_normalise(
    image: np.ndarray, sigma: float, floor: float = 1e-3
) -> np.ndarray:
    """Divide by the local standard deviation, estimated at scale ``sigma``.

    Makes a locally faint region match as readily as a locally strong one, which
    matters because vignetting and charging vary the contrast across the frame.
    """
    if sigma <= 0:
        return np.asarray(image, dtype=np.float64)
    out = np.asarray(image, dtype=np.float64)
    radius = _radius_for_sigma(sigma)
    local_mean = box_blur(out, radius)
    centred = out - local_mean
    local_variance = box_blur(centred * centred, radius)
    scale = np.sqrt(np.maximum(local_variance, 0.0))

    # Floor the divisor against the *median* local contrast, not the maximum.
    #
    # Flooring against the maximum makes the floor negligible whenever any part of
    # the frame is high-contrast, so quiet regions get divided by a near-zero
    # local standard deviation and their noise is amplified without limit. On the
    # tuned imaging regime that never bit; on a held-out regime with a third of
    # the dose it was catastrophic -- median error 219 px against 0.80 px with
    # preprocessing switched off entirely. The median is a robust scale estimate
    # and does not collapse when the frame contains a few strong features.
    typical = float(np.median(scale))
    if typical <= 0:
        typical = float(scale.mean())
    return centred / np.maximum(scale, max(floor * typical, 1e-12))


def preprocess(image: np.ndarray, config: PreprocessConfig | None = None) -> np.ndarray:
    """Band-pass then contrast-normalise, returning float64."""
    config = config or PreprocessConfig()
    out = bandpass(image, config.highpass_sigma_px, config.lowpass_sigma_px)
    return local_contrast_normalise(out, config.contrast_sigma_px, config.contrast_floor)

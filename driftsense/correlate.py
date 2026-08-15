"""Normalised cross-correlation, peak extraction and sub-pixel peak refinement.

WHY NORMALISED, AND WHY THIS FORMULATION
-----------------------------------------
Raw cross-correlation is useless here: it is maximised by bright regions rather
than by matching structure, so on an image with any illumination gradient it
reports the brightest area regardless of content. Normalised cross-correlation
divides out the local mean and local standard deviation of the search window,
which makes the score invariant to any affine change in intensity. That matters
because the reference and search captures are separate physical acquisitions at
different dose, and will differ in both brightness and contrast.

The formulation follows Lewis (1995), "Fast Normalized Cross-Correlation":

    NCC(u,v) = sum[ (S_w - mean(S_w)) * (T - mean(T)) ]
               / sqrt( sum[(S_w - mean(S_w))^2] * sum[(T - mean(T))^2] )

where S_w is the search window at offset (u,v). Two observations make this cheap:

1. Once the template is zero-meaned its sum vanishes, so the numerator collapses
   to a plain correlation of the search image with the zero-meaned template. That
   is one FFT multiply.
2. The per-window mean and variance in the denominator need window sums of S and
   S^2, which come from integral images in O(1) per position.

Total cost is three FFTs of the search image plus two cumulative sums, which is
milliseconds for the 1000x1000 / 100x100 case in this problem.

CIRCULAR VERSUS LINEAR CORRELATION
----------------------------------
The FFT computes circular correlation. No padding beyond the search size is
needed anyway: for offsets in [0, N-M] the template window lies wholly inside the
search image, so no term wraps around and the circular result equals the linear
one exactly. Only that valid range is returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

__all__ = [
    "Peak",
    "normalised_cross_correlation",
    "window_sum",
    "find_peaks",
    "refine_peak_subpixel",
    "peak_to_sidelobe_ratio",
]


@dataclass(frozen=True)
class Peak:
    """A local maximum of a correlation surface, in surface index coordinates.

    ``row`` and ``col`` are the offsets at which the template's top-left corner
    aligns with the search image. ``row_offset`` and ``col_offset`` are the
    sub-pixel corrections from parabolic interpolation, in the same units.
    """

    score: float
    row: int
    col: int
    row_offset: float = 0.0
    col_offset: float = 0.0

    @property
    def refined_row(self) -> float:
        return self.row + self.row_offset

    @property
    def refined_col(self) -> float:
        return self.col + self.col_offset


def window_sum(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Sum of every ``height x width`` window of ``image``, via an integral image.

    Returns an array of shape ``(H - height + 1, W - width + 1)`` where element
    ``(u, v)`` is the sum over ``image[u:u+height, v:v+width]``.
    """
    if image.ndim != 2:
        raise ValueError("image must be two-dimensional")
    rows, cols = image.shape
    if height > rows or width > cols:
        raise ValueError(
            f"window {height}x{width} does not fit inside image {rows}x{cols}"
        )
    integral = np.zeros((rows + 1, cols + 1), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(np.asarray(image, dtype=np.float64), axis=0), axis=1)
    return (
        integral[height:, width:]
        - integral[:-height, width:]
        - integral[height:, :-width]
        + integral[:-height, :-width]
    )


def normalised_cross_correlation(search: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Zero-mean normalised cross-correlation over all valid template offsets.

    Parameters
    ----------
    search, template:
        Two-dimensional arrays; ``template`` must fit inside ``search``.

    Returns
    -------
    Surface of shape ``(N0 - M0 + 1, N1 - M1 + 1)`` with values in ``[-1, 1]``.
    Positions where the search window is perfectly flat have no defined
    correlation and are reported as 0 rather than as a division by zero.
    """
    S = np.asarray(search, dtype=np.float64)
    T = np.asarray(template, dtype=np.float64)
    if S.ndim != 2 or T.ndim != 2:
        raise ValueError("both inputs must be two-dimensional")

    rows, cols = S.shape
    t_rows, t_cols = T.shape
    if t_rows > rows or t_cols > cols:
        raise ValueError(
            f"template {t_rows}x{t_cols} does not fit inside search {rows}x{cols}"
        )

    count = t_rows * t_cols
    if count < 2:
        raise ValueError("template must contain at least two pixels")

    centred_template = T - T.mean()
    template_energy = float(np.sum(centred_template * centred_template))
    if template_energy <= 0.0:
        raise ValueError("template has zero contrast; correlation is undefined")

    # Removing the global mean leaves NCC unchanged (it is invariant to a
    # constant offset) but keeps the magnitudes in the variance calculation
    # small, which limits cancellation error in the sum-of-squares term.
    S = S - S.mean()

    # Numerator: circular correlation, valid region only (see module docstring).
    spectrum = np.fft.rfft2(S)
    kernel = np.fft.rfft2(centred_template, s=S.shape)
    correlation = np.fft.irfft2(spectrum * np.conj(kernel), s=S.shape)
    numerator = correlation[: rows - t_rows + 1, : cols - t_cols + 1]

    # Denominator: local variance of the search window, from integral images.
    totals = window_sum(S, t_rows, t_cols)
    squares = window_sum(S * S, t_rows, t_cols)
    variance = squares - (totals * totals) / count
    np.maximum(variance, 0.0, out=variance)  # guard against cancellation noise
    denominator = np.sqrt(variance * template_energy)

    surface = np.zeros_like(numerator)
    largest = float(denominator.max())
    if largest > 0.0:
        usable = denominator > largest * 1e-12
        np.divide(numerator, denominator, out=surface, where=usable)
    return np.clip(surface, -1.0, 1.0, out=surface)


def find_peaks(surface: np.ndarray, min_distance: int, max_peaks: int) -> List[Peak]:
    """Extract the strongest local maxima, separated by ``min_distance``.

    SHORTLIST RATHER THAN ARGMAX
    ----------------------------
    On a periodic layout the correlation surface has a lattice of near-equal
    peaks, and the global maximum is frequently the wrong one -- a fraction of a
    percent of score separates the true location from an alias one cell over.
    Returning a shortlist converts an unforgiving "be right first time" problem
    into a ranking problem, and it is what makes the centre tiebreak required by
    the problem statement implementable at all.

    Suppression uses a square neighbourhood, which is the right shape here: the
    aliases lie on a Manhattan lattice, not a circle.
    """
    if surface.ndim != 2:
        raise ValueError("surface must be two-dimensional")
    if max_peaks < 1:
        raise ValueError("max_peaks must be at least 1")
    radius = max(1, int(min_distance))

    working = np.array(surface, dtype=np.float64, copy=True)
    rows, cols = working.shape
    peaks: List[Peak] = []

    for _ in range(max_peaks):
        flat_index = int(np.argmax(working))
        row, col = divmod(flat_index, cols)
        if not np.isfinite(working[row, col]):
            break
        peaks.append(Peak(score=float(surface[row, col]), row=row, col=col))
        working[
            max(0, row - radius) : min(rows, row + radius + 1),
            max(0, col - radius) : min(cols, col + radius + 1),
        ] = -np.inf

    return peaks


def _parabolic_offset(before: float, centre: float, after: float) -> float:
    """Sub-sample offset of a peak fitted through three consecutive samples.

    Fitting a parabola through the peak and its two neighbours gives the vertex
    at ``0.5 * (before - after) / (before - 2*centre + after)``. For a genuine
    maximum the denominator is negative; when it vanishes the samples are
    collinear and no correction is defined.
    """
    denominator = before - 2.0 * centre + after
    if denominator == 0.0 or not np.isfinite(denominator):
        return 0.0
    offset = 0.5 * (before - after) / denominator
    if not np.isfinite(offset):
        return 0.0
    # A correction beyond half a sample means the discrete peak was mislocated;
    # clamping keeps the refined position attached to the peak it came from.
    return float(np.clip(offset, -0.5, 0.5))


def refine_peak_subpixel(surface: np.ndarray, peak: Peak) -> Peak:
    """Refine a peak to sub-pixel precision by parabolic interpolation.

    The correlation surface is sampled on the integer grid, so an unrefined peak
    can only ever be accurate to +/- 0.5 px. Because the true offsets in this
    problem are sub-pixel by construction (crop origins in whole nanometres land
    on tenths of a search pixel), this step is what separates a locator that is
    right to the nearest pixel from one that is right to a tenth of one.

    Peaks on the boundary of the surface are returned unrefined.
    """
    rows, cols = surface.shape
    row_offset = 0.0
    col_offset = 0.0
    if 0 < peak.row < rows - 1:
        row_offset = _parabolic_offset(
            float(surface[peak.row - 1, peak.col]),
            float(surface[peak.row, peak.col]),
            float(surface[peak.row + 1, peak.col]),
        )
    if 0 < peak.col < cols - 1:
        col_offset = _parabolic_offset(
            float(surface[peak.row, peak.col - 1]),
            float(surface[peak.row, peak.col]),
            float(surface[peak.row, peak.col + 1]),
        )
    return Peak(
        score=peak.score,
        row=peak.row,
        col=peak.col,
        row_offset=row_offset,
        col_offset=col_offset,
    )


def peak_to_sidelobe_ratio(surface: np.ndarray, peak: Peak, exclusion_radius: int) -> float:
    """Confidence measure: how far the peak stands above the surrounding surface.

    Defined as ``(peak - mean(sidelobe)) / std(sidelobe)``, where the sidelobe is
    everything outside a square exclusion window around the peak. The measure is
    standard in correlation-filter tracking (Bolme et al., 2010) and is used here
    as a *self-reported ambiguity flag*: a low ratio means the surface has other
    peaks nearly as strong, which on a periodic layout is the correct description
    of a genuinely ambiguous case rather than a defect.

    Returns ``inf`` when the sidelobe is perfectly flat, and ``0`` when there is
    no sidelobe left to measure.
    """
    rows, cols = surface.shape
    radius = max(1, int(exclusion_radius))
    mask = np.ones(surface.shape, dtype=bool)
    mask[
        max(0, peak.row - radius) : min(rows, peak.row + radius + 1),
        max(0, peak.col - radius) : min(cols, peak.col + radius + 1),
    ] = False

    sidelobe = surface[mask]
    if sidelobe.size == 0:
        return 0.0
    spread = float(sidelobe.std())
    if spread == 0.0:
        return float("inf")
    return float((peak.score - sidelobe.mean()) / spread)


def weighted_normalised_cross_correlation(
    search: np.ndarray, template: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    """NCC in which template pixels contribute in proportion to ``weight``.

    WHY WEIGHTING IS THE RIGHT TOOL FOR PERIODIC AMBIGUITY
    ------------------------------------------------------
    Plain NCC treats every template pixel as equally informative. On a DRAM array
    that is badly wrong: the repeating interior looks the same at every lattice
    alias and contributes nothing to distinguishing them, while a mat boundary
    contributes everything. Because the interior is the overwhelming majority of
    the pixels, it dominates the score and drowns out the very features that
    resolve the ambiguity.

    Down-weighting the self-similar interior concentrates the score on the
    content that actually discriminates between candidate locations.

    The weighted statistics still factorise the way Lewis's formulation needs.
    With ``m = sum(w)`` over the window, the weighted window mean is
    ``corr(S, w) / m`` and the weighted second moment is ``corr(S^2, w) / m``, so
    the denominator costs two extra FFT correlations and the numerator one --
    three in total against one for the unweighted form.
    """
    S = np.asarray(search, dtype=np.float64)
    T = np.asarray(template, dtype=np.float64)
    W = np.asarray(weight, dtype=np.float64)
    if T.shape != W.shape:
        raise ValueError(f"weight shape {W.shape} does not match template {T.shape}")
    if np.any(W < 0):
        raise ValueError("weights must be non-negative")

    rows, cols = S.shape
    t_rows, t_cols = T.shape
    if t_rows > rows or t_cols > cols:
        raise ValueError("template does not fit inside search")

    total_weight = float(W.sum())
    if total_weight <= 0:
        raise ValueError("weights sum to zero")

    weighted_mean = float((W * T).sum() / total_weight)
    centred_template = T - weighted_mean
    template_energy = float((W * centred_template * centred_template).sum())
    if template_energy <= 0:
        raise ValueError("template has zero weighted contrast")

    S = S - S.mean()
    valid = (slice(0, rows - t_rows + 1), slice(0, cols - t_cols + 1))

    def correlate(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        spectrum = np.fft.rfft2(image)
        response = np.fft.rfft2(kernel, s=image.shape)
        return np.fft.irfft2(spectrum * np.conj(response), s=image.shape)[valid]

    numerator = correlate(S, W * centred_template)
    weighted_sum = correlate(S, W)
    weighted_square = correlate(S * S, W)

    variance = weighted_square - (weighted_sum * weighted_sum) / total_weight
    np.maximum(variance, 0.0, out=variance)
    denominator = np.sqrt(variance * template_energy)

    surface = np.zeros_like(numerator)
    largest = float(denominator.max())
    if largest > 0.0:
        np.divide(numerator, denominator, out=surface, where=denominator > largest * 1e-12)
    return np.clip(surface, -1.0, 1.0, out=surface)

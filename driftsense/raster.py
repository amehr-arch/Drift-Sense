"""Closed-form exact rasterisation of separable Manhattan layouts.

WHY THIS MODULE EXISTS
----------------------
The obvious way to build the search image is to raster the whole 10 um world at
1 nm/px (a 10000 x 10000 array, 400 MB in float32) and then box-downsample it by
10. That is memory-hungry and, more importantly, it is only an *approximation* of
what the wide-field capture physically does, which is to integrate the specimen
signal over each pixel footprint.

Semiconductor layouts are Manhattan (axis-aligned) and, in the array regions,
*separable*: the set of pixels covered by the bit lines is a product set
``X_set x Y_set``. For a product set the exact area of overlap with a pixel is the
product of the two one-dimensional overlaps. So the whole 2D coverage field can be
computed from two 1D coverage vectors, at any pixel size and any sub-pixel origin,
with negligible memory.

Concretely this renders the 10 nm/px search image directly, with coverage
integrated exactly over each 10 nm pixel footprint -- mathematically equivalent to
the render-then-downsample approach, but with a peak working set of 16 MB against
400 MB for a 10000x10000 float32 world, and with none of the resampling-kernel
ambiguity.

WHY CLOSED FORM RATHER THAN SUPERSAMPLING
-----------------------------------------
An alternative approach computes the 1D vectors by supersampling each output pixel 64
times and averaging, which handles arbitrary set algebra but quantises edge-pixel
coverage to 1/64. That quantisation was measurable: it left a residual of up to
0.79 of an 8-bit grey level between a fine render of a region and a coarse render
of the same region, because the coarse render's larger pixels carry the full
quantisation error while the fine render's averages down over a hundred pixels.

The sets actually needed here are unions of clipped periodic stripes, and for
those the covered length has a closed form. Integrating it exactly removes the
quantisation entirely -- the residual falls to floating-point noise -- and is also
faster, since no 64x oversampled array is ever built.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "cumulative_stripe_length",
    "periodic_stripe_coverage",
    "outer_coverage",
]


def cumulative_stripe_length(
    positions: np.ndarray, period_nm: float, width_nm: float, phase_nm: float = 0.0
) -> np.ndarray:
    """Total stripe length in ``(-inf, u]`` for an infinite periodic stripe set.

    Stripe ``k`` occupies ``[k*period + phase, k*period + phase + width)``. Writing
    ``t = u - phase``, ``n = floor(t / period)`` and ``r = t - n*period``, the
    length accumulated by ``u`` is ``n*width`` from the completed periods plus
    ``clip(r, 0, width)`` from the partial one.

    Coverage of any interval follows by differencing, which is what makes exact
    area integration possible at any pixel size.
    """
    if period_nm <= 0:
        raise ValueError("period_nm must be positive")
    width = float(np.clip(width_nm, 0.0, period_nm))
    if width == 0.0:
        return np.zeros_like(np.asarray(positions, dtype=np.float64))

    offset = np.asarray(positions, dtype=np.float64) - phase_nm
    completed = np.floor(offset / period_nm)
    remainder = offset - completed * period_nm
    return completed * width + np.clip(remainder, 0.0, width)


def periodic_stripe_coverage(
    n_px: int,
    pixel_size_nm: float,
    origin_nm: float,
    period_nm: float,
    width_nm: float,
    phase_nm: float = 0.0,
    lo_nm: float | None = None,
    hi_nm: float | None = None,
) -> np.ndarray:
    """Exact per-pixel coverage of a periodic stripe set, optionally clipped.

    Parameters
    ----------
    n_px, pixel_size_nm, origin_nm:
        The output sampling grid. Pixel ``k`` spans
        ``[origin + k*pixel_size, origin + (k+1)*pixel_size)``.
    period_nm, width_nm, phase_nm:
        The stripe set.
    lo_nm, hi_nm:
        Optional half-open clipping interval. Only the part of the stripe set
        inside ``[lo, hi)`` contributes, which is how a mat's own lattice is
        confined to that mat.

    Returns
    -------
    float32 array of length ``n_px`` with values in ``[0, 1]``.
    """
    if n_px <= 0:
        raise ValueError("n_px must be positive")
    if pixel_size_nm <= 0:
        raise ValueError("pixel_size_nm must be positive")

    edges = origin_nm + np.arange(n_px + 1, dtype=np.float64) * pixel_size_nm
    if lo_nm is not None or hi_nm is not None:
        low = -np.inf if lo_nm is None else lo_nm
        high = np.inf if hi_nm is None else hi_nm
        if high < low:
            return np.zeros(n_px, dtype=np.float32)
        edges = np.clip(edges, low, high)

    cumulative = cumulative_stripe_length(edges, period_nm, width_nm, phase_nm)
    covered = np.diff(cumulative)
    return (np.clip(covered, 0.0, pixel_size_nm) / pixel_size_nm).astype(np.float32)


def outer_coverage(coverage_y: np.ndarray, coverage_x: np.ndarray) -> np.ndarray:
    """Build the 2D coverage field of the product set ``X_set x Y_set``.

    For an axis-aligned product set the area of overlap with a pixel factorises
    exactly into the product of the two 1D overlaps, so this is not an
    approximation.

    Returns
    -------
    float32 array of shape ``(len(coverage_y), len(coverage_x))``, indexed
    ``[row, col] == [y, x]``.
    """
    return np.multiply.outer(
        np.asarray(coverage_y, dtype=np.float32), np.asarray(coverage_x, dtype=np.float32)
    ).astype(np.float32)

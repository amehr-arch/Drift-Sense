"""Resampling primitives: area-average reduction and bilinear sampling.

WHY THESE ARE HAND-WRITTEN RATHER THAN IMPORTED
-----------------------------------------------
Both operations exist in ``scipy.ndimage``. They are implemented here in plain
NumPy for two reasons.

First, dependency surface. The one failure mode the problem statement calls fatal
is an inference script that will not run on the reviewer's machine. Every
dependency removed is a class of installation failure removed, and these two
functions are the only reason ``scipy`` would have been needed.

Second, and more importantly, *correctness of the reduction*. The wide-field
capture integrates the specimen signal over each pixel footprint, so reducing the
reference by the zoom ratio must be an exact area average -- not a spline
interpolation, and not a decimation. ``area_average_reduce`` implements the box
integral directly via cumulative sums, which is exact for both integer and
fractional factors. Getting this wrong introduces a systematic bias between the
template and the search content that no amount of algorithmic sophistication
downstream can recover.

The zoom ratio is fixed at 10 by the problem statement, so in practice the
integer fast path is what runs; the general path exists so that scale search at
an earlier revision (which must try factors like 9.4 or 10.7) uses the same exact integral.
"""

from __future__ import annotations

import numpy as np

__all__ = ["area_average_reduce", "area_average_reduce_1d", "bilinear_sample", "sample_patch"]


def area_average_reduce_1d(values: np.ndarray, factor: float, axis: int = -1) -> np.ndarray:
    """Reduce one axis by ``factor`` using an exact box integral.

    Output element ``k`` is the mean of the input over the continuous span
    ``[k * factor, (k + 1) * factor)``. The integral of the piecewise-constant
    input is obtained from a cumulative sum and evaluated at fractional
    positions, so partially covered input pixels contribute their exact overlap.
    """
    if factor <= 0:
        raise ValueError("factor must be positive")
    data = np.moveaxis(np.asarray(values, dtype=np.float64), axis, -1)
    length = data.shape[-1]
    n_out = int(np.floor(length / factor + 1e-9))
    if n_out < 1:
        raise ValueError(f"factor {factor} is too large for length {length}")

    # cumulative[..., i] = integral of the signal over [0, i]
    cumulative = np.concatenate(
        [np.zeros(data.shape[:-1] + (1,)), np.cumsum(data, axis=-1)], axis=-1
    )
    edges = np.arange(n_out + 1, dtype=np.float64) * factor
    whole = np.floor(edges).astype(np.intp)
    frac = edges - whole
    whole_clipped = np.clip(whole, 0, length)
    # Value of the pixel the edge falls inside, for the partial contribution.
    inside = np.clip(whole, 0, length - 1)

    integral = np.take(cumulative, whole_clipped, axis=-1) + frac * np.take(
        data, inside, axis=-1
    )
    reduced = np.diff(integral, axis=-1) / factor
    return np.moveaxis(reduced, -1, axis)


def area_average_reduce(image: np.ndarray, factor: float) -> np.ndarray:
    """Reduce a 2-D image by ``factor`` on both axes using exact area averaging.

    An integer factor that divides both dimensions takes a reshape-and-mean fast
    path, which is bit-for-bit the same result as the general path but avoids
    building the cumulative sums.
    """
    if image.ndim != 2:
        raise ValueError("image must be two-dimensional")
    if factor <= 0:
        raise ValueError("factor must be positive")

    height, width = image.shape
    rounded = int(round(factor))
    if abs(factor - rounded) < 1e-9 and rounded >= 1 and height % rounded == 0 and width % rounded == 0:
        data = image.astype(np.float64)
        return data.reshape(height // rounded, rounded, width // rounded, rounded).mean(axis=(1, 3))

    reduced = area_average_reduce_1d(image, factor, axis=0)
    return area_average_reduce_1d(reduced, factor, axis=1)


def bilinear_sample(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Sample ``image`` at fractional array indices with bilinear interpolation.

    ``xs`` and ``ys`` are *array index* coordinates (index 0 is the first pixel),
    not the continuous coordinates of :mod:`driftsense.geometry`. Out-of-range
    samples clamp to the border, which avoids fabricating structure beyond the
    frame edge.
    """
    if image.ndim != 2:
        raise ValueError("image must be two-dimensional")
    data = image.astype(np.float64)
    height, width = data.shape

    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    x0 = np.floor(xs)
    y0 = np.floor(ys)
    fx = xs - x0
    fy = ys - y0

    x0i = np.clip(x0.astype(np.intp), 0, width - 1)
    x1i = np.clip(x0.astype(np.intp) + 1, 0, width - 1)
    y0i = np.clip(y0.astype(np.intp), 0, height - 1)
    y1i = np.clip(y0.astype(np.intp) + 1, 0, height - 1)

    top = data[y0i, x0i] * (1.0 - fx) + data[y0i, x1i] * fx
    bottom = data[y1i, x0i] * (1.0 - fx) + data[y1i, x1i] * fx
    return top * (1.0 - fy) + bottom * fy


def sample_patch(image: np.ndarray, box_x: float, box_y: float, size: int) -> np.ndarray:
    """Sample a ``size x size`` patch whose top-left corner is at ``(box_x, box_y)``.

    Continuous coordinate ``x`` corresponds to array index ``x - 0.5`` under the
    convention fixed in :mod:`driftsense.geometry`, so the centre of output cell
    ``b`` -- which spans ``[box_x + b, box_x + b + 1)`` and is therefore centred at
    ``box_x + b + 0.5`` -- lands on array index ``box_x + b``.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    cols = box_x + np.arange(size, dtype=np.float64)
    rows = box_y + np.arange(size, dtype=np.float64)
    grid_rows, grid_cols = np.meshgrid(rows, cols, indexing="ij")
    return bilinear_sample(image, grid_cols, grid_rows)

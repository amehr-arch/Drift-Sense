"""Verification panels: look at the data before trusting it.

The an earlier revision exit gate is a human one. The panel produced here places, side by
side, the search image with the ground-truth box drawn on it, the reference as it
should appear after the 10x reduction, the search content actually found at the
ground-truth location, and the absolute difference between the two. If the
geometry is correct the middle two tiles are visually indistinguishable and the
difference tile is flat.

This is the cheapest possible defence against a whole class of coordinate bugs
that are otherwise invisible until they cap accuracy three stages later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .generate import GeneratedPair
from .validate import block_reduce, sample_patch, zncc

__all__ = [
    "make_verification_panel",
    "save_verification_panel",
    "draw_ground_truth_box",
    "render_accuracy_curve",
    "make_match_panel",
    "save_match_panel",
]

_TILE = 300  # display size of each magnified 100 px tile
_MARGIN = 12
_LABEL_H = 16


def draw_ground_truth_box(
    search: np.ndarray, box: Tuple[float, float, float, float], width: int = 2
) -> Image.Image:
    """Return an RGB copy of the search image with the match box and crosshair drawn."""
    canvas = Image.fromarray(search, mode="L").convert("RGB")
    draw = ImageDraw.Draw(canvas)
    x, y, w, h = box
    draw.rectangle([x, y, x + w, y + h], outline=(255, 210, 60), width=width)
    cx, cy = x + w / 2.0, y + h / 2.0
    arm = max(8.0, w * 0.28)
    draw.line([cx - arm, cy, cx + arm, cy], fill=(255, 90, 70), width=width)
    draw.line([cx, cy - arm, cx, cy + arm], fill=(255, 90, 70), width=width)
    return canvas


def _tile(array: np.ndarray, label: str) -> Image.Image:
    """Magnify a small float array to a labelled display tile."""
    finite = np.nan_to_num(array.astype(np.float64))
    lo, hi = float(finite.min()), float(finite.max())
    spread = hi - lo
    normalised = (finite - lo) / spread if spread > 1e-12 else np.zeros_like(finite)
    image = Image.fromarray((normalised * 255).astype(np.uint8), mode="L").convert("RGB")
    image = image.resize((_TILE, _TILE), Image.NEAREST)

    framed = Image.new("RGB", (_TILE, _TILE + _LABEL_H), (18, 18, 20))
    framed.paste(image, (0, _LABEL_H))
    ImageDraw.Draw(framed).text((2, 3), label, fill=(230, 230, 235))
    return framed


def make_verification_panel(pair: GeneratedPair) -> Image.Image:
    """Build the full verification panel for one pair."""
    geometry = pair.geometry
    gt = pair.ground_truth
    side = int(round(geometry.template_size_px))

    template = block_reduce(pair.reference, geometry.zoom_ratio)
    patch = sample_patch(pair.search, gt.box_x, gt.box_y, side)
    difference = np.abs(template - patch)
    score = zncc(template, patch)

    search_view = draw_ground_truth_box(pair.search, gt.box)

    tiles = [
        _tile(template, "reference / 10"),
        _tile(patch, "search @ ground truth"),
        _tile(difference, f"abs difference (max {difference.max():.1f})"),
    ]

    column_h = sum(tile.height for tile in tiles) + _MARGIN * (len(tiles) - 1)
    height = max(search_view.height, column_h) + _LABEL_H + _MARGIN
    width = search_view.width + _MARGIN + _TILE + _MARGIN

    panel = Image.new("RGB", (width, height), (18, 18, 20))
    panel.paste(search_view, (0, _LABEL_H))

    y = _LABEL_H
    for tile in tiles:
        panel.paste(tile, (search_view.width + _MARGIN, y))
        y += tile.height + _MARGIN

    caption = (
        f"{pair.name}   gt=({gt.x:.2f}, {gt.y:.2f}) px   "
        f"box={side}x{side} px   ZNCC={score:.4f}   "
        f"origin=({pair.placement.origin_x_nm:.1f}, {pair.placement.origin_y_nm:.1f}) nm"
    )
    ImageDraw.Draw(panel).text((2, 3), caption, fill=(235, 235, 240))
    return panel


def save_verification_panel(pair: GeneratedPair, path: Path) -> Path:
    """Write the verification panel for ``pair`` to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_verification_panel(pair).save(path)
    return path


# ---------------------------------------------------------------------------
# Evaluation visuals
# ---------------------------------------------------------------------------

_CURVE_W, _CURVE_H = 760, 430
_PLOT_PAD = (70, 30, 24, 56)  # left, top, right, bottom


def render_accuracy_curve(summary, path: Path) -> Path:
    """Draw the accuracy-versus-tolerance curve as a PNG.

    Tolerances are placed at even spacing rather than on a linear axis. They span
    two orders of magnitude, so a linear axis would compress everything below 5 px
    into the left margin -- which is precisely the region that decides the score.
    """
    tolerances = list(summary.tolerances)
    values = [summary.accuracy[f"{t:g}"] * 100.0 for t in tolerances]

    left, top, right, bottom = _PLOT_PAD
    width, height = _CURVE_W, _CURVE_H
    plot_w = width - left - right
    plot_h = height - top - bottom

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text((left - 46, 10), "accuracy (%) vs error tolerance (px)", fill=(30, 30, 40))

    # horizontal grid and y labels
    for percent in range(0, 101, 25):
        y = top + plot_h - plot_h * percent / 100.0
        draw.line([left, y, left + plot_w, y], fill=(226, 230, 236))
        draw.text((left - 34, y - 6), f"{percent:>3}%", fill=(110, 115, 125))

    draw.line([left, top, left, top + plot_h], fill=(90, 95, 105))
    draw.line([left, top + plot_h, left + plot_w, top + plot_h], fill=(90, 95, 105))

    if len(tolerances) > 1:
        step = plot_w / (len(tolerances) - 1)
    else:
        step = 0.0
    points = []
    for index, (tolerance, value) in enumerate(zip(tolerances, values)):
        x = left + index * step
        y = top + plot_h - plot_h * value / 100.0
        points.append((x, y))
        draw.line([x, top, x, top + plot_h], fill=(240, 243, 247))
        label = f"{tolerance:g}"
        draw.text((x - 3 * len(label), top + plot_h + 8), label, fill=(110, 115, 125))

    if len(points) > 1:
        draw.line(points, fill=(31, 78, 121), width=3, joint="curve")
    for (x, y), value in zip(points, values):
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(31, 78, 121))
        draw.text((x - 12, y - 20), f"{value:.0f}", fill=(31, 78, 121))

    draw.text(
        (left, height - 18),
        f"n = {summary.n_pairs}   median error {summary.median_error_px:.3f} px   "
        f"median time {summary.median_elapsed_s * 1000:.0f} ms/pair",
        fill=(110, 115, 125),
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def make_match_panel(
    reference: np.ndarray,
    search: np.ndarray,
    result,
    ground_truth: Optional[Tuple[float, float]] = None,
    zoom_ratio: float = 10.0,
) -> Image.Image:
    """Show what the locator decided and, when known, what it should have decided.

    The prediction is drawn in blue and the ground truth, if supplied, in amber.
    Every shortlisted candidate is marked, because on an ambiguous pair the useful
    information is not where the locator landed but how many other places scored
    almost as well.
    """
    from .resample import area_average_reduce, sample_patch

    side = int(round(reference.shape[0] / zoom_ratio))
    canvas = Image.fromarray(np.asarray(search, dtype=np.uint8), mode="L").convert("RGB")
    draw = ImageDraw.Draw(canvas)

    for candidate in getattr(result, "candidates", []):
        draw.ellipse(
            [candidate.x - 3, candidate.y - 3, candidate.x + 3, candidate.y + 3],
            outline=(120, 200, 255),
            width=1,
        )

    if ground_truth is not None:
        gx, gy = ground_truth
        draw.rectangle(
            [gx - side / 2, gy - side / 2, gx + side / 2, gy + side / 2],
            outline=(255, 200, 60),
            width=2,
        )

    draw.rectangle(
        [result.x - side / 2, result.y - side / 2, result.x + side / 2, result.y + side / 2],
        outline=(70, 150, 255),
        width=2,
    )
    draw.line([result.x - 14, result.y, result.x + 14, result.y], fill=(255, 90, 70), width=2)
    draw.line([result.x, result.y - 14, result.x, result.y + 14], fill=(255, 90, 70), width=2)

    template = area_average_reduce(reference, zoom_ratio)
    predicted = sample_patch(search, result.x - side / 2, result.y - side / 2, side)
    tiles = [_tile(template, "reference / 10"), _tile(predicted, "search @ prediction")]
    if ground_truth is not None:
        gx, gy = ground_truth
        tiles.append(
            _tile(sample_patch(search, gx - side / 2, gy - side / 2, side), "search @ truth")
        )

    column_h = sum(tile.height for tile in tiles) + _MARGIN * (len(tiles) - 1)
    height = max(canvas.height, column_h) + _LABEL_H + _MARGIN
    panel = Image.new("RGB", (canvas.width + _MARGIN + _TILE + _MARGIN, height), (18, 18, 20))
    panel.paste(canvas, (0, _LABEL_H))

    y = _LABEL_H
    for tile in tiles:
        panel.paste(tile, (canvas.width + _MARGIN, y))
        y += tile.height + _MARGIN

    caption = (
        f"predicted=({result.x:.2f}, {result.y:.2f})  score={result.score:.4f}  "
        f"psr={result.confidence:.2f}  candidates={len(getattr(result, 'candidates', []))}"
    )
    if ground_truth is not None:
        error = float(np.hypot(result.x - ground_truth[0], result.y - ground_truth[1]))
        caption += f"  truth=({ground_truth[0]:.2f}, {ground_truth[1]:.2f})  error={error:.3f} px"
    ImageDraw.Draw(panel).text((2, 3), caption, fill=(235, 235, 240))
    return panel


def save_match_panel(
    reference: np.ndarray,
    search: np.ndarray,
    result,
    path: Path,
    ground_truth: Optional[Tuple[float, float]] = None,
    zoom_ratio: float = 10.0,
) -> Path:
    """Write a match panel to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_match_panel(reference, search, result, ground_truth, zoom_ratio).save(path)
    return path

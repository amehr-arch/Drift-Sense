"""Degradation studies: accuracy as a function of one imaging term at a time.

WHY ONE TERM AT A TIME
----------------------
The imaging model has a dozen knobs. Turning them all up together tells you only
that the images got worse. Sweeping one at a time, with everything else held at a
fixed operating point, tells you *which* term costs accuracy and where the cliff
is -- which is the question that matters when deciding what the locator must be
made robust to later.

The problem statement asks for exactly this study: accuracy against increasing
degradation, and identifies where the method breaks.

WHAT IS HELD FIXED
------------------
The same layouts and the same crop placements are used at every level of the
sweep, drawn from a fixed seed. Only the swept parameter changes. Without that,
a difference between two levels could be the parameter or could be a different
draw of ambiguous placements, and the two would be indistinguishable.

Accuracy is reported on the *solvable* subset -- pairs whose reference window
carries a structural anchor on both axes. Pairs without one are ambiguous
regardless of noise, so including them would add a large constant offset that
hides the effect being measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .generate import generate_pair
from .geometry import ImagingGeometry
from .imaging import CaptureParams
from .locate import LocalisationConfig, locate
from .sampling import (
    CaptureRanges,
    DramParamRanges,
    PlacementSampler,
    sample_captures,
    sample_dram_layout,
)

__all__ = ["SweepPoint", "sweep_parameter", "format_sweep"]


@dataclass
class SweepPoint:
    """Accuracy at one level of a swept parameter."""

    value: float
    n_solvable: int
    median_error_px: float
    within_1px: float
    within_5px: float
    median_margin: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "value": self.value,
            "n_solvable": self.n_solvable,
            "median_error_px": round(self.median_error_px, 4),
            "within_1px": round(self.within_1px, 4),
            "within_5px": round(self.within_5px, 4),
            "median_margin": round(self.median_margin, 6),
        }


def sweep_parameter(
    parameter: str,
    values: Sequence[float],
    n_pairs: int = 12,
    seed: int = 4242,
    apply_to: str = "search",
    geometry: Optional[ImagingGeometry] = None,
    locator: Optional[LocalisationConfig] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[SweepPoint]:
    """Measure localisation accuracy across levels of one capture parameter.

    Parameters
    ----------
    parameter:
        Field name on :class:`~driftsense.imaging.CaptureParams`.
    values:
        Levels to sweep.
    apply_to:
        ``"search"``, ``"reference"`` or ``"both"`` -- which capture the swept
        parameter is varied on.
    """
    geometry = geometry or ImagingGeometry()
    if not hasattr(CaptureParams(), parameter):
        raise AttributeError(f"CaptureParams has no field {parameter!r}")
    if apply_to not in ("search", "reference", "both"):
        raise ValueError("apply_to must be 'search', 'reference' or 'both'")

    # Fix the scene: same layouts, same placements, at every level.
    scenes = []
    for child in np.random.SeedSequence(seed).spawn(n_pairs):
        rng = np.random.default_rng(child)
        layout = sample_dram_layout(rng, DramParamRanges(), geometry)
        placement = PlacementSampler(geometry=geometry, boundary_bias=0.9).sample(layout, rng)
        base_reference, base_search = sample_captures(rng, CaptureRanges())
        scenes.append((layout, placement, base_reference, base_search))

    results: List[SweepPoint] = []
    total = len(values) * len(scenes)
    done = 0

    for value in values:
        errors: List[float] = []
        margins: List[float] = []
        for index, (layout, placement, base_reference, base_search) in enumerate(scenes):
            reference = (
                base_reference.with_changes(**{parameter: value})
                if apply_to in ("reference", "both")
                else base_reference
            )
            search = (
                base_search.with_changes(**{parameter: value})
                if apply_to in ("search", "both")
                else base_search
            )
            pair = generate_pair(
                layout,
                placement,
                geometry,
                pair_id=index,
                seed=index,
                reference_capture=reference,
                search_capture=search,
                rng=np.random.default_rng(1000 + index),
            )
            done += 1
            if progress is not None:
                progress(done, total)
            if pair.anchor != "both":
                continue  # ambiguous regardless of noise; see module docstring
            outcome = locate(pair.reference, pair.search, locator)
            errors.append(
                float(np.hypot(outcome.x - pair.ground_truth.x, outcome.y - pair.ground_truth.y))
            )
            margins.append(outcome.runner_up_margin or 0.0)

        if not errors:
            raise RuntimeError(
                "no solvable pairs in the sweep scene; raise boundary_bias or n_pairs"
            )
        results.append(
            SweepPoint(
                value=float(value),
                n_solvable=len(errors),
                median_error_px=float(np.median(errors)),
                within_1px=float(np.mean([e <= 1.0 for e in errors])),
                within_5px=float(np.mean([e <= 5.0 for e in errors])),
                median_margin=float(np.median(margins)),
            )
        )
    return results


def format_sweep(parameter: str, points: Sequence[SweepPoint]) -> str:
    """Render a sweep as a plain-text table."""
    lines = [
        "",
        f"  {parameter}",
        f"    {'value':>10}{'n':>5}{'median err':>13}{'<=1px':>9}{'<=5px':>9}{'margin':>10}",
    ]
    for point in points:
        lines.append(
            f"    {point.value:>10.4g}{point.n_solvable:>5}"
            f"{point.median_error_px:>12.4f}px{point.within_1px * 100:>8.1f}%"
            f"{point.within_5px * 100:>8.1f}%{point.median_margin:>10.5f}"
        )
    return "\n".join(lines)

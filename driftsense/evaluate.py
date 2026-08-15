"""Evaluation harness: run the locator over a dataset and report accuracy.

PURPOSE
-------------------------------------------------
The harness is built later, alongside the simplest locator that could work,
rather than later when there is something impressive to measure. Without a
scoreboard, every subsequent change to the algorithm is a matter of opinion. With
one, each of the seven increments planned for an earlier revision can be measured
independently and kept or discarded on evidence.

WHAT IS REPORTED, AND WHY
-------------------------
Not a single accuracy number. The problem statement scores localisation as a
precision/recall style study across pixel-error thresholds -- "is the prediction
within 1 px, within 2, within 5" -- because the tolerance that matters depends on
the application. So the primary output is an accuracy-versus-tolerance curve.

Computation time is reported alongside it, since the organisers use it as the
tiebreak between submissions of equal accuracy. Timing covers the algorithm only;
image loading is excluded, because that measures the disk rather than the method.

Two diagnostic columns are carried through to the per-pair results: the
peak-to-sidelobe confidence and the margin over the runner-up candidate. Together
they let a failure be classified without re-running anything -- a large error with
a small margin is periodic ambiguity, a large error with a large margin is
something else and more interesting.
"""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .locate import LocalisationConfig, load_greyscale, locate

__all__ = [
    "DEFAULT_TOLERANCES",
    "PairResult",
    "EvaluationSummary",
    "evaluate_dataset",
    "write_evaluation",
    "format_report",
]

#: Error thresholds, in search-image pixels, at which accuracy is reported.
#: One search pixel is 10 nm of wafer, so 1 px is already a demanding target.
DEFAULT_TOLERANCES: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0)

RESULT_COLUMNS = [
    "pair_id",
    "anchor",
    "gt_x",
    "gt_y",
    "pred_x",
    "pred_y",
    "error_px",
    "error_x",
    "error_y",
    "score",
    "confidence_psr",
    "runner_up_margin",
    "tie_broken",
    "elapsed_s",
]


@dataclass
class PairResult:
    """Outcome for a single pair."""

    pair_id: int
    anchor: str
    gt_x: float
    gt_y: float
    pred_x: float
    pred_y: float
    error_px: float
    error_x: float
    error_y: float
    score: float
    confidence_psr: float
    runner_up_margin: Optional[float]
    tie_broken: bool
    elapsed_s: float

    def as_row(self) -> Dict[str, object]:
        row = asdict(self)
        for key in ("gt_x", "gt_y", "pred_x", "pred_y", "error_px", "error_x", "error_y"):
            row[key] = round(float(row[key]), 5)
        row["score"] = round(float(row["score"]), 6)
        row["confidence_psr"] = round(float(row["confidence_psr"]), 4)
        row["runner_up_margin"] = (
            "" if row["runner_up_margin"] is None else round(float(row["runner_up_margin"]), 6)
        )
        row["elapsed_s"] = round(float(row["elapsed_s"]), 6)
        return row


@dataclass
class EvaluationSummary:
    """Aggregate metrics over a dataset."""

    n_pairs: int
    tolerances: Tuple[float, ...]
    accuracy: Dict[str, float]  # tolerance (as string) -> fraction within it
    mean_error_px: float
    median_error_px: float
    p95_error_px: float
    max_error_px: float
    mean_elapsed_s: float
    median_elapsed_s: float
    total_elapsed_s: float
    n_tie_broken: int
    by_anchor: Dict[str, Dict[str, float]] = field(default_factory=dict)
    worst_pairs: List[Dict[str, float]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["tolerances"] = list(self.tolerances)
        return payload


def _accuracy_at(errors: Sequence[float], tolerance: float) -> float:
    if not errors:
        return 0.0
    return float(sum(1 for e in errors if e <= tolerance) / len(errors))


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), fraction * 100.0))


def evaluate_dataset(
    dataset_dir: str | Path,
    config: LocalisationConfig | None = None,
    tolerances: Sequence[float] = DEFAULT_TOLERANCES,
    limit: Optional[int] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[List[PairResult], EvaluationSummary]:
    """Run the locator over every pair in a generated dataset.

    Parameters
    ----------
    dataset_dir:
        Directory containing ``ground_truth.csv`` and a ``pairs/`` subdirectory,
        as written by ``generate_dataset.py``.
    limit:
        Evaluate only the first *n* pairs. Useful while iterating.

    Returns
    -------
    ``(per-pair results, summary)``.
    """
    dataset_dir = Path(dataset_dir)
    csv_path = dataset_dir / "ground_truth.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"no ground_truth.csv in {dataset_dir}")

    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"{csv_path} contains no rows")

    results: List[PairResult] = []
    for index, row in enumerate(rows):
        reference = load_greyscale(dataset_dir / row["reference_path"])
        search = load_greyscale(dataset_dir / row["search_path"])
        outcome = locate(reference, search, config)

        gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])
        error_x = outcome.x - gt_x
        error_y = outcome.y - gt_y
        results.append(
            PairResult(
                pair_id=int(row["pair_id"]),
                anchor=row.get("anchor", "unknown"),
                gt_x=gt_x,
                gt_y=gt_y,
                pred_x=outcome.x,
                pred_y=outcome.y,
                error_px=float(np.hypot(error_x, error_y)),
                error_x=error_x,
                error_y=error_y,
                score=outcome.score,
                confidence_psr=outcome.confidence,
                runner_up_margin=outcome.runner_up_margin,
                tie_broken=outcome.tie_broken,
                elapsed_s=outcome.elapsed_s,
            )
        )
        if progress is not None:
            progress(index + 1, len(rows))

    errors = [r.error_px for r in results]
    times = [r.elapsed_s for r in results]
    worst = sorted(results, key=lambda r: r.error_px, reverse=True)[:5]

    # Split by difficulty class. An axis without a structural anchor is periodic
    # within the reference window, so its position is only recoverable up to one
    # cell pitch -- reporting those pairs mixed in with the rest would hide both
    # the real accuracy and the real limitation.
    by_anchor: Dict[str, Dict[str, float]] = {}
    for label in sorted({r.anchor for r in results}):
        subset = [r.error_px for r in results if r.anchor == label]
        by_anchor[label] = {
            "n": len(subset),
            "median_error_px": float(statistics.median(subset)),
            "mean_error_px": float(statistics.fmean(subset)),
            "within_1px": _accuracy_at(subset, 1.0),
            "within_5px": _accuracy_at(subset, 5.0),
        }

    summary = EvaluationSummary(
        n_pairs=len(results),
        tolerances=tuple(tolerances),
        accuracy={f"{t:g}": _accuracy_at(errors, t) for t in tolerances},
        mean_error_px=float(statistics.fmean(errors)),
        median_error_px=float(statistics.median(errors)),
        p95_error_px=_percentile(errors, 0.95),
        max_error_px=float(max(errors)),
        mean_elapsed_s=float(statistics.fmean(times)),
        median_elapsed_s=float(statistics.median(times)),
        total_elapsed_s=float(sum(times)),
        n_tie_broken=sum(1 for r in results if r.tie_broken),
        by_anchor=by_anchor,
        worst_pairs=[
            {
                "pair_id": r.pair_id,
                "anchor": r.anchor,
                "error_px": round(r.error_px, 4),
                "confidence_psr": round(r.confidence_psr, 3),
                "runner_up_margin": (
                    None if r.runner_up_margin is None else round(r.runner_up_margin, 5)
                ),
            }
            for r in worst
        ],
    )
    return results, summary


def format_report(summary: EvaluationSummary) -> str:
    """Render the summary as a plain-text report."""
    lines = [
        "",
        "  Drift-Sense localisation evaluation",
        "  " + "-" * 52,
        f"  pairs evaluated        {summary.n_pairs}",
        "",
        "  accuracy within tolerance",
    ]
    for tolerance in summary.tolerances:
        fraction = summary.accuracy[f"{tolerance:g}"]
        bar = "#" * int(round(fraction * 30))
        lines.append(f"    <= {tolerance:>6g} px   {fraction * 100:6.2f}%  {bar}")
    lines += [
        "",
        "  error (px)",
        f"    mean                 {summary.mean_error_px:.4f}",
        f"    median               {summary.median_error_px:.4f}",
        f"    95th percentile      {summary.p95_error_px:.4f}",
        f"    max                  {summary.max_error_px:.4f}",
        "",
        "  computation time",
        f"    mean per pair        {summary.mean_elapsed_s * 1000:.1f} ms",
        f"    median per pair      {summary.median_elapsed_s * 1000:.1f} ms",
        f"    total                {summary.total_elapsed_s:.2f} s",
        "",
        f"  tiebreak invoked       {summary.n_tie_broken} of {summary.n_pairs}",
    ]
    if summary.by_anchor:
        lines += [
            "",
            "  by anchor class  (an axis without a structural anchor is periodic",
            "  within the reference window and recoverable only to one cell pitch)",
            f"    {'class':<8}{'n':>5}{'median err':>13}{'<=1px':>9}{'<=5px':>9}",
        ]
        for label in ("both", "x", "y", "none", "unknown"):
            entry = summary.by_anchor.get(label)
            if not entry:
                continue
            lines.append(
                f"    {label:<8}{int(entry['n']):>5}{entry['median_error_px']:>12.4f}px"
                f"{entry['within_1px'] * 100:>8.1f}%{entry['within_5px'] * 100:>8.1f}%"
            )
    if summary.worst_pairs:
        lines += ["", "  worst pairs by error"]
        for entry in summary.worst_pairs:
            margin = entry["runner_up_margin"]
            margin_text = "n/a" if margin is None else f"{margin:.5f}"
            lines.append(
                f"    pair {entry['pair_id']:>4}  anchor {str(entry['anchor']):<5}"
                f"  error {entry['error_px']:>9.4f} px"
                f"   psr {entry['confidence_psr']:>7.3f}   margin {margin_text}"
            )
    lines.append("")
    return "\n".join(lines)


def write_evaluation(
    output_dir: str | Path,
    results: Sequence[PairResult],
    summary: EvaluationSummary,
    plot: bool = True,
    dataset_dir: str | Path | None = None,
) -> Dict[str, Path]:
    """Write per-pair results, the summary and the accuracy curve to disk.

    When ``dataset_dir`` is supplied the failure taxonomy is written too. It
    needs the dataset rather than just the results, because classifying a failure
    requires the layout pitches and capture settings that produced the pair.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(result.as_row() for result in results)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary.as_dict(), indent=2), encoding="utf-8")

    report_path = output_dir / "report.txt"
    report_path.write_text(format_report(summary), encoding="utf-8")

    written = {"results": results_path, "summary": summary_path, "report": report_path}

    if dataset_dir is not None:
        from .failures import classify_dataset, format_taxonomy_report, summarise_modes

        verdicts = classify_dataset(dataset_dir, results)

        taxonomy_path = output_dir / "failure_modes.csv"
        with taxonomy_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "pair_id",
                    "mode",
                    "error_px",
                    "anchor",
                    "evidence",
                    "flagged_low_confidence",
                    "confidence_psr",
                    "runner_up_margin",
                    "detail",
                ],
            )
            writer.writeheader()
            writer.writerows(v.as_row() for v in verdicts)
        written["taxonomy"] = taxonomy_path

        modes_path = output_dir / "failure_modes.json"
        modes_path.write_text(
            json.dumps(summarise_modes(verdicts), indent=2), encoding="utf-8"
        )
        written["modes"] = modes_path

        taxonomy_report = output_dir / "failure_report.txt"
        taxonomy_report.write_text(format_taxonomy_report(verdicts), encoding="utf-8")
        written["failures"] = taxonomy_report

    if plot:
        from .visualise import render_accuracy_curve

        curve_path = output_dir / "accuracy_curve.png"
        render_accuracy_curve(summary, curve_path)
        written["curve"] = curve_path

    return written

#!/usr/bin/env python3
"""Evaluate the locator against a generated dataset.

Runs ``driftsense.locate`` over every pair listed in a dataset's
``ground_truth.csv``, then reports accuracy as a function of error tolerance
together with computation time.

Examples
--------
    python evaluate_dataset.py data/dram_stage1
    python evaluate_dataset.py data/dram_stage1 --out results/stage2 --panels 5
    python evaluate_dataset.py data/dram_stage1 --limit 5 --no-subpixel

Outputs (written to ``<dataset>/evaluation`` unless ``--out`` is given)
----------------------------------------------------------------------
    results.csv          one row per pair: prediction, error, score, timing
    summary.json         aggregate metrics
    report.txt           the same report printed to the console
    accuracy_curve.png   accuracy versus error tolerance
    failure_modes.csv    one row per pair: root-cause mode and the evidence for it
    failure_modes.json   counts and confidence signature per mode
    failure_report.txt   the taxonomy as a readable block
    panels/              match panels for the worst pairs (with --panels)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from driftsense.evaluate import (  # noqa: E402
    DEFAULT_TOLERANCES,
    evaluate_dataset,
    format_report,
    write_evaluation,
)
from driftsense.failures import classify_dataset, format_taxonomy_report  # noqa: E402
from driftsense.locate import LocalisationConfig, load_greyscale, locate  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_dataset.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dataset", type=Path, help="dataset directory containing ground_truth.csv")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: <dataset>/evaluation)",
    )
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N pairs")
    parser.add_argument(
        "--zoom-ratio", type=float, default=10.0, help="search-to-reference pixel size ratio"
    )
    parser.add_argument(
        "--tie-tolerance",
        type=float,
        default=0.01,
        help="NCC score window within which the centre tiebreak applies",
    )
    parser.add_argument(
        "--max-candidates", type=int, default=16, help="size of the candidate shortlist"
    )
    parser.add_argument(
        "--no-subpixel", action="store_true", help="disable parabolic peak refinement"
    )
    parser.add_argument(
        "--tolerances",
        type=float,
        nargs="+",
        default=list(DEFAULT_TOLERANCES),
        help="error thresholds in pixels at which to report accuracy",
    )
    parser.add_argument(
        "--panels",
        type=int,
        default=0,
        help="write match panels for the N worst pairs (0 to disable)",
    )
    parser.add_argument("--no-plot", action="store_true", help="skip the accuracy curve PNG")
    parser.add_argument(
        "--no-taxonomy",
        action="store_true",
        help="skip the failure-mode classification",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not (args.dataset / "ground_truth.csv").exists():
        print(f"error: no ground_truth.csv in {args.dataset}", file=sys.stderr)
        return 1

    config = LocalisationConfig(
        zoom_ratio=args.zoom_ratio,
        tie_tolerance=args.tie_tolerance,
        max_candidates=args.max_candidates,
        subpixel=not args.no_subpixel,
    )

    def progress(done: int, total: int) -> None:
        if not args.quiet:
            print(f"\r  evaluated {done}/{total} pairs", end="", flush=True)

    results, summary = evaluate_dataset(
        args.dataset,
        config=config,
        tolerances=tuple(args.tolerances),
        limit=args.limit,
        progress=progress,
    )
    if not args.quiet:
        print()

    output_dir = args.out or (args.dataset / "evaluation")
    written = write_evaluation(
        output_dir,
        results,
        summary,
        plot=not args.no_plot,
        dataset_dir=None if args.no_taxonomy else args.dataset,
    )

    if args.panels > 0:
        _write_panels(args.dataset, results, output_dir / "panels", args.panels, config)

    print(format_report(summary))
    if not args.no_taxonomy:
        print(format_taxonomy_report(classify_dataset(args.dataset, results)))
    if not args.quiet:
        for label, path in written.items():
            print(f"  {label:<8} {path}")
        print()
    return 0


def _write_panels(dataset_dir: Path, results, panels_dir: Path, count: int, config) -> None:
    """Write match panels for the worst-performing pairs."""
    import csv

    from driftsense.visualise import save_match_panel

    with (dataset_dir / "ground_truth.csv").open(encoding="utf-8") as handle:
        rows = {int(row["pair_id"]): row for row in csv.DictReader(handle)}

    worst = sorted(results, key=lambda r: r.error_px, reverse=True)[:count]
    panels_dir.mkdir(parents=True, exist_ok=True)
    for result in worst:
        row = rows[result.pair_id]
        reference = load_greyscale(dataset_dir / row["reference_path"])
        search = load_greyscale(dataset_dir / row["search_path"])
        outcome = locate(reference, search, config)
        save_match_panel(
            reference,
            search,
            outcome,
            panels_dir / f"pair_{result.pair_id:04d}_match.png",
            ground_truth=(result.gt_x, result.gt_y),
            zoom_ratio=config.zoom_ratio,
        )


if __name__ == "__main__":
    raise SystemExit(main())

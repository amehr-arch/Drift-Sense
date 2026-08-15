#!/usr/bin/env python3
"""Batch localisation over a folder of image pairs, with no ground truth needed.

WHEN TO USE THIS RATHER THAN THE OTHER TWO SCRIPTS
--------------------------------------------------
    locate_pattern.py    one pair in, one coordinate out. The reference entry
                         point, and the one to quote in a submission.
    predict_dataset.py   many pairs in, a CSV of coordinates out. Use this on a
                         test set you have been *given*, where you have images
                         but no answers.
    evaluate_dataset.py  many pairs in, an accuracy report out. Needs a
                         ground_truth.csv with the true coordinates.

Usage
-----
Pairs are matched by filename. By default the script looks for files ending
``_reference.png`` and pairs each with the matching ``_search.png``::

    python predict_dataset.py test_data/
    python predict_dataset.py test_data/ --out my_predictions.csv

If the naming differs, say so::

    python predict_dataset.py test_data/ --reference-suffix _hi.tif --search-suffix _lo.tif
    python predict_dataset.py test_data/ --pattern "**/*_ref.png" --search-suffix _srch.png

Or drive it from a manifest listing the pairs explicitly, which is the safest
option when the naming is irregular::

    python predict_dataset.py --manifest pairs.csv

where ``pairs.csv`` has columns ``reference_path`` and ``search_path``, relative
to the manifest's own directory, and optionally ``pair_id``.

Output
------
A CSV with one row per pair::

    pair_id,reference_path,search_path,x,y,score,confidence_psr,runner_up_margin,preprocessed,elapsed_s

``x`` and ``y`` are the answer, in search-image pixels, under the convention
fixed in ``driftsense.geometry``: pixel (row=i, col=j) occupies
[j, j+1) x [i, i+1).

READ THE CONFIDENCE COLUMNS
---------------------------
``runner_up_margin`` is the gap between the best candidate's score and the next
distinct one. It is the single most useful number here after the coordinates
themselves: on the measured datasets every failure carried a small margin. But
its *precision* is poor on unseen data -- a threshold catching every failure also
discards a majority of correct answers -- so treat a low margin as "worth a look"
rather than as "wrong". The README section "Can a failure be spotted without
ground truth?" has the numbers.

Exit codes
----------
    0  every pair processed
    1  bad input (no pairs found, unreadable image, mismatched dimensions)
    2  finished, but one or more pairs failed; failures are listed on stderr and
       carry an empty x/y in the CSV
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Allow running directly from a clone without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from driftsense.locate import LocalisationConfig, load_greyscale, locate  # noqa: E402

COLUMNS = [
    "pair_id",
    "reference_path",
    "search_path",
    "x",
    "y",
    "score",
    "confidence_psr",
    "runner_up_margin",
    "preprocessed",
    "elapsed_s",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict_dataset.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=None,
        help="directory containing the image pairs",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="CSV listing reference_path and search_path instead of matching by name",
    )
    parser.add_argument(
        "--pattern",
        default="**/*_reference.png",
        help="glob for reference images, relative to the dataset directory",
    )
    parser.add_argument(
        "--reference-suffix",
        default="_reference.png",
        help="filename suffix identifying a reference image",
    )
    parser.add_argument(
        "--search-suffix",
        default="_search.png",
        help="filename suffix its matching search image carries instead",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output CSV (default: <dataset>/predictions.csv)",
    )
    parser.add_argument(
        "--zoom-ratio",
        type=float,
        default=10.0,
        help=(
            "search pixel size divided by reference pixel size. The problem "
            "statement fixes this at 10; change it only if your data differs"
        ),
    )
    parser.add_argument(
        "--no-arbitration",
        action="store_true",
        help="single pass instead of running band-passed and plain and comparing",
    )
    parser.add_argument("--limit", type=int, default=None, help="process only the first N pairs")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    return parser


def _pairs_from_manifest(manifest: Path) -> List[Tuple[str, Path, Path]]:
    root = manifest.parent
    with manifest.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{manifest} contains no rows")
    missing = {"reference_path", "search_path"} - set(rows[0])
    if missing:
        raise ValueError(
            f"{manifest} is missing required column(s): {', '.join(sorted(missing))}"
        )
    pairs = []
    for index, row in enumerate(rows):
        pair_id = str(row.get("pair_id") or index)
        pairs.append((pair_id, root / row["reference_path"], root / row["search_path"]))
    return pairs


def _pairs_from_directory(
    dataset: Path, pattern: str, reference_suffix: str, search_suffix: str
) -> List[Tuple[str, Path, Path]]:
    references = sorted(dataset.glob(pattern))
    if not references:
        raise ValueError(
            f"no files matching {pattern!r} under {dataset}. Use --pattern to point "
            f"at your reference images, or --manifest to list the pairs explicitly."
        )
    pairs = []
    unmatched = []
    for reference in references:
        name = reference.name
        if not name.endswith(reference_suffix):
            unmatched.append(reference)
            continue
        search = reference.with_name(name[: -len(reference_suffix)] + search_suffix)
        if not search.exists():
            unmatched.append(reference)
            continue
        pairs.append((name[: -len(reference_suffix)], reference, search))
    if unmatched:
        print(
            f"warning: {len(unmatched)} reference image(s) had no matching "
            f"{search_suffix!r} file and were skipped, e.g. {unmatched[0].name}",
            file=sys.stderr,
        )
    if not pairs:
        raise ValueError(
            f"found {len(references)} reference image(s) but none had a matching "
            f"{search_suffix!r} file. Check --search-suffix."
        )
    return pairs


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.manifest is None and args.dataset is None:
        print("error: give a dataset directory or --manifest", file=sys.stderr)
        return 1

    try:
        if args.manifest is not None:
            pairs = _pairs_from_manifest(args.manifest)
            default_out = args.manifest.parent / "predictions.csv"
        else:
            pairs = _pairs_from_directory(
                args.dataset, args.pattern, args.reference_suffix, args.search_suffix
            )
            default_out = args.dataset / "predictions.csv"
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.limit is not None:
        pairs = pairs[: args.limit]

    config = LocalisationConfig(
        zoom_ratio=args.zoom_ratio,
        arbitrate_preprocessing=not args.no_arbitration,
    )

    rows: List[dict] = []
    failures: List[str] = []
    started = time.perf_counter()

    for index, (pair_id, reference_path, search_path) in enumerate(pairs, start=1):
        row = {
            "pair_id": pair_id,
            "reference_path": str(reference_path),
            "search_path": str(search_path),
            "x": "",
            "y": "",
            "score": "",
            "confidence_psr": "",
            "runner_up_margin": "",
            "preprocessed": "",
            "elapsed_s": "",
        }
        try:
            reference = load_greyscale(reference_path)
            search = load_greyscale(search_path)
            result = locate(reference, search, config)
        except Exception as error:  # noqa: BLE001 - one bad pair must not stop the run
            failures.append(f"{pair_id}: {type(error).__name__}: {error}")
            rows.append(row)
            continue

        row.update(
            {
                "x": round(float(result.x), 4),
                "y": round(float(result.y), 4),
                "score": round(float(result.score), 6),
                "confidence_psr": round(float(result.confidence), 4),
                "runner_up_margin": (
                    "" if result.runner_up_margin is None
                    else round(float(result.runner_up_margin), 6)
                ),
                "preprocessed": int(bool(result.preprocessed)),
                "elapsed_s": round(float(result.elapsed_s), 6),
            }
        )
        rows.append(row)

        if not args.quiet:
            print(f"\r  {index}/{len(pairs)} pairs", end="", flush=True)

    if not args.quiet:
        print()

    out_path = args.out or default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.perf_counter() - started
    if not args.quiet:
        succeeded = len(rows) - len(failures)
        print(f"  wrote {out_path}")
        print(f"  {succeeded}/{len(rows)} pairs located in {elapsed:.1f} s")
        if succeeded:
            times = [float(r["elapsed_s"]) for r in rows if r["elapsed_s"] != ""]
            print(f"  {sum(times) / len(times) * 1000:.0f} ms per pair, algorithm only")

    if failures:
        print(f"\n{len(failures)} pair(s) failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Standalone localisation inference script.

THIS IS THE SCRIPT THE EVALUATORS RUN. It takes a reference image path and a
search image path, and prints the predicted centre (x, y) of the reference
pattern within the search image. It requires no editing, no configuration file
and no trained weights, and it depends only on NumPy and Pillow.

Usage
-----
    python locate_pattern.py <reference.png> <search.png>

Output (default) is one line, two space-separated floats -- x then y, in
search-image pixels::

    299.100 618.500

Machine-readable and diagnostic forms::

    python locate_pattern.py ref.png search.png --format json
    python locate_pattern.py ref.png search.png --format csv
    python locate_pattern.py ref.png search.png --format json --verbose

Coordinate convention
---------------------
Pixel (row=i, col=j) occupies [j, j+1) x [i, i+1), so an image of size N spans
[0, N). Coordinates are reported as (x, y) with x the column axis. This matches
the convention used by the ground-truth metadata released with the problem
statement.

Exit codes
----------
    0  success
    1  bad input (unreadable image, wrong dimensions, no contrast)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running directly from a clone without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from driftsense.locate import LocalisationConfig, locate_files  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="locate_pattern.py",
        description=(
            "Locate a high-magnification reference pattern inside a wide-field "
            "search image and report its centre in search-image pixels."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Output: 'x y' on one line, in search-image pixels.",
    )
    parser.add_argument("reference", type=Path, help="path to the reference (high-mag) image")
    parser.add_argument("search", type=Path, help="path to the wide-field search image")
    parser.add_argument(
        "--format",
        choices=("plain", "json", "csv"),
        default="plain",
        help="output format (default: plain, 'x y')",
    )
    parser.add_argument(
        "--zoom-ratio",
        type=float,
        default=10.0,
        help="search-to-reference pixel size ratio (default: 10.0)",
    )
    parser.add_argument(
        "--tie-tolerance",
        type=float,
        default=0.01,
        help=(
            "candidates within this NCC score of the best are treated as equally "
            "good, and the one closest to the image centre is returned "
            "(default: 0.01)"
        ),
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=16,
        help="size of the candidate shortlist considered for the tiebreak (default: 16)",
    )
    parser.add_argument(
        "--no-subpixel",
        action="store_true",
        help="report the integer correlation peak without parabolic refinement",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="include the candidate shortlist in JSON output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    for label, path in (("reference", args.reference), ("search", args.search)):
        if not path.exists():
            print(f"error: {label} image not found: {path}", file=sys.stderr)
            return 1

    config = LocalisationConfig(
        zoom_ratio=args.zoom_ratio,
        tie_tolerance=args.tie_tolerance,
        max_candidates=args.max_candidates,
        subpixel=not args.no_subpixel,
    )

    try:
        result = locate_files(args.reference, args.search, config)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.format == "plain":
        print(f"{result.x:.3f} {result.y:.3f}")
    elif args.format == "csv":
        print("x,y")
        print(f"{result.x:.4f},{result.y:.4f}")
    else:
        payload = result.as_dict()
        if not args.verbose:
            payload.pop("candidates", None)
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

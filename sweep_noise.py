#!/usr/bin/env python3
"""Measure localisation accuracy against one imaging parameter at a time.

Examples
--------
    python sweep_noise.py --parameter dose --values 1600 400 100 25
    python sweep_noise.py --parameter spot_size_nm --values 4 14 26 40
    python sweep_noise.py --parameter rotation_deg --values 0 0.5 1.5 3 --apply-to reference

Accuracy is reported on the solvable subset only -- pairs whose reference window
carries a structural anchor on both axes. Pairs without one are ambiguous
regardless of noise, and including them would add a constant offset that hides the
effect being measured.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from driftsense.sweep import format_sweep, sweep_parameter  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sweep_noise.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--parameter", required=True, help="CaptureParams field to sweep")
    parser.add_argument("--values", type=float, nargs="+", required=True, help="levels to sweep")
    parser.add_argument(
        "--apply-to",
        choices=("search", "reference", "both"),
        default="search",
        help="which capture the parameter is varied on (default: search)",
    )
    parser.add_argument("--pairs", type=int, default=10, help="scenes per level (default: 10)")
    parser.add_argument("--seed", type=int, default=4242, help="scene seed (default: 4242)")
    parser.add_argument("--out", type=Path, default=None, help="write results as JSON")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    def progress(done: int, total: int) -> None:
        if not args.quiet:
            print(f"\r  {done}/{total} pairs", end="", flush=True)

    try:
        points = sweep_parameter(
            args.parameter,
            args.values,
            n_pairs=args.pairs,
            seed=args.seed,
            apply_to=args.apply_to,
            progress=progress,
        )
    except (AttributeError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if not args.quiet:
        print()
    print(format_sweep(f"{args.parameter}  (applied to {args.apply_to} capture)", points))
    print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps([p.as_dict() for p in points], indent=2), encoding="utf-8"
        )
        if not args.quiet:
            print(f"  wrote {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

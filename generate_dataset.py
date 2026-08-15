#!/usr/bin/env python3
"""Standalone dataset generator for the Drift-Sense navigation-error problem.

Generates reference/search image pairs with exact ground-truth coordinates.

Examples
--------
Generate the default 30-pair self-evaluation set::

    python generate_dataset.py --pairs 30 --out data/dram_v1

Reproduce one specific pair for debugging, with a verification panel::

    python generate_dataset.py --pairs 1 --seed 1234 --out /tmp/one --overlays

Harder sub-pixel regime (crop origins are not snapped to whole nanometres)::

    python generate_dataset.py --pairs 30 --out data/subpixel --subpixel

Outputs
-------
``<out>/pairs/pair_XXXX_reference.png``   1000x1000 8-bit greyscale
``<out>/pairs/pair_XXXX_search.png``      1000x1000 8-bit greyscale
``<out>/pairs/pair_XXXX_meta.json``       per-pair parameters and ground truth
``<out>/overlays/pair_XXXX_overlay.png``  verification panel (with --overlays)
``<out>/ground_truth.csv``                one row per pair
``<out>/dataset_manifest.json``           dataset-level configuration record
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running directly from a clone without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from driftsense import (  # noqa: E402
    GenerationConfig,
    ImagingGeometry,
    available_architectures,
    generate_dataset,
)
from driftsense.optical import OpticalParams  # noqa: E402
from driftsense.roughness import RoughnessParams  # noqa: E402
from driftsense.generate import default_validator  # noqa: E402
from driftsense.validate import ValidationThresholds, make_validator  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_dataset.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--architecture",
        default="dram",
        choices=available_architectures(),
        help="die architecture style to generate (default: dram)",
    )
    parser.add_argument(
        "--pairs", type=int, default=30, help="number of image pairs (default: 30)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/generated"),
        help="output directory (default: data/generated)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="master random seed (default: 42)"
    )
    parser.add_argument(
        "--image-size", type=int, default=1000, help="image side in pixels (default: 1000)"
    )
    parser.add_argument(
        "--reference-pixel-nm",
        type=float,
        default=1.0,
        help="reference pixel size in nm (default: 1.0)",
    )
    parser.add_argument(
        "--zoom-ratio",
        type=float,
        default=10.0,
        help="search-to-reference pixel size ratio (default: 10.0)",
    )
    parser.add_argument(
        "--boundary-bias",
        type=float,
        default=0.35,
        help=(
            "per-axis probability of steering the crop onto a structural boundary; "
            "lower values produce harder, more ambiguous pairs (default: 0.35)"
        ),
    )
    parser.add_argument(
        "--subpixel",
        action="store_true",
        help="allow crop origins at fractional nanometres (harder sub-pixel regime)",
    )
    parser.add_argument(
        "--overlays",
        action="store_true",
        help="write a verification panel per pair",
    )
    parser.add_argument(
        "--imaging",
        action="store_true",
        help="apply the SEM imaging model (noise, beam blur, stage error, charging)",
    )
    parser.add_argument(
        "--min-zncc",
        type=float,
        default=None,
        help=(
            "minimum ground-truth correlation accepted by validation "
            "(default: 0.95 for layout renders, 0.55 with --imaging)"
        ),
    )
    parser.add_argument(
        "--optical",
        action="store_true",
        help=(
            "generate 3-channel optical-microscope pairs instead of SEM greyscale "
            "(bonus scope; see driftsense/optical.py -- characterised, not solved)"
        ),
    )
    parser.add_argument(
        "--ler-nm",
        type=float,
        default=None,
        help=(
            "line-edge roughness sigma in nm (default: 1.2 with --imaging, none "
            "without). 0 disables it"
        ),
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="skip per-pair validation (not recommended)",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    geometry = ImagingGeometry(
        image_size_px=args.image_size,
        reference_pixel_size_nm=args.reference_pixel_nm,
        zoom_ratio=args.zoom_ratio,
    )
    config = GenerationConfig(
        output_dir=args.out,
        architecture=args.architecture,
        roughness=(
            None if args.ler_nm is None else RoughnessParams(amplitude_nm=args.ler_nm)
        ),
        apply_roughness=(args.ler_nm is None or args.ler_nm > 0.0),
        optical=OpticalParams() if args.optical else None,
        n_pairs=args.pairs,
        seed=args.seed,
        geometry=geometry,
        boundary_bias=args.boundary_bias,
        subpixel_placement=args.subpixel,
        save_overlays=args.overlays,
        imaging=args.imaging,
    )

    if args.no_validate:
        validator = None
    elif args.min_zncc is not None:
        validator = make_validator(ValidationThresholds(min_zncc=args.min_zncc))
    else:
        validator = default_validator(config)

    def progress(done: int, total: int) -> None:
        if not args.quiet:
            print(f"\r  generated {done}/{total} pairs", end="", flush=True)

    started = time.perf_counter()
    manifest = generate_dataset(config, validator=validator, progress=progress)
    elapsed = time.perf_counter() - started

    if not args.quiet:
        print()
        print(f"  wrote {manifest['n_pairs']} pairs to {config.output_dir}")
        print(f"  ground truth: {config.output_dir / manifest['ground_truth_csv']}")
        print(f"  elapsed: {elapsed:.2f} s ({elapsed / max(1, config.n_pairs):.2f} s/pair)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

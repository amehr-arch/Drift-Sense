"""Dataset generation: build reference/search pairs with exact ground truth.

THE GENERATION CONTRACT
-----------------------
One layout is instantiated per pair. Both images are then rendered from *that same
layout*:

    reference : window at (origin_x, origin_y), 1000 px at 1 nm/px
    search    : window at (0, 0),               1000 px at 10 nm/px

Because both are drawn from one procedural model, the reference pattern is
genuinely present inside the search image, and the ground truth follows
arithmetically from the crop origin rather than being estimated by any matching
procedure. That is the property the whole project rests on, and it is verified
independently in :mod:`driftsense.validate`.

IMAGE FORMATION
---------------
With ``imaging=True`` each capture is passed through :mod:`driftsense.imaging`,
which converts material coverage into a simulated micrograph. The two captures are
given *independent* random generators, so their noise is uncorrelated as two
separate physical acquisitions require, and different dose -- the wide-field image
collects about a tenth of the electrons per pixel, which is why it is noisier.

The default is ``imaging=False``, producing noise-free layout renders. That is
deliberate: introducing the imaging model must not change what earlier
stages generate, and the noise-free renders remain the right input for geometry
regression tests, where a coordinate error has nowhere to hide.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .geometry import GroundTruth, ImagingGeometry, Placement, ground_truth_for
from .imaging import (
    REFERENCE_CAPTURE,
    SEARCH_CAPTURE,
    CaptureParams,
    MaterialYields,
    form_image,
)
from .optical import OpticalParams, form_rgb_image
from .roughness import RoughnessParams, apply_roughness
from .layouts import LayoutModel, RenderWindow
from .sampling import (
    FinfetParamRanges,
    CaptureRanges,
    DramParamRanges,
    PlacementSampler,
    sample_captures,
    sample_dram_layout,
    sample_finfet_layout,
)

__all__ = [
    "GeneratedPair",
    "GenerationConfig",
    "generate_pair",
    "generate_dataset",
    "default_validator",
]

#: Columns of ``ground_truth.csv``. The first four are what a scoring utility
#: needs; the rest exist so any failing pair can be reproduced and diagnosed.
CSV_COLUMNS = [
    "pair_id",
    "reference_path",
    "search_path",
    "gt_x",
    "gt_y",
    "box_x",
    "box_y",
    "box_w",
    "box_h",
    "architecture",
    "anchor",
    "anchor_x",
    "anchor_y",
    "template_size_px",
    "origin_x_nm",
    "origin_y_nm",
    "feature_size_nm",
    "cell_architecture",
    "pitch_x_nm",
    "pitch_y_nm",
    "bitline_pitch_nm",
    "wordline_pitch_nm",
    "mat_size_nm",
    "strip_width_nm",
    "mat_pitch_variation",
    "seed",
]


@dataclass
class GeneratedPair:
    """One reference/search pair together with its exact answer.

    ``anchor_x`` and ``anchor_y`` record whether the reference window contains a
    structural boundary on each axis. They are the difficulty label: an axis with
    no anchor is periodic within the window, so its position is recoverable only
    up to one cell pitch and the pair is genuinely ambiguous on that axis. Making
    this an explicit property of the data -- rather than an unexplained ceiling
    discovered later in the accuracy figures -- is what lets evaluation report
    solvable and ambiguous pairs separately.
    """

    pair_id: int
    reference: np.ndarray  # uint8, (N, N)
    search: np.ndarray  # uint8, (N, N)
    ground_truth: GroundTruth
    placement: Placement
    geometry: ImagingGeometry
    layout_description: Dict[str, object]
    seed: int
    anchor_x: bool = False
    anchor_y: bool = False
    reference_capture: Optional[CaptureParams] = None
    search_capture: Optional[CaptureParams] = None

    @property
    def name(self) -> str:
        return f"pair_{self.pair_id:04d}"

    @property
    def anchor(self) -> str:
        """Difficulty class: ``both``, ``x``, ``y`` or ``none``."""
        if self.anchor_x and self.anchor_y:
            return "both"
        if self.anchor_x:
            return "x"
        if self.anchor_y:
            return "y"
        return "none"

    def as_metadata(self) -> Dict[str, object]:
        """Per-pair metadata record.

        Keys ``architecture``, ``gt_x``, ``gt_y``, ``gt_box`` and ``seed`` are
        named to match the schema of the metadata block released with the
        organisers' sample pair, so downstream tooling written against theirs can
        read ours unchanged. Everything else is namespaced under ``driftsense``.
        """
        return {
            "architecture": self.layout_description.get("architecture"),
            **self.ground_truth.as_dict(),
            "seed": self.seed,
            "driftsense": {
                "pair_id": self.pair_id,
                "stage": 3,
                "imaging_model": (
                    "none (layout render only)"
                    if self.reference_capture is None and self.search_capture is None
                    else "sem"
                ),
                "capture": {
                    "reference": (
                        None if self.reference_capture is None
                        else self.reference_capture.as_dict()
                    ),
                    "search": (
                        None if self.search_capture is None
                        else self.search_capture.as_dict()
                    ),
                },
                "geometry": self.geometry.as_dict(),
                "placement": self.placement.as_dict(),
                "anchor": {"x": self.anchor_x, "y": self.anchor_y, "class": self.anchor},
                "layout": self.layout_description,
            },
        }


@dataclass
class GenerationConfig:
    """Everything needed to reproduce a dataset from a single integer seed."""

    output_dir: Path
    architecture: str = "dram"
    n_pairs: int = 30
    seed: int = 42
    geometry: ImagingGeometry = field(default_factory=ImagingGeometry)
    boundary_bias: float = 0.35
    subpixel_placement: bool = False
    save_overlays: bool = True
    dram_ranges: DramParamRanges = field(default_factory=DramParamRanges)
    finfet_ranges: FinfetParamRanges = field(default_factory=FinfetParamRanges)
    #: Apply the SEM imaging model. Defaults to off so that adding an earlier revision does
    #: not change what earlier configurations generate; the imaging model
    #: is opted into explicitly.
    imaging: bool = False
    #: Per-capture acquisition settings. ``None`` means sample them per pair.
    reference_capture: Optional[CaptureParams] = None
    search_capture: Optional[CaptureParams] = None
    yields: Optional[MaterialYields] = None
    capture_ranges: CaptureRanges = field(default_factory=CaptureRanges)
    #: Line-edge roughness. ``None`` means "use the default when the imaging model
    #: is on, and none when it is off" -- roughness is structural realism that only
    #: makes sense alongside the rest of the physical model, and switching it on
    #: for the noise-free layout renders would break the exactness those are for.
    roughness: Optional[RoughnessParams] = None
    #: Set false to generate perfectly straight edges even with imaging enabled.
    apply_roughness: bool = True
    #: Generate three-channel optical-microscope images instead of SEM
    #: greyscale. Bonus scope, and characterised rather than solved.
    optical: Optional[OpticalParams] = None

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.n_pairs <= 0:
            raise ValueError("n_pairs must be positive")

    def resolved_roughness(self) -> Optional[RoughnessParams]:
        """The roughness actually in force, after the imaging/flag interaction."""
        if not self.apply_roughness or not self.imaging:
            return None
        return self.roughness or RoughnessParams()

    def as_dict(self) -> Dict[str, object]:
        return {
            "architecture": self.architecture,
            "n_pairs": self.n_pairs,
            "seed": self.seed,
            "boundary_bias": self.boundary_bias,
            "subpixel_placement": self.subpixel_placement,
            "imaging": self.imaging,
            "roughness": (
                self.resolved_roughness().as_dict() if self.resolved_roughness() else None
            ),
            "geometry": self.geometry.as_dict(),
            "output_dir": str(self.output_dir),
        }


# ---------------------------------------------------------------------------
# Single pair
# ---------------------------------------------------------------------------


def generate_pair(
    layout: LayoutModel,
    placement: Placement,
    geometry: ImagingGeometry,
    pair_id: int = 0,
    seed: int = 0,
    reference_capture: Optional[CaptureParams] = None,
    search_capture: Optional[CaptureParams] = None,
    yields: Optional[MaterialYields] = None,
    rng: Optional[np.random.Generator] = None,
    roughness: Optional[RoughnessParams] = None,
    optical: Optional[OpticalParams] = None,
) -> GeneratedPair:
    """Render one reference/search pair from a layout and a crop placement.

    When no capture parameters are supplied the images are noise-free layout
    renders. Supplying either one switches on the imaging model for that capture;
    each is then given its own generator so the two noise fields are independent.
    """
    reference_window = RenderWindow(
        origin_x_nm=placement.origin_x_nm,
        origin_y_nm=placement.origin_y_nm,
        size_px=geometry.image_size_px,
        pixel_size_nm=geometry.reference_pixel_size_nm,
    )
    search_window = RenderWindow(
        origin_x_nm=0.0,
        origin_y_nm=0.0,
        size_px=geometry.image_size_px,
        pixel_size_nm=geometry.search_pixel_size_nm,
    )

    reference_render = layout.render(reference_window)
    search_render = layout.render(search_window)

    # Line-edge roughness is a property of the specimen, not of the capture, so
    # both windows are displaced by the *same* world-coordinate field. Applying
    # it per capture would have made it independent noise and both physically
    # wrong and easier than reality.
    if roughness is not None and roughness.enabled:
        reference_render = apply_roughness(reference_render, reference_window, roughness)
        search_render = apply_roughness(search_render, search_window, roughness)

    if optical is not None:
        # Optical path: three channels, diffraction-limited. Bonus scope; see
        # driftsense.optical for what it measured and why it is not solved.
        reference = form_rgb_image(
            reference_render, optical, geometry.reference_pixel_size_nm,
            np.random.default_rng((seed, 0xC0)),
        )
        search = form_rgb_image(
            search_render, optical, geometry.search_pixel_size_nm,
            np.random.default_rng((seed, 0xC1)),
        )
    elif reference_capture is None and search_capture is None:
        reference = reference_render.to_uint8()
        search = search_render.to_uint8()
    else:
        # Independent streams: two separate physical acquisitions must not share
        # a noise realisation. Spawning from one parent keeps the pair as a whole
        # reproducible from its seed.
        parent = rng if rng is not None else np.random.default_rng(seed)
        ref_rng, search_rng = (
            np.random.default_rng(child) for child in parent.bit_generator.seed_seq.spawn(2)
        )
        reference = form_image(
            reference_render,
            reference_capture or REFERENCE_CAPTURE,
            geometry.reference_pixel_size_nm,
            ref_rng,
            yields,
            apply_geometry=True,
        )
        search = form_image(
            search_render,
            search_capture or SEARCH_CAPTURE,
            geometry.search_pixel_size_nm,
            search_rng,
            yields,
            apply_geometry=False,
        )

    # Difficulty label: does the reference window contain a structural anchor?
    span = geometry.reference_fov_nm
    anchor_x = bool(
        layout.boundary_coordinates_nm(
            "x", placement.origin_x_nm, placement.origin_x_nm + span
        )
    )
    anchor_y = bool(
        layout.boundary_coordinates_nm(
            "y", placement.origin_y_nm, placement.origin_y_nm + span
        )
    )

    return GeneratedPair(
        pair_id=pair_id,
        reference=reference,
        search=search,
        ground_truth=ground_truth_for(placement, geometry),
        placement=placement,
        geometry=geometry,
        layout_description=layout.describe(),
        seed=seed,
        anchor_x=anchor_x,
        anchor_y=anchor_y,
        reference_capture=reference_capture,
        search_capture=search_capture,
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _pitch_nm(layout_desc: Dict[str, object], axis: str) -> float:
    """Repeat distance of the finest grating along ``axis``, whatever the architecture.

    Both layouts are two crossed 1-D gratings; only the names differ. Recording a
    neutral pitch per axis is what lets the failure taxonomy reason about aliasing
    without knowing which architecture produced the pair.
    """
    keys = {
        "x": ("bitline_pitch_nm", "fin_pitch_nm"),
        "y": ("wordline_pitch_nm", "gate_pitch_nm"),
    }[axis]
    for key in keys:
        value = layout_desc.get(key)
        if value is not None:
            return float(value)
    return float("nan")


def _feature_size_nm(layout_desc: Dict[str, object]) -> float:
    """Smallest printed dimension, used to judge whether a capture resolves it.

    DRAM states its feature size directly. A FinFET layout's finest printed
    feature is the fin itself, so the fin width stands in for it.
    """
    for key in ("feature_size_nm", "fin_width_nm"):
        value = layout_desc.get(key)
        if value is not None:
            return float(value)
    return float("nan")


def _build_layout(architecture: str, rng: np.random.Generator, config: GenerationConfig) -> LayoutModel:
    if architecture == "dram":
        return sample_dram_layout(rng, config.dram_ranges, config.geometry)
    if architecture == "finfet":
        return sample_finfet_layout(rng, config.finfet_ranges, config.geometry)
    raise KeyError(
        f"no sampler registered for architecture {architecture!r}. "
        "Add one alongside sample_dram_layout in driftsense.sampling."
    )


def default_validator(config: "GenerationConfig"):
    """Validation callback matched to whether the imaging model is in use."""
    from .validate import ValidationThresholds, make_validator

    thresholds = (
        ValidationThresholds.for_imaging(config.architecture)
        if config.imaging
        else ValidationThresholds()
    )
    return make_validator(thresholds)


def generate_dataset(
    config: GenerationConfig,
    validator: Optional[callable] = None,
    progress: Optional[callable] = None,
) -> Dict[str, object]:
    """Generate, validate and write a complete dataset.

    Parameters
    ----------
    config:
        Generation settings. The whole dataset is a deterministic function of
        ``config.seed``.
    validator:
        Optional callable ``(GeneratedPair) -> list[str]``. Any issue it reports
        aborts generation, on the principle that a malformed dataset is
        far more expensive than a loud failure.
    progress:
        Optional callable ``(index, total)`` for CLI feedback.

    Returns
    -------
    The manifest dictionary, which is also written to ``dataset_manifest.json``.
    """
    from PIL import Image  # imported lazily so the core stays import-light

    out = Path(config.output_dir)
    pairs_dir = out / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = out / "overlays"
    if config.save_overlays:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    # One seed sequence per pair, spawned from the master seed. Spawning (rather
    # than reusing one stream) means pair k is reproducible in isolation.
    master = np.random.SeedSequence(config.seed)
    children = master.spawn(config.n_pairs)

    rows: List[Dict[str, object]] = []

    # One roughness template for the run; each pair reseeds it from its own pair
    # seed, so a pair reproduces in isolation and no two specimens share edges.
    base_roughness = config.resolved_roughness()

    for index, child in enumerate(children):
        rng = np.random.default_rng(child)
        pair_seed = int(child.generate_state(1, dtype=np.uint32)[0])

        layout = _build_layout(config.architecture, rng, config)
        sampler = PlacementSampler(
            geometry=config.geometry,
            boundary_bias=config.boundary_bias,
            subpixel=config.subpixel_placement,
        )
        placement = sampler.sample(layout, rng)

        if config.imaging:
            sampled_reference, sampled_search = sample_captures(rng, config.capture_ranges)
            reference_capture = config.reference_capture or sampled_reference
            search_capture = config.search_capture or sampled_search
        else:
            reference_capture = search_capture = None

        pair = generate_pair(
            layout,
            placement,
            config.geometry,
            pair_id=index,
            seed=pair_seed,
            reference_capture=reference_capture,
            search_capture=search_capture,
            yields=config.yields,
            rng=rng,
            roughness=(
                None
                if base_roughness is None
                else replace(base_roughness, seed=pair_seed)
            ),
            optical=config.optical,
        )

        if validator is not None:
            issues = validator(pair)
            if issues:
                raise RuntimeError(
                    f"pair {index} failed validation:\n  " + "\n  ".join(issues)
                )

        ref_path = pairs_dir / f"{pair.name}_reference.png"
        search_path = pairs_dir / f"{pair.name}_search.png"
        # Optical pairs are three-channel; SEM pairs are single-channel. Let the
        # array's own shape decide the mode rather than hard-coding "L".
        def _save(array, path):
            mode = "RGB" if array.ndim == 3 else "L"
            Image.fromarray(array, mode=mode).save(path)

        _save(pair.reference, ref_path)
        _save(pair.search, search_path)
        (pairs_dir / f"{pair.name}_meta.json").write_text(
            json.dumps(pair.as_metadata(), indent=2), encoding="utf-8"
        )

        if config.save_overlays:
            from .visualise import save_verification_panel

            save_verification_panel(pair, overlays_dir / f"{pair.name}_overlay.png")

        layout_desc = pair.layout_description
        gt = pair.ground_truth
        rows.append(
            {
                "pair_id": pair.pair_id,
                "reference_path": str(ref_path.relative_to(out).as_posix()),
                "search_path": str(search_path.relative_to(out).as_posix()),
                "gt_x": round(gt.x, 6),
                "gt_y": round(gt.y, 6),
                "box_x": round(gt.box_x, 6),
                "box_y": round(gt.box_y, 6),
                "box_w": round(gt.box_w, 6),
                "box_h": round(gt.box_h, 6),
                "architecture": layout_desc.get("architecture"),
                "anchor": pair.anchor,
                "anchor_x": int(pair.anchor_x),
                "anchor_y": int(pair.anchor_y),
                "template_size_px": config.geometry.template_size_px,
                "origin_x_nm": round(placement.origin_x_nm, 6),
                "origin_y_nm": round(placement.origin_y_nm, 6),
                "feature_size_nm": round(_feature_size_nm(layout_desc), 4),
                "cell_architecture": layout_desc.get("cell_architecture"),
                "pitch_x_nm": round(_pitch_nm(layout_desc, "x"), 4),
                "pitch_y_nm": round(_pitch_nm(layout_desc, "y"), 4),
                "bitline_pitch_nm": round(float(layout_desc.get("bitline_pitch_nm", float("nan"))), 4),
                "wordline_pitch_nm": round(float(layout_desc.get("wordline_pitch_nm", float("nan"))), 4),
                "mat_size_nm": round(float(layout_desc.get("mat_size_nm", float("nan"))), 4),
                "strip_width_nm": round(float(layout_desc.get("strip_width_nm", float("nan"))), 4),
                "mat_pitch_variation": round(
                    float(layout_desc.get("mat_pitch_variation", float("nan"))), 4
                ),
                "seed": pair.seed,
            }
        )

        if progress is not None:
            progress(index + 1, config.n_pairs)

    csv_path = out / "ground_truth.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "stage": 3,
        "imaging_model": "sem" if config.imaging else "none (layout render only)",
        "config": config.as_dict(),
        "n_pairs": len(rows),
        "ground_truth_csv": csv_path.name,
        "coordinate_convention": (
            "pixel (i, j) occupies [j, j+1) x [i, i+1); an image of size N spans "
            "[0, N); reported coordinates are (x=col, y=row)"
        ),
    }
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

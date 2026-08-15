"""Failure taxonomy: name the root cause of every wrong answer.

PURPOSE
-------
An accuracy number says how often the locator is wrong. It says nothing about
*why*, and "why" is the question a reviewer actually asks. A method that fails
on genuinely unsolvable inputs is in a completely different position from one
that fails on solvable ones. Before this module the repository had a single
worked failure case, discussed in prose. One anecdote is not a failure analysis.

This classifies every pair into exactly one named mode, using only evidence that
is available after evaluation: the ground-truth row, the pair metadata written by
the generator, and the locator's own reported confidence. Each mode carries the
evidence that selected it, so a classification can be argued with rather than
taken on trust.

SCOPE
-----
Not part of inference. Nothing here is importable from ``driftsense.locate``,
and nothing here runs when a user calls ``locate_pattern.py``. The classifier is
allowed to read the layout parameters that produced a pair -- cell pitches, mat
sizes, capture settings -- precisely because it is an *analysis* tool run after
the fact by someone who has the ground truth. The locator has none of that and
must not.

PRECEDENCE ORDER
----------------
The modes are not disjoint in nature: a pair can be both blur-limited *and*
periodic on one axis. Classification therefore walks the modes in a fixed
precedence order and assigns the first that matches, so the taxonomy partitions
the pairs rather than double-counting them. The order runs from *most
fundamental* to *most contingent*:

    1. correct                 -- not a failure at all
    2. unanchored_axis         -- the axis was never recoverable, by any method
    3. periodic_alias          -- landed on a true repeat of the pattern
    4. blur_limited            -- the search capture cannot resolve the structure
    5. subpixel_drift          -- right cell, imprecise centre
    6. unexplained             -- none of the above

``unanchored_axis`` precedes ``periodic_alias`` deliberately. Both describe a
landing on a repeat, but the first says the repeat was unavoidable given what the
reference window contained, and the second says the window did carry a landmark
and the locator failed to use it. Only the second is a criticism of the method.

``unexplained`` exists and is reported. A taxonomy in which every failure is
explained is usually a taxonomy with a bucket wide enough to swallow anything.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

__all__ = [
    "FailureMode",
    "FailureVerdict",
    "TaxonomyConfig",
    "MODE_DESCRIPTIONS",
    "classify_pair",
    "classify_dataset",
    "summarise_modes",
    "format_taxonomy_report",
]


class FailureMode:
    """The mode names, as constants rather than bare strings."""

    CORRECT = "correct"
    UNANCHORED_AXIS = "unanchored_axis"
    PERIODIC_ALIAS = "periodic_alias"
    BLUR_LIMITED = "blur_limited"
    SUBPIXEL_DRIFT = "subpixel_drift"
    UNEXPLAINED = "unexplained"

    ORDER = (
        CORRECT,
        UNANCHORED_AXIS,
        PERIODIC_ALIAS,
        BLUR_LIMITED,
        SUBPIXEL_DRIFT,
        UNEXPLAINED,
    )


MODE_DESCRIPTIONS: Dict[str, str] = {
    FailureMode.CORRECT: (
        "Within tolerance. Reported so the modes sum to the dataset."
    ),
    FailureMode.UNANCHORED_AXIS: (
        "The reference window is periodic on the failing axis and carries no "
        "structural landmark there. The true position is recoverable only to one "
        "cell pitch, by any method. Not a defect in the locator."
    ),
    FailureMode.PERIODIC_ALIAS: (
        "The window did carry a landmark on the failing axis, but the prediction "
        "landed on an integer multiple of the cell pitch away -- a true repeat of "
        "the pattern. This one is a criticism of the method. Each verdict reports "
        "the chance a random offset would pass the same test, so a weak "
        "attribution on a fine pitch is visible as such."
    ),
    FailureMode.BLUR_LIMITED: (
        "The search capture's beam spot is large enough relative to the feature "
        "size that the discriminating structure is not present in the image. "
        "Correlation cannot recover what the optics removed."
    ),
    FailureMode.SUBPIXEL_DRIFT: (
        "The correct region was found -- the error is well under one cell pitch -- "
        "but the sub-pixel centre is imprecise. A precision limit, not a "
        "mis-identification."
    ),
    FailureMode.UNEXPLAINED: (
        "Wrong, and none of the above accounts for it. Reported rather than "
        "absorbed into a neighbouring mode."
    ),
}


@dataclass(frozen=True)
class TaxonomyConfig:
    """Thresholds separating the modes.

    Every value here is a judgement call, so each is named, defaulted and
    documented rather than buried as a literal in the classifier.
    """

    #: Error at or below this counts as correct. Matches the tolerance the
    #: problem statement asks accuracy to be reported against.
    tolerance_px: float = 1.0
    #: A residual this small, after subtracting the nearest whole number of cell
    #: pitches, means the prediction sits on a genuine repeat rather than near
    #: one by coincidence. Expressed as a fraction of the pitch.
    alias_residual_fraction: float = 0.25
    #: The same test in absolute pixels, and the binding one on a fine pitch.
    #: A fraction alone is nearly vacuous when the pitch is small: at a 4.8 px
    #: pitch, a residual within 0.25 of a multiple happens for half of all random
    #: offsets, so "alias" would be barely better than a coin flip. Requiring the
    #: residual to be within roughly the locator's own precision fixes that.
    alias_residual_px: float = 1.0
    #: An alias must be at least this many whole pitches away, so that a
    #: sub-pitch wobble is not read as landing on the neighbouring cell.
    alias_min_pitch_multiples: float = 0.75
    #: Ratio of search-capture beam spot to feature size above which the
    #: structure is treated as unresolved. The an earlier revision sweep put the collapse
    #: between 16 and 30 nm at a ~35-46 nm feature size, so the crossover sits
    #: near 0.6; 0.7 is the conservative side of it.
    blur_ratio: float = 0.7
    #: Error below this fraction of a cell pitch means the right cell was found.
    subpixel_pitch_fraction: float = 0.5
    #: Runner-up margin at or below this is treated as the locator declaring low
    #: confidence. Confident pairs in the measured sets sit near 0.07-0.14.
    low_margin: float = 0.02
    #: Peak-to-sidelobe ratio at or below this is likewise low confidence.
    low_psr: float = 5.0

    def __post_init__(self) -> None:
        for name in (
            "tolerance_px",
            "alias_residual_fraction",
            "alias_residual_px",
            "alias_min_pitch_multiples",
            "blur_ratio",
            "subpixel_pitch_fraction",
            "low_margin",
            "low_psr",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
        if self.alias_residual_fraction >= 0.5:
            raise ValueError(
                "alias_residual_fraction must be below 0.5, otherwise every offset "
                "is within half a pitch of some multiple and the test is vacuous"
            )


@dataclass
class FailureVerdict:
    """One pair's classification, with the evidence that produced it."""

    pair_id: int
    mode: str
    error_px: float
    anchor: str
    #: Human-readable justification naming the numbers that decided it.
    evidence: str
    #: Whether the locator itself signalled low confidence on this pair.
    flagged_low_confidence: bool
    confidence_psr: float
    runner_up_margin: Optional[float]
    #: Per-axis detail, present when pitch information was available.
    detail: Dict[str, float] = field(default_factory=dict)

    def as_row(self) -> Dict[str, object]:
        row = asdict(self)
        row["error_px"] = round(float(row["error_px"]), 5)
        row["confidence_psr"] = round(float(row["confidence_psr"]), 4)
        row["runner_up_margin"] = (
            "" if row["runner_up_margin"] is None else round(float(row["runner_up_margin"]), 6)
        )
        row["detail"] = json.dumps({k: round(v, 4) for k, v in row["detail"].items()})
        return row


def _pitches(row: Mapping[str, object]) -> Dict[str, Optional[float]]:
    """Cell pitch per axis, in *search-image pixels*.

    The ground-truth CSV records pitches in nanometres, and the search image is
    sampled at ``search_pixel_size_nm``. Bit lines run vertically, so the bit-line
    pitch is the repeat distance along x; word lines run horizontally, so the
    word-line pitch is the repeat along y.
    """

    def get(name: str) -> Optional[float]:
        value = row.get(name)
        if value in (None, ""):
            return None
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return number if number > 0.0 else None

    pixel_nm = get("search_pixel_size_nm") or 10.0
    # ``pitch_x_nm``/``pitch_y_nm`` are architecture-neutral and are what the
    # generator writes now. The DRAM-specific names are still read so that
    # datasets generated before those columns existed still classify.
    along_x = get("pitch_x_nm") or get("bitline_pitch_nm")
    along_y = get("pitch_y_nm") or get("wordline_pitch_nm")
    return {
        "x": None if along_x is None else along_x / pixel_nm,
        "y": None if along_y is None else along_y / pixel_nm,
    }


def _alias_residual(offset: float, pitch: float, residual_px: float) -> Dict[str, float]:
    """How far ``offset`` sits from the nearest whole multiple of ``pitch``.

    ``chance`` is the probability that a *uniformly random* offset would pass the
    absolute residual test at this pitch. It is reported because it is the
    measure of how much the alias verdict is worth: at a coarse pitch a passing
    residual is strong evidence, and at a fine pitch it is nearly none.
    """
    multiples = offset / pitch
    nearest = round(multiples)
    fraction = abs(multiples - nearest)
    return {
        "multiples": abs(multiples),
        "nearest": abs(float(nearest)),
        "residual_fraction": fraction,
        "residual_px": fraction * pitch,
        "chance": min(1.0, 2.0 * residual_px / pitch),
    }


def _spot_ratio(meta: Optional[Mapping[str, object]], row: Mapping[str, object]) -> Optional[float]:
    """Search-capture beam spot as a multiple of the feature size."""
    feature = row.get("feature_size_nm")
    try:
        feature_nm = float(feature)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if feature_nm <= 0.0:
        return None
    if not meta:
        return None
    capture = meta.get("driftsense", {})
    if isinstance(capture, Mapping):
        capture = capture.get("capture", {})
    if not isinstance(capture, Mapping):
        return None
    search = capture.get("search")
    if not isinstance(search, Mapping):
        return None
    spot = search.get("spot_size_nm")
    try:
        spot_nm = float(spot)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return spot_nm / feature_nm if spot_nm > 0.0 else None


def classify_pair(
    row: Mapping[str, object],
    *,
    error_x: float,
    error_y: float,
    error_px: float,
    confidence_psr: float,
    runner_up_margin: Optional[float],
    meta: Optional[Mapping[str, object]] = None,
    config: Optional[TaxonomyConfig] = None,
) -> FailureVerdict:
    """Assign one pair to exactly one mode.

    ``row`` is a ground-truth CSV row; ``meta`` is the pair's ``*_meta.json``
    contents, used only for the capture settings. Both are analysis inputs; see
    the module docstring on why the classifier may read them and the locator may
    not.
    """
    config = config or TaxonomyConfig()
    pair_id = int(row.get("pair_id", -1))
    anchor = str(row.get("anchor", "unknown"))
    anchor_x = str(row.get("anchor_x", "")) == "1"
    anchor_y = str(row.get("anchor_y", "")) == "1"

    margin = runner_up_margin
    flagged = bool(
        (margin is not None and margin <= config.low_margin)
        or confidence_psr <= config.low_psr
    )

    pitches = _pitches(row)
    detail: Dict[str, float] = {}
    for axis, pitch in pitches.items():
        if pitch is not None:
            detail[f"pitch_{axis}_px"] = pitch
    detail["error_x_px"] = float(error_x)
    detail["error_y_px"] = float(error_y)

    def verdict(mode: str, evidence: str) -> FailureVerdict:
        return FailureVerdict(
            pair_id=pair_id,
            mode=mode,
            error_px=float(error_px),
            anchor=anchor,
            evidence=evidence,
            flagged_low_confidence=flagged,
            confidence_psr=float(confidence_psr),
            runner_up_margin=margin,
            detail=detail,
        )

    # 1 -- correct
    if error_px <= config.tolerance_px:
        return verdict(
            FailureMode.CORRECT,
            f"error {error_px:.3f} px is within the {config.tolerance_px:g} px tolerance",
        )

    # Which axis carries the error, and is it periodic there?
    axes = (("x", error_x, anchor_x), ("y", error_y, anchor_y))
    dominant_axis, dominant_error, dominant_anchored = max(
        axes, key=lambda item: abs(item[1])
    )
    pitch = pitches.get(dominant_axis)

    if pitch is not None:
        alias = _alias_residual(abs(dominant_error), pitch, config.alias_residual_px)
        detail[f"alias_multiples_{dominant_axis}"] = alias["multiples"]
        detail[f"alias_residual_{dominant_axis}"] = alias["residual_fraction"]
        detail[f"alias_residual_px_{dominant_axis}"] = alias["residual_px"]
        detail[f"alias_chance_{dominant_axis}"] = alias["chance"]
        on_a_repeat = (
            alias["nearest"] >= config.alias_min_pitch_multiples
            and alias["residual_fraction"] <= config.alias_residual_fraction
            and alias["residual_px"] <= config.alias_residual_px
        )
    else:
        alias = {
            "multiples": float("nan"),
            "residual_fraction": float("nan"),
            "residual_px": float("nan"),
            "chance": float("nan"),
        }
        on_a_repeat = False

    # 2 -- the axis was never recoverable
    if not dominant_anchored:
        return verdict(
            FailureMode.UNANCHORED_AXIS,
            f"{dominant_axis} axis has no structural anchor (anchor={anchor}); "
            f"error of {abs(dominant_error):.1f} px on that axis is not recoverable "
            f"from the two images",
        )

    # 3 -- landed on a genuine repeat despite having a landmark
    if on_a_repeat:
        caveat = ""
        if alias["chance"] >= 0.35:
            caveat = (
                f"; weak evidence -- at this pitch {alias['chance'] * 100:.0f}% of "
                f"random offsets would pass the same test"
            )
        return verdict(
            FailureMode.PERIODIC_ALIAS,
            f"{dominant_axis} error {abs(dominant_error):.1f} px is "
            f"{alias['multiples']:.2f} cell pitches ({pitch:.2f} px), residual "
            f"{alias['residual_px']:.3f} px -- a true alias, on an axis that did "
            f"carry an anchor{caveat}",
        )

    # 4 -- the structure is not in the image to begin with
    ratio = _spot_ratio(meta, row)
    if ratio is not None:
        detail["spot_over_feature"] = ratio
        if ratio >= config.blur_ratio:
            return verdict(
                FailureMode.BLUR_LIMITED,
                f"search beam spot is {ratio:.2f}x the feature size, at or above "
                f"the {config.blur_ratio:g} crossover where the array stops being "
                f"resolved",
            )

    # 5 -- right cell, imprecise centre
    if pitch is not None and error_px < config.subpixel_pitch_fraction * pitch:
        return verdict(
            FailureMode.SUBPIXEL_DRIFT,
            f"error {error_px:.3f} px is under half a cell pitch ({pitch:.2f} px), "
            f"so the correct cell was identified and only the sub-pixel centre is off",
        )

    # 6 -- residual bucket
    return verdict(
        FailureMode.UNEXPLAINED,
        f"error {error_px:.3f} px on an anchored {dominant_axis} axis, not an "
        f"integer alias, not blur-limited, and larger than half a cell pitch",
    )


def classify_dataset(
    dataset_dir: str | Path,
    results: Sequence[object],
    config: Optional[TaxonomyConfig] = None,
) -> List[FailureVerdict]:
    """Classify every result from ``evaluate_dataset`` against its dataset.

    ``results`` is the list of ``PairResult`` returned by
    ``driftsense.evaluate.evaluate_dataset``. Taking it as a plain sequence keeps
    this module free of an import from ``evaluate``, which imports the locator.
    """
    import csv

    dataset_dir = Path(dataset_dir)
    csv_path = dataset_dir / "ground_truth.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"no ground_truth.csv in {dataset_dir}")
    with csv_path.open(encoding="utf-8") as handle:
        rows = {int(r["pair_id"]): r for r in csv.DictReader(handle)}

    manifest_pixel_nm: Optional[float] = None
    manifest_path = dataset_dir / "dataset_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            geometry = manifest.get("config", {}).get("geometry", {})
            manifest_pixel_nm = geometry.get("search_pixel_size_nm")
        except (json.JSONDecodeError, AttributeError):
            manifest_pixel_nm = None

    verdicts: List[FailureVerdict] = []
    for result in results:
        pair_id = int(getattr(result, "pair_id"))
        row = dict(rows.get(pair_id, {"pair_id": pair_id}))
        if manifest_pixel_nm and "search_pixel_size_nm" not in row:
            row["search_pixel_size_nm"] = manifest_pixel_nm

        meta = None
        meta_path = dataset_dir / "pairs" / f"pair_{pair_id:04d}_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = None

        verdicts.append(
            classify_pair(
                row,
                error_x=float(getattr(result, "error_x")),
                error_y=float(getattr(result, "error_y")),
                error_px=float(getattr(result, "error_px")),
                confidence_psr=float(getattr(result, "confidence_psr")),
                runner_up_margin=getattr(result, "runner_up_margin", None),
                meta=meta,
                config=config,
            )
        )
    return verdicts


def summarise_modes(verdicts: Sequence[FailureVerdict]) -> Dict[str, Dict[str, float]]:
    """Counts and confidence signature per mode.

    The confidence signature is the point of the exercise: if failures carry a
    visibly different margin from successes, they are flaggable at inference time
    without ground truth, which is the difference between a method that fails and
    one that fails without warning.
    """
    summary: Dict[str, Dict[str, float]] = {}
    total = len(verdicts) or 1
    for mode in FailureMode.ORDER:
        subset = [v for v in verdicts if v.mode == mode]
        if not subset:
            continue
        margins = [v.runner_up_margin for v in subset if v.runner_up_margin is not None]
        errors = [v.error_px for v in subset]
        psrs = [v.confidence_psr for v in subset]
        summary[mode] = {
            "n": len(subset),
            "fraction": len(subset) / total,
            "median_error_px": float(sorted(errors)[len(errors) // 2]),
            "median_psr": float(sorted(psrs)[len(psrs) // 2]),
            "median_margin": (
                float(sorted(margins)[len(margins) // 2]) if margins else float("nan")
            ),
            "n_flagged_low_confidence": sum(1 for v in subset if v.flagged_low_confidence),
        }
    return summary


def format_taxonomy_report(verdicts: Sequence[FailureVerdict]) -> str:
    """Render the taxonomy as a plain-text block, in the style of the evaluation report."""
    summary = summarise_modes(verdicts)
    n = len(verdicts)
    lines = [
        "",
        "  Failure taxonomy",
        "  " + "-" * 52,
        f"  pairs classified       {n}",
        "",
        "  mode                  n      share   median err   med PSR   med margin   flagged",
    ]
    for mode in FailureMode.ORDER:
        stats = summary.get(mode)
        if stats is None:
            continue
        margin = stats["median_margin"]
        margin_text = "     --" if margin != margin else f"{margin:7.4f}"
        lines.append(
            f"  {mode:<20}{int(stats['n']):>3}   {stats['fraction'] * 100:5.1f}%   "
            f"{stats['median_error_px']:9.3f}   {stats['median_psr']:7.3f}   "
            f"{margin_text}      {int(stats['n_flagged_low_confidence']):>3}/{int(stats['n'])}"
        )

    failures = [v for v in verdicts if v.mode != FailureMode.CORRECT]
    if failures:
        flagged = sum(1 for v in failures if v.flagged_low_confidence)
        lines += [
            "",
            f"  of {len(failures)} failures, {flagged} were flagged low-confidence by the",
            "  locator itself -- detectable at inference time, without ground truth",
            "",
            "  worst pair per mode",
        ]
        for mode in FailureMode.ORDER:
            if mode == FailureMode.CORRECT:
                continue
            subset = [v for v in failures if v.mode == mode]
            if not subset:
                continue
            worst = max(subset, key=lambda v: v.error_px)
            lines.append(f"    pair {worst.pair_id:>4}  {mode}")
            lines.append(f"      {worst.evidence}")
    lines.append("")
    return "\n".join(lines)

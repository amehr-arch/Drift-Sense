"""Failure taxonomy: every mode, its boundaries, and the ordering between them."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from driftsense.failures import (
    FailureMode,
    TaxonomyConfig,
    classify_dataset,
    classify_pair,
    format_taxonomy_report,
    summarise_modes,
)

# A pitch of 10 px on both axes keeps the arithmetic in these tests readable:
# bit-line pitch 100 nm at 10 nm/px is 10 px along x, likewise word lines along y.
ROW = {
    "pair_id": "3",
    "anchor": "both",
    "anchor_x": "1",
    "anchor_y": "1",
    "bitline_pitch_nm": "100.0",
    "wordline_pitch_nm": "100.0",
    "search_pixel_size_nm": "10.0",
    "feature_size_nm": "40.0",
}


def row(**overrides):
    merged = dict(ROW)
    merged.update({k: str(v) for k, v in overrides.items()})
    return merged


def meta(search_spot_nm: float):
    return {"driftsense": {"capture": {"search": {"spot_size_nm": search_spot_nm}}}}


def classify(*, error_x=0.0, error_y=0.0, error_px=None, psr=6.0, margin=0.08,
             row_overrides=None, meta_dict=None, config=None):
    if error_px is None:
        error_px = (error_x ** 2 + error_y ** 2) ** 0.5
    return classify_pair(
        row(**(row_overrides or {})),
        error_x=error_x,
        error_y=error_y,
        error_px=error_px,
        confidence_psr=psr,
        runner_up_margin=margin,
        meta=meta_dict,
        config=config,
    )


class TestCorrect(unittest.TestCase):
    def test_small_error_is_correct(self):
        self.assertEqual(classify(error_x=0.3, error_y=0.2).mode, FailureMode.CORRECT)

    def test_the_tolerance_boundary_is_inclusive(self):
        v = classify(error_x=1.0, error_y=0.0)
        self.assertEqual(v.mode, FailureMode.CORRECT)

    def test_just_past_the_boundary_is_a_failure(self):
        v = classify(error_x=1.001, error_y=0.0)
        self.assertNotEqual(v.mode, FailureMode.CORRECT)

    def test_evidence_names_the_tolerance(self):
        self.assertIn("tolerance", classify(error_x=0.1).evidence)


class TestUnanchoredAxis(unittest.TestCase):
    def test_error_on_an_unanchored_axis(self):
        v = classify(error_y=37.0, row_overrides={"anchor": "x", "anchor_y": 0})
        self.assertEqual(v.mode, FailureMode.UNANCHORED_AXIS)

    def test_it_outranks_periodic_alias(self):
        """An exact alias on an unanchored axis is still unanchored first.

        Both descriptions are true of the same pair. The taxonomy reports the
        unanchored one because it says the error was unavoidable, where the alias
        label would imply the locator ignored a landmark that was there.
        """
        v = classify(error_y=30.0, row_overrides={"anchor": "x", "anchor_y": 0})
        self.assertEqual(v.mode, FailureMode.UNANCHORED_AXIS)

    def test_the_anchored_axis_does_not_trigger_it(self):
        v = classify(error_x=30.0, row_overrides={"anchor": "x", "anchor_y": 0})
        self.assertEqual(v.mode, FailureMode.PERIODIC_ALIAS)

    def test_evidence_names_the_axis(self):
        v = classify(error_y=37.0, row_overrides={"anchor": "x", "anchor_y": 0})
        self.assertIn("y axis", v.evidence)


class TestPeriodicAlias(unittest.TestCase):
    def test_an_exact_multiple_of_the_pitch(self):
        v = classify(error_x=30.0)
        self.assertEqual(v.mode, FailureMode.PERIODIC_ALIAS)

    def test_a_near_multiple_still_counts(self):
        """20.4 px against a 10 px pitch is 2.04 pitches -- residual 0.04."""
        v = classify(error_x=20.4)
        self.assertEqual(v.mode, FailureMode.PERIODIC_ALIAS)

    def test_a_half_pitch_offset_is_not_an_alias(self):
        """15 px is 1.5 pitches: exactly between two cells, so not a repeat."""
        v = classify(error_x=15.0)
        self.assertNotEqual(v.mode, FailureMode.PERIODIC_ALIAS)

    def test_evidence_reports_the_pitch_count(self):
        self.assertIn("cell pitches", classify(error_x=30.0).evidence)

    def test_detail_records_the_residual(self):
        v = classify(error_x=30.0)
        self.assertIn("alias_residual_x", v.detail)
        self.assertLess(v.detail["alias_residual_x"], 0.01)

    def test_a_fine_pitch_requires_an_absolute_residual(self):
        """The defect this fixed: on a fine pitch the fraction test is near-vacuous.

        A real pair had a 4.80 px pitch, where a residual within 0.25 of a
        multiple catches half of all random offsets. "Alias" then means almost
        nothing. The absolute test binds instead.
        """
        fine = {"bitline_pitch_nm": 20.0}  # 2 px pitch at 10 nm/px
        # 0.4 px off a multiple: 0.2 of a pitch, passes the fraction test, but
        # 0.4 px is within the absolute window too, so this one is a real alias.
        self.assertEqual(
            classify(error_x=10.4, row_overrides=fine, meta_dict=meta(8.0)).mode,
            FailureMode.PERIODIC_ALIAS,
        )
        # Widen the absolute window to nothing and the same pair stops qualifying.
        strict = TaxonomyConfig(alias_residual_px=0.1)
        self.assertNotEqual(
            classify(error_x=10.4, row_overrides=fine, meta_dict=meta(8.0),
                     config=strict).mode,
            FailureMode.PERIODIC_ALIAS,
        )

    def test_a_coarse_pitch_rejects_a_large_absolute_residual(self):
        """40 px pitch, 2 px residual: 0.05 of a pitch, but 2 px is too far."""
        coarse = {"bitline_pitch_nm": 400.0}
        v = classify(error_x=82.0, row_overrides=coarse, meta_dict=meta(8.0))
        self.assertNotEqual(v.mode, FailureMode.PERIODIC_ALIAS)

    def test_weak_attributions_say_so(self):
        """A 2 px pitch means the test passes for a random offset most of the time."""
        v = classify(error_x=10.4, row_overrides={"bitline_pitch_nm": 20.0},
                     meta_dict=meta(8.0))
        self.assertIn("weak evidence", v.evidence)

    def test_a_strong_attribution_carries_no_caveat(self):
        v = classify(error_x=30.0)  # 10 px pitch -> 20% chance
        self.assertNotIn("weak evidence", v.evidence)

    def test_detail_records_the_chance(self):
        v = classify(error_x=30.0)
        self.assertAlmostEqual(v.detail["alias_chance_x"], 0.2, places=6)

    def test_the_chance_is_capped_at_certainty(self):
        """At a pitch below twice the window, every offset passes."""
        v = classify(error_x=3.0, row_overrides={"bitline_pitch_nm": 10.0},
                     meta_dict=meta(8.0))
        self.assertLessEqual(v.detail.get("alias_chance_x", 0.0), 1.0)

    def test_the_dominant_axis_is_the_one_classified(self):
        """Error is larger in y, so y decides the verdict even though x is nonzero."""
        v = classify(error_x=2.0, error_y=30.0)
        self.assertIn("y error", v.evidence)


class TestBlurLimited(unittest.TestCase):
    def test_a_large_spot_relative_to_the_feature(self):
        v = classify(error_x=15.0, meta_dict=meta(32.0))  # 32/40 = 0.8
        self.assertEqual(v.mode, FailureMode.BLUR_LIMITED)

    def test_a_small_spot_is_not_blur_limited(self):
        v = classify(error_x=15.0, meta_dict=meta(8.0))  # 8/40 = 0.2
        self.assertNotEqual(v.mode, FailureMode.BLUR_LIMITED)

    def test_alias_outranks_blur(self):
        """A pair can be both. The alias is the more specific claim, so it wins."""
        v = classify(error_x=30.0, meta_dict=meta(32.0))
        self.assertEqual(v.mode, FailureMode.PERIODIC_ALIAS)

    def test_missing_metadata_does_not_crash(self):
        v = classify(error_x=15.0, meta_dict=None)
        self.assertIn(v.mode, FailureMode.ORDER)

    def test_malformed_metadata_does_not_crash(self):
        v = classify(error_x=15.0, meta_dict={"driftsense": "not a mapping"})
        self.assertIn(v.mode, FailureMode.ORDER)

    def test_detail_records_the_ratio(self):
        v = classify(error_x=15.0, meta_dict=meta(32.0))
        self.assertAlmostEqual(v.detail["spot_over_feature"], 0.8, places=6)


class TestSubpixelDrift(unittest.TestCase):
    def test_error_under_half_a_pitch_but_over_tolerance(self):
        """4 px against a 10 px pitch: the right cell, an imprecise centre."""
        v = classify(error_x=4.0, meta_dict=meta(8.0))
        self.assertEqual(v.mode, FailureMode.SUBPIXEL_DRIFT)

    def test_evidence_says_the_correct_cell_was_found(self):
        v = classify(error_x=4.0, meta_dict=meta(8.0))
        self.assertIn("correct cell", v.evidence)


class TestUnexplained(unittest.TestCase):
    def test_a_failure_matching_nothing_is_reported_as_such(self):
        """15 px: anchored, not an alias, spot is small, over half a pitch."""
        v = classify(error_x=15.0, meta_dict=meta(8.0))
        self.assertEqual(v.mode, FailureMode.UNEXPLAINED)

    def test_it_does_not_absorb_an_alias(self):
        self.assertEqual(classify(error_x=30.0, meta_dict=meta(8.0)).mode,
                         FailureMode.PERIODIC_ALIAS)


class TestMissingPitchInformation(unittest.TestCase):
    def test_no_pitch_columns_still_classifies(self):
        v = classify_pair(
            {"pair_id": "0", "anchor": "both", "anchor_x": "1", "anchor_y": "1"},
            error_x=30.0, error_y=0.0, error_px=30.0,
            confidence_psr=6.0, runner_up_margin=0.08,
        )
        self.assertEqual(v.mode, FailureMode.UNEXPLAINED)

    def test_zero_pitch_is_treated_as_absent(self):
        v = classify(error_x=30.0, row_overrides={"bitline_pitch_nm": 0.0},
                     meta_dict=meta(8.0))
        self.assertEqual(v.mode, FailureMode.UNEXPLAINED)

    def test_non_numeric_pitch_is_treated_as_absent(self):
        v = classify(error_x=30.0, row_overrides={"bitline_pitch_nm": "n/a"},
                     meta_dict=meta(8.0))
        self.assertEqual(v.mode, FailureMode.UNEXPLAINED)


class TestConfidenceFlag(unittest.TestCase):
    def test_a_small_margin_is_flagged(self):
        self.assertTrue(classify(error_x=30.0, margin=0.001).flagged_low_confidence)

    def test_a_low_psr_is_flagged(self):
        self.assertTrue(classify(error_x=30.0, psr=4.0).flagged_low_confidence)

    def test_a_confident_pair_is_not_flagged(self):
        self.assertFalse(classify(error_x=30.0, psr=7.0, margin=0.12).flagged_low_confidence)

    def test_a_missing_margin_falls_back_to_psr(self):
        v = classify_pair(row(), error_x=30.0, error_y=0.0, error_px=30.0,
                          confidence_psr=4.0, runner_up_margin=None)
        self.assertTrue(v.flagged_low_confidence)

    def test_the_flag_is_independent_of_correctness(self):
        """A correct answer can still have been low-confidence. Both are recorded."""
        v = classify(error_x=0.2, margin=0.001)
        self.assertEqual(v.mode, FailureMode.CORRECT)
        self.assertTrue(v.flagged_low_confidence)


class TestTaxonomyConfig(unittest.TestCase):
    def test_rejects_a_negative_threshold(self):
        with self.assertRaises(ValueError):
            TaxonomyConfig(tolerance_px=-1.0)

    def test_rejects_a_negative_absolute_alias_window(self):
        with self.assertRaises(ValueError):
            TaxonomyConfig(alias_residual_px=-0.5)

    def test_rejects_a_vacuous_alias_window(self):
        """At 0.5 every offset is within half a pitch of some multiple."""
        with self.assertRaises(ValueError):
            TaxonomyConfig(alias_residual_fraction=0.5)

    def test_rejects_a_non_finite_threshold(self):
        with self.assertRaises(ValueError):
            TaxonomyConfig(blur_ratio=float("nan"))

    def test_a_tighter_tolerance_reclassifies(self):
        strict = TaxonomyConfig(tolerance_px=0.1)
        self.assertEqual(classify(error_x=0.5, meta_dict=meta(8.0)).mode, FailureMode.CORRECT)
        self.assertNotEqual(
            classify(error_x=0.5, meta_dict=meta(8.0), config=strict).mode,
            FailureMode.CORRECT,
        )

    def test_the_config_is_frozen(self):
        with self.assertRaises(Exception):
            TaxonomyConfig().tolerance_px = 2.0  # type: ignore[misc]


@dataclass
class FakeResult:
    pair_id: int
    error_x: float
    error_y: float
    error_px: float
    confidence_psr: float
    runner_up_margin: Optional[float]


class TestSummaries(unittest.TestCase):
    def setUp(self):
        self.verdicts = [
            classify(error_x=0.2),
            classify(error_x=0.3),
            classify(error_x=30.0, margin=0.001),
            classify(error_y=37.0, row_overrides={"anchor": "x", "anchor_y": 0}),
        ]

    def test_counts_partition_the_input(self):
        summary = summarise_modes(self.verdicts)
        self.assertEqual(sum(int(s["n"]) for s in summary.values()), len(self.verdicts))

    def test_fractions_sum_to_one(self):
        summary = summarise_modes(self.verdicts)
        self.assertAlmostEqual(sum(s["fraction"] for s in summary.values()), 1.0, places=9)

    def test_modes_appear_in_declared_order(self):
        keys = list(summarise_modes(self.verdicts))
        self.assertEqual(keys, [m for m in FailureMode.ORDER if m in keys])

    def test_absent_modes_are_omitted_rather_than_zeroed(self):
        self.assertNotIn(FailureMode.BLUR_LIMITED, summarise_modes(self.verdicts))

    def test_empty_input_summarises_to_nothing(self):
        self.assertEqual(summarise_modes([]), {})

    def test_report_mentions_every_present_mode(self):
        report = format_taxonomy_report(self.verdicts)
        for mode in summarise_modes(self.verdicts):
            self.assertIn(mode, report)

    def test_report_counts_the_flagged_failures(self):
        self.assertIn("flagged low-confidence", format_taxonomy_report(self.verdicts))

    def test_report_on_empty_input_does_not_crash(self):
        self.assertIn("Failure taxonomy", format_taxonomy_report([]))


class TestClassifyDataset(unittest.TestCase):
    def _dataset(self, tmp: Path):
        pairs = tmp / "pairs"
        pairs.mkdir(parents=True)
        header = ",".join(ROW)
        lines = [header]
        for pair_id in (0, 1):
            r = row(pair_id=pair_id)
            lines.append(",".join(r[k] for k in ROW))
        (tmp / "ground_truth.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (pairs / "pair_0000_meta.json").write_text(json.dumps(meta(32.0)), encoding="utf-8")
        return tmp

    def test_classifies_every_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._dataset(Path(tmp))
            results = [
                FakeResult(0, 15.0, 0.0, 15.0, 6.0, 0.08),
                FakeResult(1, 0.2, 0.0, 0.2, 6.0, 0.08),
            ]
            verdicts = classify_dataset(root, results)
            self.assertEqual(len(verdicts), 2)
            self.assertEqual(verdicts[0].mode, FailureMode.BLUR_LIMITED)
            self.assertEqual(verdicts[1].mode, FailureMode.CORRECT)

    def test_a_pair_without_metadata_still_classifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._dataset(Path(tmp))
            verdicts = classify_dataset(root, [FakeResult(1, 15.0, 0.0, 15.0, 6.0, 0.08)])
            self.assertEqual(verdicts[0].mode, FailureMode.UNEXPLAINED)

    def test_corrupt_metadata_is_ignored_rather_than_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._dataset(Path(tmp))
            (root / "pairs" / "pair_0000_meta.json").write_text("{not json", encoding="utf-8")
            verdicts = classify_dataset(root, [FakeResult(0, 15.0, 0.0, 15.0, 6.0, 0.08)])
            self.assertEqual(verdicts[0].mode, FailureMode.UNEXPLAINED)

    def test_missing_ground_truth_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                classify_dataset(Path(tmp), [])

    def test_rows_are_matched_by_pair_id_not_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._dataset(Path(tmp))
            verdicts = classify_dataset(root, [FakeResult(1, 0.1, 0.0, 0.1, 6.0, 0.08)])
            self.assertEqual(verdicts[0].pair_id, 1)

    def test_as_row_is_csv_safe(self):
        verdict = classify(error_x=30.0)
        row_out = verdict.as_row()
        self.assertIsInstance(row_out["detail"], str)
        self.assertIsInstance(json.loads(row_out["detail"]), dict)


class TestTaxonomyDoesNotTouchInference(unittest.TestCase):
    def test_failures_does_not_import_the_locator(self):
        """The classifier reads layout parameters. The locator must never be able to.

        Guarding this with a test rather than a convention, because the import
        would be an easy and invisible mistake to make later.
        """
        source = (Path(__file__).resolve().parents[1] / "driftsense" / "failures.py").read_text(
            encoding="utf-8"
        )
        for forbidden in ("from .locate", "from .correlate", "import locate"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()

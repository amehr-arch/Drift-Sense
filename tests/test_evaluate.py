"""Evaluation harness and the two command-line entry points."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from driftsense import GenerationConfig, generate_dataset
from driftsense.evaluate import evaluate_dataset, format_report, write_evaluation


class DatasetCase(unittest.TestCase):
    """Builds one small real dataset, shared by every test in the class."""

    n_pairs = 6

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.dataset = Path(cls._tmp.name) / "data"
        generate_dataset(
            GenerationConfig(
                output_dir=cls.dataset, n_pairs=cls.n_pairs, seed=42, save_overlays=False
            )
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def setUp(self) -> None:
        self._out = tempfile.TemporaryDirectory()
        self.out = Path(self._out.name)
        self.addCleanup(self._out.cleanup)


class TestEvaluateDataset(DatasetCase):
    def test_produces_one_result_per_pair(self):
        results, summary = evaluate_dataset(self.dataset)
        self.assertEqual(len(results), self.n_pairs)
        self.assertEqual(summary.n_pairs, self.n_pairs)

    def test_limit_truncates_the_run(self):
        results, summary = evaluate_dataset(self.dataset, limit=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(summary.n_pairs, 2)

    def test_accuracy_is_monotonic_in_tolerance(self):
        _, summary = evaluate_dataset(self.dataset)
        values = [summary.accuracy[f"{t:g}"] for t in summary.tolerances]
        self.assertEqual(values, sorted(values))
        for value in values:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_anchored_pairs_are_located_accurately(self):
        """The an earlier revision gate: on pairs whose window carries anchors on both axes,
        the locator must be sub-pixel."""
        results, _ = evaluate_dataset(self.dataset)
        anchored = [r for r in results if r.anchor == "both"]
        self.assertGreater(len(anchored), 0, "dataset produced no fully anchored pairs")
        for result in anchored:
            with self.subTest(pair=result.pair_id):
                self.assertLess(result.error_px, 1.0)

    def test_ambiguous_pairs_report_a_smaller_runner_up_margin(self):
        """Ambiguity must be self-reported, not merely suffered."""
        results, _ = evaluate_dataset(self.dataset)
        anchored = [r.runner_up_margin for r in results if r.anchor == "both"]
        ambiguous = [r.runner_up_margin for r in results if r.anchor != "both"]
        if not anchored or not ambiguous:
            self.skipTest("dataset does not contain both classes")
        self.assertGreater(min(anchored), max(ambiguous))

    def test_summary_splits_by_anchor_class(self):
        _, summary = evaluate_dataset(self.dataset)
        self.assertTrue(summary.by_anchor)
        total = sum(entry["n"] for entry in summary.by_anchor.values())
        self.assertEqual(total, self.n_pairs)

    def test_timing_is_recorded_and_plausible(self):
        _, summary = evaluate_dataset(self.dataset)
        self.assertGreater(summary.mean_elapsed_s, 0.0)
        self.assertLess(summary.mean_elapsed_s, 30.0)

    def test_custom_tolerances_are_honoured(self):
        _, summary = evaluate_dataset(self.dataset, tolerances=(1.0, 3.0))
        self.assertEqual(summary.tolerances, (1.0, 3.0))
        self.assertEqual(set(summary.accuracy), {"1", "3"})

    def test_missing_dataset_is_reported_clearly(self):
        with self.assertRaises(FileNotFoundError):
            evaluate_dataset(self.out / "does-not-exist")

    def test_summary_is_json_serialisable(self):
        _, summary = evaluate_dataset(self.dataset, limit=2)
        payload = json.loads(json.dumps(summary.as_dict()))
        self.assertEqual(payload["n_pairs"], 2)
        self.assertIn("by_anchor", payload)


class TestReportingAndOutput(DatasetCase):
    def test_report_mentions_the_headline_numbers(self):
        _, summary = evaluate_dataset(self.dataset, limit=3)
        report = format_report(summary)
        self.assertIn("accuracy within tolerance", report)
        self.assertIn("computation time", report)
        self.assertIn("by anchor class", report)

    def test_write_evaluation_produces_every_artefact(self):
        results, summary = evaluate_dataset(self.dataset, limit=3)
        written = write_evaluation(self.out, results, summary, plot=True)
        for key in ("results", "summary", "report", "curve"):
            self.assertTrue(written[key].exists(), key)

    def test_results_csv_is_complete_and_consistent(self):
        results, summary = evaluate_dataset(self.dataset, limit=3)
        write_evaluation(self.out, results, summary, plot=False)
        with (self.out / "results.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3)
        for row in rows:
            with self.subTest(pair=row["pair_id"]):
                self.assertIn(row["anchor"], {"both", "x", "y", "none"})
                self.assertGreaterEqual(float(row["error_px"]), 0.0)
                self.assertGreater(float(row["elapsed_s"]), 0.0)

    def test_plot_can_be_skipped(self):
        results, summary = evaluate_dataset(self.dataset, limit=2)
        written = write_evaluation(self.out, results, summary, plot=False)
        self.assertNotIn("curve", written)


class TestCommandLine(DatasetCase):
    def test_locate_pattern_prints_two_numbers(self):
        import locate_pattern

        with (self.dataset / "ground_truth.csv").open(encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exit_code = locate_pattern.main(
                [
                    str(self.dataset / row["reference_path"]),
                    str(self.dataset / row["search_path"]),
                ]
            )
        self.assertEqual(exit_code, 0)
        parts = buffer.getvalue().split()
        self.assertEqual(len(parts), 2)
        for value in parts:
            self.assertGreaterEqual(float(value), 0.0)
            self.assertLessEqual(float(value), 1000.0)

    def test_locate_pattern_json_output_is_parseable(self):
        import locate_pattern

        with (self.dataset / "ground_truth.csv").open(encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            locate_pattern.main(
                [
                    str(self.dataset / row["reference_path"]),
                    str(self.dataset / row["search_path"]),
                    "--format",
                    "json",
                ]
            )
        payload = json.loads(buffer.getvalue())
        self.assertIn("x", payload)
        self.assertNotIn("candidates", payload)  # only with --verbose

    def test_locate_pattern_reports_a_missing_file(self):
        import locate_pattern

        with contextlib.redirect_stderr(io.StringIO()):
            exit_code = locate_pattern.main(["nope.png", "also-nope.png"])
        self.assertEqual(exit_code, 1)

    def test_evaluate_dataset_cli_runs_end_to_end(self):
        import evaluate_dataset as cli

        with contextlib.redirect_stdout(io.StringIO()):
            exit_code = cli.main(
                [str(self.dataset), "--out", str(self.out), "--limit", "3", "--quiet", "--panels", "1"]
            )
        self.assertEqual(exit_code, 0)
        self.assertTrue((self.out / "results.csv").exists())
        self.assertTrue((self.out / "accuracy_curve.png").exists())
        self.assertEqual(len(list((self.out / "panels").glob("*.png"))), 1)

    def test_evaluate_dataset_cli_rejects_a_missing_dataset(self):
        import evaluate_dataset as cli

        with contextlib.redirect_stderr(io.StringIO()):
            exit_code = cli.main([str(self.out / "nothing")])
        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()

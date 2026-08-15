"""End-to-end generation, validation and determinism tests."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from driftsense import (
    DramLayout,
    DramParams,
    GenerationConfig,
    ImagingGeometry,
    Placement,
    generate_dataset,
    generate_pair,
)
from driftsense.geometry import GroundTruth
from driftsense.sampling import PlacementSampler, sample_dram_layout
from driftsense.validate import (
    ValidationThresholds,
    block_reduce,
    make_validator,
    sample_patch,
    validate_pair,
    zncc,
)

GEOMETRY = ImagingGeometry()


def make_layout() -> DramLayout:
    return DramLayout(
        DramParams(feature_size_nm=35.0, mat_size_nm=2600.0, strip_width_nm=320.0)
    )


class TempDirCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)


class TestGeneratePair(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = make_layout()

    def test_images_have_the_required_shape_and_dtype(self):
        pair = generate_pair(self.layout, Placement(2491.0, 5685.0), GEOMETRY)
        for name, image in (("reference", pair.reference), ("search", pair.search)):
            with self.subTest(image=name):
                self.assertEqual(image.shape, (1000, 1000))
                self.assertEqual(image.dtype, np.uint8)

    def test_ground_truth_matches_the_organiser_sample(self):
        pair = generate_pair(self.layout, Placement(2491.0, 5685.0), GEOMETRY)
        self.assertAlmostEqual(pair.ground_truth.x, 299.1, places=9)
        self.assertAlmostEqual(pair.ground_truth.y, 618.5, places=9)

    def test_reference_really_is_present_at_the_ground_truth(self):
        """The load-bearing test of an earlier revision.

        Reduce the reference by the zoom ratio, sample the search image at the
        claimed location, and require the two to be near-identical. Any error in
        the coordinate convention destroys this correlation.
        """
        pair = generate_pair(self.layout, Placement(2491.0, 5685.0), GEOMETRY)
        template = block_reduce(pair.reference, GEOMETRY.zoom_ratio)
        patch = sample_patch(pair.search, pair.ground_truth.box_x, pair.ground_truth.box_y, 100)
        self.assertGreater(zncc(template, patch), 0.95)

    def test_correlation_holds_across_the_whole_field(self):
        for origin in ((0.0, 0.0), (9000.0, 9000.0), (4137.0, 2718.0), (2491.5, 5685.5)):
            with self.subTest(origin=origin):
                pair = generate_pair(self.layout, Placement(*origin), GEOMETRY)
                report = validate_pair(pair)
                self.assertTrue(report.ok, str(report))

    def test_a_deliberately_wrong_location_fails_the_check(self):
        """Guards against the correlation check being vacuously satisfied."""
        pair = generate_pair(self.layout, Placement(2491.0, 5685.0), GEOMETRY)
        template = block_reduce(pair.reference, GEOMETRY.zoom_ratio)
        wrong = sample_patch(pair.search, 20.0, 20.0, 100)
        self.assertLess(zncc(template, wrong), 0.95)

    def test_metadata_is_json_serialisable_and_complete(self):
        pair = generate_pair(self.layout, Placement(1000.0, 2000.0), GEOMETRY, pair_id=7, seed=99)
        payload = json.loads(json.dumps(pair.as_metadata()))
        self.assertEqual(payload["architecture"], "dram")
        self.assertEqual(payload["seed"], 99)
        self.assertEqual(payload["driftsense"]["pair_id"], 7)
        self.assertAlmostEqual(payload["driftsense"]["layout"]["mat_size_nm"], 2600.0)

    def test_pair_name_is_zero_padded(self):
        pair = generate_pair(self.layout, Placement(0.0, 0.0), GEOMETRY, pair_id=7)
        self.assertEqual(pair.name, "pair_0007")


class TestSampling(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = make_layout()

    def test_layout_sampling_is_reproducible(self):
        a = sample_dram_layout(np.random.default_rng(7)).describe()
        b = sample_dram_layout(np.random.default_rng(7)).describe()
        self.assertEqual(a, b)

    def test_layout_sampling_actually_varies(self):
        a = sample_dram_layout(np.random.default_rng(1)).describe()
        b = sample_dram_layout(np.random.default_rng(2)).describe()
        self.assertNotEqual(a, b)

    def test_placements_stay_inside_the_legal_range(self):
        sampler = PlacementSampler(geometry=GEOMETRY, boundary_bias=0.5)
        rng = np.random.default_rng(0)
        for _ in range(200):
            placement = sampler.sample(self.layout, rng)
            self.assertGreaterEqual(placement.origin_x_nm, 0.0)
            self.assertLessEqual(placement.origin_x_nm, GEOMETRY.max_origin_nm)
            self.assertGreaterEqual(placement.origin_y_nm, 0.0)
            self.assertLessEqual(placement.origin_y_nm, GEOMETRY.max_origin_nm)

    def test_integral_placement_is_the_default(self):
        sampler = PlacementSampler(geometry=GEOMETRY)
        rng = np.random.default_rng(3)
        for _ in range(50):
            placement = sampler.sample(self.layout, rng)
            self.assertAlmostEqual(placement.origin_x_nm, round(placement.origin_x_nm), places=9)

    def test_subpixel_mode_produces_fractional_origins(self):
        sampler = PlacementSampler(geometry=GEOMETRY, subpixel=True)
        rng = np.random.default_rng(3)
        origins = [sampler.sample(self.layout, rng).origin_x_nm for _ in range(50)]
        self.assertTrue(any(abs(o - round(o)) > 1e-6 for o in origins))

    def test_full_boundary_bias_puts_an_edge_inside_the_window(self):
        sampler = PlacementSampler(geometry=GEOMETRY, boundary_bias=1.0)
        rng = np.random.default_rng(11)
        hits = 0
        for _ in range(40):
            placement = sampler.sample(self.layout, rng)
            lo = placement.origin_x_nm
            hi = lo + GEOMETRY.reference_fov_nm
            if self.layout.boundary_coordinates_nm("x", lo, hi):
                hits += 1
        self.assertGreaterEqual(hits, 36)  # a few are lost to clipping at the field edges

    def test_zero_bias_lands_on_boundaries_far_less_often(self):
        sampler = PlacementSampler(geometry=GEOMETRY, boundary_bias=0.0)
        rng = np.random.default_rng(11)
        hits = 0
        for _ in range(60):
            placement = sampler.sample(self.layout, rng)
            lo = placement.origin_x_nm
            hi = lo + GEOMETRY.reference_fov_nm
            if self.layout.boundary_coordinates_nm("x", lo, hi):
                hits += 1
        self.assertLess(hits, 45)

    def test_invalid_bias_is_rejected(self):
        for bias in (-0.1, 1.1):
            with self.subTest(bias=bias):
                with self.assertRaises(ValueError):
                    PlacementSampler(boundary_bias=bias)

    def test_invalid_margin_is_rejected(self):
        with self.assertRaises(ValueError):
            PlacementSampler(interior_margin=0.6)


class TestValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = make_layout()

    def _pair(self):
        return generate_pair(self.layout, Placement(3000.0, 3000.0), GEOMETRY)

    def test_a_corrupted_search_image_is_caught(self):
        pair = self._pair()
        pair.search = np.zeros_like(pair.search)
        self.assertFalse(validate_pair(pair).ok)

    def test_a_shifted_ground_truth_is_caught(self):
        pair = self._pair()
        gt = pair.ground_truth
        pair.ground_truth = GroundTruth(
            gt.x + 37, gt.y + 21, gt.box_x + 37, gt.box_y + 21, gt.box_w, gt.box_h
        )
        report = validate_pair(pair)
        self.assertFalse(report.ok)
        self.assertTrue(any("ZNCC" in issue for issue in report.issues))

    def test_wrong_image_size_is_caught(self):
        pair = self._pair()
        pair.reference = pair.reference[:512, :512]
        report = validate_pair(pair)
        self.assertFalse(report.ok)
        self.assertTrue(any("shape" in issue for issue in report.issues))

    def test_a_centre_inconsistent_with_its_box_is_caught(self):
        pair = self._pair()
        gt = pair.ground_truth
        pair.ground_truth = GroundTruth(gt.x + 5, gt.y, gt.box_x, gt.box_y, gt.box_w, gt.box_h)
        report = validate_pair(pair)
        self.assertFalse(report.ok)
        self.assertTrue(any("centre disagrees" in issue for issue in report.issues))

    def test_report_exposes_useful_metrics(self):
        report = validate_pair(self._pair())
        self.assertTrue(report.ok, str(report))
        self.assertGreater(report.metrics["gt_zncc"], 0.95)
        self.assertIn("zncc_margin", report.metrics)
        self.assertIn("pair 0000 [OK]", str(report))


class TestDataset(TempDirCase):
    def test_generates_and_writes_every_artefact(self):
        config = GenerationConfig(
            output_dir=self.tmp_path, n_pairs=3, seed=5, save_overlays=True
        )
        manifest = generate_dataset(config, validator=make_validator())

        self.assertEqual(manifest["n_pairs"], 3)
        self.assertTrue((self.tmp_path / "ground_truth.csv").exists())
        self.assertTrue((self.tmp_path / "dataset_manifest.json").exists())
        for index in range(3):
            name = f"pair_{index:04d}"
            for suffix in ("_reference.png", "_search.png", "_meta.json"):
                self.assertTrue((self.tmp_path / "pairs" / f"{name}{suffix}").exists(), name + suffix)
            self.assertTrue((self.tmp_path / "overlays" / f"{name}_overlay.png").exists())

    def test_csv_rows_are_complete_and_consistent(self):
        generate_dataset(GenerationConfig(output_dir=self.tmp_path, n_pairs=4, seed=8))
        with (self.tmp_path / "ground_truth.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 4)
        for row in rows:
            with self.subTest(pair=row["pair_id"]):
                self.assertTrue(all(value != "" for value in row.values()))
                self.assertAlmostEqual(
                    float(row["gt_x"]), float(row["box_x"]) + float(row["box_w"]) / 2, places=5
                )
                self.assertGreaterEqual(float(row["gt_x"]), 0.0)
                self.assertLessEqual(float(row["gt_x"]), 1000.0)
                self.assertAlmostEqual(float(row["box_w"]), 100.0)

    def test_generated_pairs_are_bit_identical_for_a_given_seed(self):
        first, second = self.tmp_path / "a", self.tmp_path / "b"
        for target in (first, second):
            generate_dataset(GenerationConfig(output_dir=target, n_pairs=2, seed=1234))
        for name in ("pair_0000_reference.png", "pair_0001_search.png"):
            with self.subTest(file=name):
                self.assertEqual(
                    (first / "pairs" / name).read_bytes(), (second / "pairs" / name).read_bytes()
                )

    def test_different_seeds_produce_different_data(self):
        for seed in (1, 2):
            generate_dataset(
                GenerationConfig(output_dir=self.tmp_path / str(seed), n_pairs=1, seed=seed)
            )
        a = (self.tmp_path / "1" / "pairs" / "pair_0000_search.png").read_bytes()
        b = (self.tmp_path / "2" / "pairs" / "pair_0000_search.png").read_bytes()
        self.assertNotEqual(a, b)

    def test_validation_failure_aborts_generation(self):
        with self.assertRaises(RuntimeError):
            generate_dataset(
                GenerationConfig(output_dir=self.tmp_path, n_pairs=1),
                validator=lambda pair: ["synthetic failure"],
            )

    def test_every_pair_of_a_realistic_run_validates(self):
        """A full self-evaluation-sized run must pass validation end to end."""
        generate_dataset(
            GenerationConfig(
                output_dir=self.tmp_path, n_pairs=12, seed=2024, save_overlays=False
            ),
            validator=make_validator(ValidationThresholds(min_zncc=0.95)),
        )
        with (self.tmp_path / "ground_truth.csv").open(encoding="utf-8") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 12)

    def test_unknown_architecture_is_rejected(self):
        """This test used to name 'finfet' as the unknown architecture.

        It is now implemented, so the test was passing for a reason that had
        stopped being true. Named a genuinely absent architecture instead.
        """
        config = GenerationConfig(output_dir=self.tmp_path, n_pairs=1)
        config.architecture = "gaafet"
        with self.assertRaises(KeyError):
            generate_dataset(config)

    def test_finfet_is_accepted(self):
        config = GenerationConfig(output_dir=self.tmp_path, n_pairs=2, architecture="finfet")
        manifest = generate_dataset(config)
        self.assertEqual(manifest["n_pairs"], 2)

    def test_finfet_records_neutral_pitch_columns(self):
        """The taxonomy reasons about aliasing without knowing the architecture."""
        config = GenerationConfig(output_dir=self.tmp_path, n_pairs=2, architecture="finfet")
        generate_dataset(config)
        with (self.tmp_path / "ground_truth.csv").open(encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        self.assertGreater(float(row["pitch_x_nm"]), 0.0)
        self.assertGreater(float(row["pitch_y_nm"]), 0.0)

    def test_dram_records_neutral_pitch_columns_too(self):
        config = GenerationConfig(output_dir=self.tmp_path, n_pairs=2)
        generate_dataset(config)
        with (self.tmp_path / "ground_truth.csv").open(encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        self.assertAlmostEqual(
            float(row["pitch_x_nm"]), float(row["bitline_pitch_nm"]), places=3
        )
        self.assertAlmostEqual(
            float(row["pitch_y_nm"]), float(row["wordline_pitch_nm"]), places=3
        )

    def test_zero_pairs_is_rejected(self):
        with self.assertRaises(ValueError):
            GenerationConfig(output_dir=self.tmp_path, n_pairs=0)


class TestCli(TempDirCase):
    def test_cli_runs_end_to_end(self):
        import generate_dataset as cli

        exit_code = cli.main(
            ["--pairs", "2", "--out", str(self.tmp_path), "--seed", "77", "--overlays", "--quiet"]
        )
        self.assertEqual(exit_code, 0)
        self.assertTrue((self.tmp_path / "ground_truth.csv").exists())
        self.assertTrue((self.tmp_path / "overlays" / "pair_0000_overlay.png").exists())

    def test_cli_rejects_an_unknown_architecture(self):
        import contextlib
        import io

        import generate_dataset as cli

        # argparse writes its usage message to stderr; swallow it so the test
        # output stays readable.
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli.main(["--architecture", "nonsense", "--out", str(self.tmp_path)])


if __name__ == "__main__":
    unittest.main()

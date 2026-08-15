"""Batch prediction script, and the guards against a mismatched convention."""

from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

from driftsense.locate import LocalisationConfig, locate

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    """Import predict_dataset.py by path; it is a script, not a package module."""
    spec = importlib.util.spec_from_file_location(
        "predict_dataset", ROOT / "predict_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["predict_dataset"] = module
    spec.loader.exec_module(module)
    return module


predict_dataset = _load_script()


def write_pair(directory: Path, stem: str, ref_suffix="_reference.png", search_suffix="_search.png"):
    """A pair whose search image genuinely contains the reference, shrunk 10x.

    Both images are the same pixel count, as the problem statement specifies, so
    these fixtures do not trip the mismatched-size warning.
    """
    rng = np.random.default_rng(abs(hash(stem)) % (2**32))
    search = (rng.random((200, 200)) * 120 + 60).astype(np.uint8)
    search[60:80, 80:100] = np.clip(
        search[60:80, 80:100].astype(int) + 90, 0, 255
    ).astype(np.uint8)
    reference = np.kron(search[60:80, 80:100], np.ones((10, 10), dtype=np.uint8))
    Image.fromarray(reference, mode="L").save(directory / f"{stem}{ref_suffix}")
    Image.fromarray(search, mode="L").save(directory / f"{stem}{search_suffix}")


def read_rows(path: Path):
    """Read a CSV and close the handle; an open file per assertion is a leak."""
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestPairDiscovery(unittest.TestCase):
    def test_finds_pairs_by_the_default_naming(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for stem in ("pair_0000", "pair_0001"):
                write_pair(root, stem)
            pairs = predict_dataset._pairs_from_directory(
                root, "**/*_reference.png", "_reference.png", "_search.png"
            )
            self.assertEqual(len(pairs), 2)

    def test_honours_a_custom_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_pair(root, "a", "_hi.png", "_lo.png")
            pairs = predict_dataset._pairs_from_directory(
                root, "**/*_hi.png", "_hi.png", "_lo.png"
            )
            self.assertEqual(len(pairs), 1)

    def test_an_empty_directory_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                predict_dataset._pairs_from_directory(
                    Path(tmp), "**/*_reference.png", "_reference.png", "_search.png"
                )
            self.assertIn("--pattern", str(ctx.exception))

    def test_references_without_a_partner_are_reported_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_pair(root, "good")
            Image.fromarray(np.zeros((10, 10), np.uint8), mode="L").save(
                root / "lonely_reference.png"
            )
            pairs = predict_dataset._pairs_from_directory(
                root, "**/*_reference.png", "_reference.png", "_search.png"
            )
            self.assertEqual([p[0] for p in pairs], ["good"])

    def test_no_partners_at_all_names_the_suffix_option(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_pair(root, "a")
            with self.assertRaises(ValueError) as ctx:
                predict_dataset._pairs_from_directory(
                    root, "**/*_reference.png", "_reference.png", "_absent.png"
                )
            self.assertIn("--search-suffix", str(ctx.exception))


class TestManifest(unittest.TestCase):
    def _manifest(self, root: Path, rows, fieldnames=("pair_id", "reference_path", "search_path")):
        path = root / "manifest.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_reads_the_listed_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_pair(root, "a")
            manifest = self._manifest(root, [{
                "pair_id": "first",
                "reference_path": "a_reference.png",
                "search_path": "a_search.png",
            }])
            pairs = predict_dataset._pairs_from_manifest(manifest)
            self.assertEqual(pairs[0][0], "first")

    def test_paths_are_relative_to_the_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_pair(root, "a")
            manifest = self._manifest(root, [{
                "pair_id": "0",
                "reference_path": "a_reference.png",
                "search_path": "a_search.png",
            }])
            self.assertTrue(predict_dataset._pairs_from_manifest(manifest)[1 - 1][1].exists())

    def test_a_missing_column_is_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._manifest(
                root, [{"reference_path": "a.png"}], fieldnames=("reference_path",)
            )
            with self.assertRaises(ValueError) as ctx:
                predict_dataset._pairs_from_manifest(manifest)
            self.assertIn("search_path", str(ctx.exception))

    def test_an_empty_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._manifest(Path(tmp), [])
            with self.assertRaises(ValueError):
                predict_dataset._pairs_from_manifest(manifest)

    def test_pair_id_defaults_to_the_row_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_pair(root, "a")
            manifest = self._manifest(
                root,
                [{"reference_path": "a_reference.png", "search_path": "a_search.png"}],
                fieldnames=("reference_path", "search_path"),
            )
            self.assertEqual(predict_dataset._pairs_from_manifest(manifest)[0][0], "0")


class TestEndToEnd(unittest.TestCase):
    def test_it_writes_a_prediction_per_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for stem in ("p0", "p1"):
                write_pair(root, stem)
            code = predict_dataset.main([str(root), "--quiet"])
            self.assertEqual(code, 0)
            rows = read_rows(root / "predictions.csv")
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertNotEqual(row["x"], "")
                self.assertNotEqual(row["y"], "")

    def test_the_columns_are_the_documented_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_pair(root, "p0")
            predict_dataset.main([str(root), "--quiet"])
            rows = read_rows(root / "predictions.csv")
            self.assertEqual(list(rows[0]), predict_dataset.COLUMNS)

    def test_predictions_match_calling_locate_directly(self):
        """The script must be a wrapper, not a second implementation."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_pair(root, "p0")
            predict_dataset.main([str(root), "--quiet"])
            row = read_rows(root / "predictions.csv")[0]
            from driftsense.locate import load_greyscale

            expected = locate(
                load_greyscale(root / "p0_reference.png"),
                load_greyscale(root / "p0_search.png"),
            )
            self.assertAlmostEqual(float(row["x"]), round(expected.x, 4), places=3)

    def test_limit_truncates_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for stem in ("p0", "p1", "p2"):
                write_pair(root, stem)
            predict_dataset.main([str(root), "--quiet", "--limit", "2"])
            rows = read_rows(root / "predictions.csv")
            self.assertEqual(len(rows), 2)

    def test_out_overrides_the_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_pair(root, "p0")
            destination = root / "nested" / "mine.csv"
            predict_dataset.main([str(root), "--quiet", "--out", str(destination)])
            self.assertTrue(destination.exists())

    def test_a_broken_pair_does_not_stop_the_run(self):
        """One unreadable image must not cost you the other 29 answers."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_pair(root, "good")
            (root / "bad_reference.png").write_text("not an image", encoding="utf-8")
            (root / "bad_search.png").write_text("not an image", encoding="utf-8")
            code = predict_dataset.main([str(root), "--quiet"])
            self.assertEqual(code, 2)
            rows = {r["pair_id"]: r for r in read_rows(root / "predictions.csv")}
            self.assertNotEqual(rows["good"]["x"], "")
            self.assertEqual(rows["bad"]["x"], "")

    def test_no_arguments_is_an_error_not_a_crash(self):
        self.assertEqual(predict_dataset.main([]), 1)

    def test_arbitration_can_be_switched_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_pair(root, "p0")
            predict_dataset.main([str(root), "--quiet", "--no-arbitration"])
            row = read_rows(root / "predictions.csv")[0]
            self.assertEqual(row["preprocessed"], "0")


class TestConventionGuards(unittest.TestCase):
    """A wrong zoom ratio or a mispaired file must be loud, not confidently wrong."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.search = (rng.random((200, 200)) * 200).astype(np.uint8)

    def test_a_wrong_zoom_ratio_names_the_option(self):
        rng = np.random.default_rng(1)
        reference = (rng.random((3000, 3000)) * 200).astype(np.uint8)
        with self.assertRaises(ValueError) as ctx:
            locate(reference, self.search)
        message = str(ctx.exception)
        self.assertIn("zoom_ratio", message)
        self.assertIn("--zoom-ratio", message)

    def test_the_error_reports_both_actual_sizes(self):
        rng = np.random.default_rng(1)
        reference = (rng.random((3000, 3000)) * 200).astype(np.uint8)
        with self.assertRaises(ValueError) as ctx:
            locate(reference, self.search)
        self.assertIn("3000x3000", str(ctx.exception))

    def test_unequal_image_sizes_warn(self):
        rng = np.random.default_rng(2)
        reference = (rng.random((300, 300)) * 200).astype(np.uint8)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            locate(reference, self.search)
        self.assertTrue(any(issubclass(w.category, RuntimeWarning) for w in caught))

    def test_the_warning_fires_once_not_once_per_arbitration_pass(self):
        """Arbitration runs two passes; the user should hear about it once."""
        rng = np.random.default_rng(2)
        reference = (rng.random((300, 300)) * 200).astype(np.uint8)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            locate(reference, self.search)
        self.assertEqual(len([w for w in caught if issubclass(w.category, RuntimeWarning)]), 1)

    def test_equal_sizes_are_silent(self):
        rng = np.random.default_rng(3)
        reference = (rng.random((200, 200)) * 200).astype(np.uint8)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            locate(reference, self.search, LocalisationConfig(zoom_ratio=2.0))
        self.assertEqual([w for w in caught if issubclass(w.category, RuntimeWarning)], [])


if __name__ == "__main__":
    unittest.main()

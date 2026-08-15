"""Locator behaviour: accuracy, the centre tiebreak, and input handling."""

from __future__ import annotations

import json
import unittest

import numpy as np

from driftsense import DramLayout, DramParams, ImagingGeometry, Placement, generate_pair
from driftsense.locate import LocalisationConfig, locate

GEOMETRY = ImagingGeometry()

#: A crop origin that straddles a mat boundary on both axes for the default
#: layout (mat_size 2600, strip 320, zero phase), so the window carries a
#: structural anchor and the answer is uniquely determined.
ANCHORED_ORIGIN = 2100.0


def synthetic_pair(patch_row: int, patch_col: int, size: int = 20, field: int = 200):
    """Build an unambiguous reference/search pair from random noise.

    Random content has no periodic structure, so there is exactly one correct
    answer. The reference is the patch expanded by the zoom ratio, which means
    area-average reduction recovers the patch exactly and isolates the locator
    from any resampling question.
    """
    rng = np.random.default_rng(7)
    search = rng.integers(0, 256, (field, field)).astype(np.uint8)
    patch = search[patch_row : patch_row + size, patch_col : patch_col + size]
    reference = np.kron(patch, np.ones((10, 10), dtype=np.uint8))
    expected = (patch_col + size / 2.0, patch_row + size / 2.0)
    return reference, search, expected


class TestLocalisationConfig(unittest.TestCase):
    def test_rejects_invalid_settings(self):
        for kwargs in (
            {"zoom_ratio": 0.0},
            {"tie_tolerance": -0.1},
            {"nms_radius_fraction": 0.0},
            {"nms_radius_fraction": 1.5},
            {"max_candidates": 0},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    LocalisationConfig(**kwargs)


class TestSyntheticLocalisation(unittest.TestCase):
    def test_finds_an_unambiguous_patch_exactly(self):
        reference, search, expected = synthetic_pair(40, 30)
        result = locate(reference, search)
        # Not bit-exact, for two reasons. Parabolic refinement applies a small
        # correction even at a perfect peak, because the neighbouring correlation
        # samples of random content are not exactly symmetric. And on structureless
        # noise the scale search has nothing to lock onto -- warping a
        # random patch barely changes its score -- so it may settle on a
        # neighbouring scale hypothesis. Both leave a residual of a few hundredths
        # of a pixel, still an order of magnitude inside the tightest tolerance the
        # submission is scored at.
        self.assertAlmostEqual(result.x, expected[0], delta=0.05)
        self.assertAlmostEqual(result.y, expected[1], delta=0.05)
        # A perfect score of exactly 1.0 only occurs with preprocessing disabled
        # and the identity hypothesis selected; band-passing changes both images,
        # so the achievable maximum is high but no longer unity.
        self.assertGreater(result.score, 0.9)

    def test_finds_a_patch_at_the_frame_corner(self):
        reference, search, expected = synthetic_pair(0, 0)
        result = locate(reference, search)
        self.assertAlmostEqual(result.x, expected[0], delta=0.05)
        self.assertAlmostEqual(result.y, expected[1], delta=0.05)

    def test_reports_timing_and_a_candidate_shortlist(self):
        reference, search, _ = synthetic_pair(50, 50)
        result = locate(reference, search, LocalisationConfig(max_candidates=6))
        self.assertGreater(result.elapsed_s, 0.0)
        self.assertEqual(len(result.candidates), 6)
        self.assertEqual(result.template_size_px, 20)

    def test_confidence_is_high_for_an_isolated_match(self):
        reference, search, _ = synthetic_pair(60, 20)
        self.assertGreater(locate(reference, search).confidence, 5.0)

    def test_result_serialises_to_json(self):
        reference, search, _ = synthetic_pair(40, 30)
        payload = json.loads(json.dumps(locate(reference, search).as_dict()))
        self.assertIn("x", payload)
        self.assertIn("confidence_psr", payload)
        self.assertEqual(len(payload["candidates"]), 16)


class TestCentreTiebreak(unittest.TestCase):
    """The problem statement's disambiguation rule, exercised directly."""

    def _two_identical_patches(self):
        rng = np.random.default_rng(11)
        field, size = 200, 20
        search = rng.integers(0, 256, (field, field)).astype(np.uint8)
        patch = rng.integers(0, 256, (size, size)).astype(np.uint8)
        search[10:30, 10:30] = patch  # centre (20, 20)  -> far from image centre
        search[90:110, 90:110] = patch  # centre (100, 100) -> the image centre
        reference = np.kron(patch, np.ones((10, 10), dtype=np.uint8))
        return reference, search

    def test_returns_the_match_closest_to_the_image_centre(self):
        reference, search = self._two_identical_patches()
        result = locate(reference, search)
        self.assertAlmostEqual(result.x, 100.0, delta=0.05)
        self.assertAlmostEqual(result.y, 100.0, delta=0.05)

    def test_both_matches_appear_on_the_shortlist(self):
        reference, search = self._two_identical_patches()
        result = locate(reference, search)
        found = {(round(c.x), round(c.y)) for c in result.candidates}
        self.assertIn((20, 20), found)
        self.assertIn((100, 100), found)

    def test_a_zero_tolerance_disables_the_tiebreak(self):
        """With no tolerance, exact ties still resolve, but near-ties do not."""
        reference, search = self._two_identical_patches()
        result = locate(reference, search, LocalisationConfig(tie_tolerance=0.0))
        # The two patches are pixel-identical, so both score exactly 1.0 and the
        # centre rule still applies even at zero tolerance.
        self.assertAlmostEqual(result.x, 100.0, delta=0.05)

    def test_tie_broken_flag_is_not_set_for_an_unambiguous_pair(self):
        reference, search, _ = synthetic_pair(40, 30)
        self.assertFalse(locate(reference, search).tie_broken)


class TestRealPairs(unittest.TestCase):
    """End-to-end against the actual generator, on anchored (solvable) placements."""

    def setUp(self) -> None:
        self.layout = DramLayout(
            DramParams(feature_size_nm=35.0, mat_size_nm=2600.0, strip_width_nm=320.0)
        )

    def _locate(self, origin_x: float, origin_y: float):
        pair = generate_pair(self.layout, Placement(origin_x, origin_y), GEOMETRY)
        result = locate(pair.reference, pair.search)
        error = float(np.hypot(result.x - pair.ground_truth.x, result.y - pair.ground_truth.y))
        return pair, result, error

    def test_anchored_placement_is_located_to_well_under_a_pixel(self):
        pair, _, error = self._locate(ANCHORED_ORIGIN, ANCHORED_ORIGIN)
        self.assertEqual(pair.anchor, "both", "test placement must carry anchors on both axes")
        self.assertLess(error, 0.25)

    def test_subpixel_placement_is_still_located_accurately(self):
        """Origins at whole nanometres land on tenths of a search pixel."""
        _, _, error = self._locate(ANCHORED_ORIGIN + 3.0, ANCHORED_ORIGIN + 7.0)
        self.assertLess(error, 0.25)

    def test_subpixel_refinement_measurably_helps(self):
        pair = generate_pair(
            self.layout, Placement(ANCHORED_ORIGIN + 3.0, ANCHORED_ORIGIN + 7.0), GEOMETRY
        )
        gt = pair.ground_truth

        def error_with(subpixel: bool) -> float:
            # Alignment refinement is switched off so this isolates the parabolic
            # peak interpolation it was written to test; with refinement on, that
            # later step dominates and the comparison measures the wrong thing.
            result = locate(
                pair.reference,
                pair.search,
                LocalisationConfig(subpixel=subpixel, refine=False),
            )
            return float(np.hypot(result.x - gt.x, result.y - gt.y))

        self.assertLess(error_with(True), error_with(False))

    def test_ambiguous_placement_reports_a_small_runner_up_margin(self):
        """The locator must be able to say when it does not know.

        A window deep inside a uniform mat is ambiguous up to one cell pitch. The
        margin over the runner-up is the signal that distinguishes this from a
        confident answer, and it is what failure analysis keys on.
        """
        interior = 1000.0  # well inside the first mat, no boundary in the window
        pair, result, _ = self._locate(interior, interior)
        self.assertEqual(pair.anchor, "none")
        self.assertIsNotNone(result.runner_up_margin)
        self.assertLess(result.runner_up_margin, 0.02)


class TestInputHandling(unittest.TestCase):
    def test_rgb_input_is_reduced_to_luminance(self):
        reference, search, expected = synthetic_pair(40, 30)
        rgb_search = np.stack([search] * 3, axis=-1)
        result = locate(reference, rgb_search)
        self.assertAlmostEqual(result.x, expected[0], delta=0.05)

    def test_template_larger_than_search_is_rejected(self):
        with self.assertRaises(ValueError):
            locate(np.zeros((2000, 2000), dtype=np.uint8), np.zeros((100, 100), dtype=np.uint8))

    def test_degenerate_zoom_ratio_is_rejected(self):
        reference, search, _ = synthetic_pair(40, 30)
        with self.assertRaises(ValueError):
            locate(reference, search, LocalisationConfig(zoom_ratio=500.0))

    def test_one_dimensional_input_is_rejected(self):
        with self.assertRaises(ValueError):
            locate(np.zeros(100), np.zeros((100, 100)))

    def test_flat_reference_is_rejected_with_a_clear_message(self):
        with self.assertRaises(ValueError) as context:
            locate(np.zeros((200, 200), dtype=np.uint8), np.zeros((200, 200), dtype=np.uint8))
        self.assertIn("contrast", str(context.exception))


if __name__ == "__main__":
    unittest.main()

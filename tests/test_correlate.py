"""Correlation core: exactness, invariances, peak extraction, sub-pixel fitting."""

from __future__ import annotations

import unittest

import numpy as np

from driftsense.correlate import (
    Peak,
    find_peaks,
    normalised_cross_correlation,
    peak_to_sidelobe_ratio,
    refine_peak_subpixel,
    window_sum,
)


def brute_force_ncc(search: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Direct O(n^2 m^2) implementation, used only as a test oracle."""
    rows = search.shape[0] - template.shape[0] + 1
    cols = search.shape[1] - template.shape[1] + 1
    out = np.zeros((rows, cols))
    flat_template = template.astype(np.float64).ravel()
    centred_template = flat_template - flat_template.mean()
    for u in range(rows):
        for v in range(cols):
            window = search[u : u + template.shape[0], v : v + template.shape[1]]
            centred = window.astype(np.float64).ravel()
            centred = centred - centred.mean()
            denominator = np.sqrt((centred**2).sum() * (centred_template**2).sum())
            out[u, v] = 0.0 if denominator == 0 else (centred * centred_template).sum() / denominator
    return out


class TestWindowSum(unittest.TestCase):
    def test_matches_explicit_summation(self):
        rng = np.random.default_rng(0)
        image = rng.random((9, 11))
        got = window_sum(image, 3, 4)
        self.assertEqual(got.shape, (7, 8))
        for u in range(got.shape[0]):
            for v in range(got.shape[1]):
                self.assertAlmostEqual(got[u, v], image[u : u + 3, v : v + 4].sum(), places=10)

    def test_full_size_window_is_the_total(self):
        image = np.arange(12, dtype=np.float64).reshape(3, 4)
        self.assertAlmostEqual(float(window_sum(image, 3, 4)[0, 0]), image.sum())

    def test_oversized_window_is_rejected(self):
        with self.assertRaises(ValueError):
            window_sum(np.zeros((4, 4)), 5, 2)

    def test_non_2d_input_is_rejected(self):
        with self.assertRaises(ValueError):
            window_sum(np.zeros(4), 2, 2)


class TestNormalisedCrossCorrelation(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.search = rng.random((40, 44))
        self.template = self.search[7:19, 5:17].copy()

    def test_agrees_with_brute_force(self):
        fast = normalised_cross_correlation(self.search, self.template)
        slow = brute_force_ncc(self.search, self.template)
        self.assertLess(float(np.abs(fast - slow).max()), 1e-12)

    def test_output_shape_is_the_valid_region(self):
        surface = normalised_cross_correlation(self.search, self.template)
        self.assertEqual(surface.shape, (40 - 12 + 1, 44 - 12 + 1))

    def test_exact_match_scores_one_at_the_right_place(self):
        surface = normalised_cross_correlation(self.search, self.template)
        self.assertAlmostEqual(float(surface.max()), 1.0, places=10)
        self.assertEqual(np.unravel_index(int(np.argmax(surface)), surface.shape), (7, 5))

    def test_invariant_to_brightness_and_contrast(self):
        """The two captures differ in dose, so the score must ignore gain and offset."""
        plain = normalised_cross_correlation(self.search, self.template)
        scaled = normalised_cross_correlation(self.search, self.template * 3.7 + 42.0)
        np.testing.assert_allclose(plain, scaled, atol=1e-10)

    def test_inverted_template_scores_minus_one(self):
        surface = normalised_cross_correlation(self.search, -self.template)
        self.assertAlmostEqual(float(surface.min()), -1.0, places=10)

    def test_values_stay_within_bounds(self):
        rng = np.random.default_rng(3)
        surface = normalised_cross_correlation(rng.random((30, 30)), rng.random((8, 8)))
        self.assertGreaterEqual(float(surface.min()), -1.0)
        self.assertLessEqual(float(surface.max()), 1.0)

    def test_flat_search_region_yields_zero_not_a_division_error(self):
        search = np.zeros((20, 20))
        search[10:14, 10:14] = 1.0
        surface = normalised_cross_correlation(search, np.eye(4))
        self.assertTrue(np.all(np.isfinite(surface)))
        self.assertAlmostEqual(float(surface[0, 0]), 0.0)

    def test_zero_contrast_template_is_rejected(self):
        with self.assertRaises(ValueError):
            normalised_cross_correlation(np.random.default_rng(0).random((20, 20)), np.ones((4, 4)))

    def test_oversized_template_is_rejected(self):
        with self.assertRaises(ValueError):
            normalised_cross_correlation(np.zeros((10, 10)), np.zeros((12, 4)))

    def test_single_pixel_template_is_rejected(self):
        with self.assertRaises(ValueError):
            normalised_cross_correlation(np.zeros((10, 10)), np.zeros((1, 1)))

    def test_no_circular_wraparound_at_the_far_edge(self):
        """A match at the last valid offset must be found, not corrupted by wrap."""
        rng = np.random.default_rng(5)
        search = rng.random((32, 32))
        template = search[20:32, 20:32].copy()
        surface = normalised_cross_correlation(search, template)
        self.assertEqual(np.unravel_index(int(np.argmax(surface)), surface.shape), (20, 20))
        self.assertAlmostEqual(float(surface.max()), 1.0, places=10)


class TestFindPeaks(unittest.TestCase):
    def test_returns_peaks_in_descending_score_order(self):
        surface = np.zeros((60, 60))
        surface[10, 10] = 0.9
        surface[40, 40] = 0.7
        surface[20, 45] = 0.5
        peaks = find_peaks(surface, min_distance=5, max_peaks=3)
        self.assertEqual([(p.row, p.col) for p in peaks], [(10, 10), (40, 40), (20, 45)])
        self.assertEqual([round(p.score, 3) for p in peaks], [0.9, 0.7, 0.5])

    def test_suppression_prevents_reporting_the_same_peak_twice(self):
        surface = np.zeros((40, 40))
        surface[20, 20] = 1.0
        surface[20, 21] = 0.99  # shoulder of the same peak
        surface[5, 5] = 0.5
        peaks = find_peaks(surface, min_distance=6, max_peaks=3)
        self.assertEqual((peaks[0].row, peaks[0].col), (20, 20))
        self.assertEqual((peaks[1].row, peaks[1].col), (5, 5))

    def test_respects_the_requested_count(self):
        rng = np.random.default_rng(1)
        peaks = find_peaks(rng.random((50, 50)), min_distance=3, max_peaks=7)
        self.assertEqual(len(peaks), 7)

    def test_reports_the_true_surface_value_not_the_suppressed_one(self):
        surface = np.full((20, 20), 0.25)
        surface[7, 7] = 0.8
        peaks = find_peaks(surface, min_distance=4, max_peaks=2)
        self.assertAlmostEqual(peaks[0].score, 0.8)
        self.assertAlmostEqual(peaks[1].score, 0.25)

    def test_invalid_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            find_peaks(np.zeros((5, 5)), min_distance=1, max_peaks=0)
        with self.assertRaises(ValueError):
            find_peaks(np.zeros(5), min_distance=1, max_peaks=1)


class TestSubpixelRefinement(unittest.TestCase):
    def test_recovers_the_vertex_of_a_known_parabola(self):
        # A separable parabola peaking at (row, col) = (10.3, 20.0 - 0.25)
        rows, cols = np.mgrid[0:21, 0:41]
        surface = -((rows - 10.3) ** 2) - ((cols - 19.75) ** 2)
        refined = refine_peak_subpixel(surface, Peak(score=0.0, row=10, col=20))
        self.assertAlmostEqual(refined.refined_row, 10.3, places=6)
        self.assertAlmostEqual(refined.refined_col, 19.75, places=6)

    def test_symmetric_peak_needs_no_correction(self):
        surface = np.array([[0.0, 0.0, 0.0], [0.5, 1.0, 0.5], [0.0, 0.0, 0.0]])
        refined = refine_peak_subpixel(surface, Peak(score=1.0, row=1, col=1))
        self.assertAlmostEqual(refined.col_offset, 0.0)

    def test_offset_is_clamped_to_half_a_sample(self):
        surface = np.array([[0.0, 0.0, 0.0], [0.9999, 1.0, 0.0], [0.0, 0.0, 0.0]])
        refined = refine_peak_subpixel(surface, Peak(score=1.0, row=1, col=1))
        self.assertLessEqual(abs(refined.col_offset), 0.5)

    def test_boundary_peaks_are_returned_unrefined(self):
        surface = np.random.default_rng(0).random((9, 9))
        refined = refine_peak_subpixel(surface, Peak(score=1.0, row=0, col=8))
        self.assertEqual((refined.row_offset, refined.col_offset), (0.0, 0.0))

    def test_flat_neighbourhood_gives_no_correction(self):
        surface = np.ones((5, 5))
        refined = refine_peak_subpixel(surface, Peak(score=1.0, row=2, col=2))
        self.assertEqual((refined.row_offset, refined.col_offset), (0.0, 0.0))


class TestPeakToSidelobeRatio(unittest.TestCase):
    def test_isolated_peak_scores_far_higher_than_a_repeated_one(self):
        rng = np.random.default_rng(0)
        isolated = rng.normal(0.0, 0.01, (80, 80))
        isolated[40, 40] = 1.0

        repeated = rng.normal(0.0, 0.01, (80, 80))
        for row in (20, 40, 60):
            for col in (20, 40, 60):
                repeated[row, col] = 1.0

        peak = Peak(score=1.0, row=40, col=40)
        self.assertGreater(
            peak_to_sidelobe_ratio(isolated, peak, 5),
            peak_to_sidelobe_ratio(repeated, peak, 5),
        )

    def test_flat_sidelobe_reports_infinity(self):
        surface = np.zeros((20, 20))
        surface[10, 10] = 1.0
        self.assertEqual(peak_to_sidelobe_ratio(surface, Peak(1.0, 10, 10), 2), float("inf"))

    def test_fully_excluded_surface_reports_zero(self):
        surface = np.zeros((5, 5))
        self.assertEqual(peak_to_sidelobe_ratio(surface, Peak(0.0, 2, 2), 50), 0.0)


if __name__ == "__main__":
    unittest.main()

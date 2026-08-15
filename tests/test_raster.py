"""Rasterisation and resampling tests: coverage must be exact and scale-invariant."""

from __future__ import annotations

import unittest

import numpy as np

from driftsense.raster import (
    cumulative_stripe_length,
    outer_coverage,
    periodic_stripe_coverage,
)
from driftsense.resample import (
    area_average_reduce,
    area_average_reduce_1d,
    bilinear_sample,
    sample_patch,
)


def reference_coverage(
    n_px, pixel_size, origin, period, width, phase=0.0, lo=None, hi=None, samples=20_000
):
    """Monte-Carlo-free oracle: dense uniform sampling of the indicator function.

    Deliberately slow and obviously correct, used only to check the closed form.
    """
    out = np.zeros(n_px)
    for k in range(n_px):
        a = origin + k * pixel_size
        u = a + (np.arange(samples) + 0.5) * (pixel_size / samples)
        inside = np.mod(u - phase, period) < width
        if lo is not None:
            inside &= u >= lo
        if hi is not None:
            inside &= u < hi
        out[k] = inside.mean()
    return out


class TestCumulativeStripeLength(unittest.TestCase):
    def test_accumulates_one_width_per_period(self):
        got = cumulative_stripe_length(np.array([0.0, 10.0, 20.0, 30.0]), 10.0, 4.0)
        np.testing.assert_allclose(got, [0.0, 4.0, 8.0, 12.0])

    def test_partial_period_contributes_the_overlap(self):
        got = cumulative_stripe_length(np.array([2.0, 4.0, 6.0]), 10.0, 4.0)
        np.testing.assert_allclose(got, [2.0, 4.0, 4.0])

    def test_phase_shifts_the_accumulation(self):
        unshifted = cumulative_stripe_length(np.array([5.0]), 10.0, 4.0)
        shifted = cumulative_stripe_length(np.array([7.0]), 10.0, 4.0, phase_nm=2.0)
        np.testing.assert_allclose(unshifted, shifted)

    def test_negative_positions_are_handled(self):
        got = cumulative_stripe_length(np.array([-10.0, -20.0]), 10.0, 4.0)
        np.testing.assert_allclose(got, [-4.0, -8.0])

    def test_zero_width_accumulates_nothing(self):
        np.testing.assert_allclose(cumulative_stripe_length(np.array([1e6]), 10.0, 0.0), 0.0)

    def test_width_at_the_period_is_solid(self):
        got = cumulative_stripe_length(np.array([37.0]), 10.0, 10.0)
        np.testing.assert_allclose(got, 37.0)

    def test_zero_period_is_rejected(self):
        with self.assertRaises(ValueError):
            cumulative_stripe_length(np.array([0.0]), 0.0, 1.0)


class TestPeriodicStripeCoverage(unittest.TestCase):
    def test_fully_covered_pixels_read_one_and_gaps_read_zero(self):
        # Period 10 nm, 5 nm wide, 1 nm pixels: five solid then five empty.
        result = periodic_stripe_coverage(20, 1.0, 0.0, 10.0, 5.0)
        np.testing.assert_allclose(result[:5], 1.0)
        np.testing.assert_allclose(result[5:10], 0.0)
        np.testing.assert_allclose(result[10:15], 1.0)

    def test_pixel_spanning_a_whole_period_reads_the_duty_cycle(self):
        np.testing.assert_allclose(periodic_stripe_coverage(8, 2.0, 0.0, 2.0, 1.0), 0.5)

    def test_partially_covered_pixel_is_exact(self):
        """The closed form is exact, so no tolerance is needed here."""
        result = periodic_stripe_coverage(1, 4.0, 0.0, 1000.0, 1.0)
        self.assertAlmostEqual(float(result[0]), 0.25, places=6)

    def test_matches_a_dense_sampling_oracle(self):
        for period, width, phase in ((70.0, 35.0, 0.0), (13.3, 4.7, 2.1), (9.0, 8.9, 5.5)):
            with self.subTest(period=period, width=width):
                got = periodic_stripe_coverage(40, 3.0, 1.7, period, width, phase)
                want = reference_coverage(40, 3.0, 1.7, period, width, phase)
                np.testing.assert_allclose(got, want, atol=1e-3)

    def test_degenerate_widths_are_handled_exactly(self):
        np.testing.assert_allclose(periodic_stripe_coverage(5, 1.0, 0.0, 4.0, 0.0), 0.0)
        np.testing.assert_allclose(periodic_stripe_coverage(5, 1.0, 0.0, 4.0, 4.0), 1.0)
        np.testing.assert_allclose(periodic_stripe_coverage(5, 1.0, 0.0, 4.0, -3.0), 0.0)
        np.testing.assert_allclose(periodic_stripe_coverage(5, 1.0, 0.0, 4.0, 9.0), 1.0)

    def test_clipping_confines_the_pattern(self):
        result = periodic_stripe_coverage(10, 1.0, 0.0, 2.0, 2.0, lo_nm=3.0, hi_nm=7.0)
        np.testing.assert_allclose(result[:3], 0.0)
        np.testing.assert_allclose(result[3:7], 1.0)
        np.testing.assert_allclose(result[7:], 0.0)

    def test_clipping_handles_a_partial_pixel(self):
        result = periodic_stripe_coverage(4, 2.0, 0.0, 100.0, 100.0, lo_nm=1.0, hi_nm=5.0)
        np.testing.assert_allclose(result, [0.5, 1.0, 0.5, 0.0])

    def test_inverted_clip_interval_yields_nothing(self):
        np.testing.assert_allclose(
            periodic_stripe_coverage(4, 1.0, 0.0, 2.0, 1.0, lo_nm=5.0, hi_nm=1.0), 0.0
        )

    def test_coverage_is_conserved_across_pixel_sizes(self):
        """Total covered length must not depend on the sampling resolution.

        This is the property that makes the 1 nm reference render and the 10 nm
        search render describe the same physical structure.
        """
        fine = periodic_stripe_coverage(1000, 1.0, 0.0, 70.0, 35.0)
        coarse = periodic_stripe_coverage(100, 10.0, 0.0, 70.0, 35.0)
        self.assertAlmostEqual(float(fine.sum()) * 1.0, float(coarse.sum()) * 10.0, places=3)

    def test_block_reducing_fine_coverage_reproduces_coarse_coverage_exactly(self):
        fine = periodic_stripe_coverage(1000, 1.0, 0.0, 70.0, 35.0)
        coarse = periodic_stripe_coverage(100, 10.0, 0.0, 70.0, 35.0)
        reduced = fine.reshape(100, 10).mean(axis=1)
        self.assertLess(float(np.max(np.abs(reduced - coarse))), 1e-6)

    def test_result_is_float32_and_bounded(self):
        result = periodic_stripe_coverage(50, 1.3, -7.0, 5.0, 2.0, 0.4)
        self.assertEqual(result.dtype, np.float32)
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)

    def test_invalid_grids_are_rejected(self):
        with self.assertRaises(ValueError):
            periodic_stripe_coverage(0, 1.0, 0.0, 2.0, 1.0)
        with self.assertRaises(ValueError):
            periodic_stripe_coverage(4, 0.0, 0.0, 2.0, 1.0)


class TestOuterCoverage(unittest.TestCase):
    def test_product_set_area_factorises(self):
        field = outer_coverage(
            np.array([1.0, 0.5, 0.0], dtype=np.float32), np.array([0.25, 1.0], dtype=np.float32)
        )
        self.assertEqual(field.shape, (3, 2))
        self.assertAlmostEqual(float(field[0, 0]), 0.25, places=6)
        self.assertAlmostEqual(float(field[1, 1]), 0.5, places=6)
        self.assertAlmostEqual(float(field[2, 0]), 0.0, places=6)

    def test_result_is_float32_for_memory(self):
        self.assertEqual(outer_coverage(np.ones(4), np.ones(4)).dtype, np.float32)


class TestAreaAverageReduce(unittest.TestCase):
    def test_integer_factor_is_a_plain_block_mean(self):
        image = np.arange(16, dtype=np.float64).reshape(4, 4)
        np.testing.assert_allclose(area_average_reduce(image, 2.0), [[2.5, 4.5], [10.5, 12.5]])

    def test_general_path_agrees_with_the_integer_fast_path(self):
        rng = np.random.default_rng(0)
        image = rng.random((60, 60))
        fast = area_average_reduce(image, 5.0)
        general = area_average_reduce_1d(area_average_reduce_1d(image, 5.0, axis=0), 5.0, axis=1)
        np.testing.assert_allclose(fast, general, atol=1e-12)

    def test_fractional_factor_preserves_the_total(self):
        reduced = area_average_reduce_1d(np.ones(100), 7.5)
        self.assertEqual(reduced.shape[0], 13)
        np.testing.assert_allclose(reduced, 1.0, atol=1e-12)

    def test_reduction_of_a_ramp_is_the_local_mean(self):
        reduced = area_average_reduce_1d(np.arange(10, dtype=np.float64), 2.0)
        np.testing.assert_allclose(reduced, [0.5, 2.5, 4.5, 6.5, 8.5])

    def test_invalid_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            area_average_reduce(np.ones((4, 4)), 0.0)
        with self.assertRaises(ValueError):
            area_average_reduce(np.ones(4), 2.0)
        with self.assertRaises(ValueError):
            area_average_reduce_1d(np.ones(4), 100.0)


class TestBilinearSample(unittest.TestCase):
    def test_integer_indices_return_exact_pixels(self):
        image = np.arange(25, dtype=np.float64).reshape(5, 5)
        got = bilinear_sample(image, np.array([0.0, 4.0, 2.0]), np.array([0.0, 4.0, 3.0]))
        np.testing.assert_allclose(got, [0.0, 24.0, 17.0])

    def test_half_pixel_is_the_average_of_neighbours(self):
        image = np.array([[0.0, 10.0], [20.0, 30.0]])
        self.assertAlmostEqual(float(bilinear_sample(image, np.array([0.5]), np.array([0.5]))[0]), 15.0)

    def test_out_of_range_samples_clamp_to_the_border(self):
        image = np.array([[1.0, 2.0], [3.0, 4.0]])
        got = bilinear_sample(image, np.array([-5.0, 9.0]), np.array([-5.0, 9.0]))
        np.testing.assert_allclose(got, [1.0, 4.0])

    def test_rejects_non_2d_input(self):
        with self.assertRaises(ValueError):
            bilinear_sample(np.ones(4), np.array([0.0]), np.array([0.0]))


class TestSamplePatch(unittest.TestCase):
    def test_aligned_patch_is_an_exact_crop(self):
        image = np.arange(100, dtype=np.float64).reshape(10, 10)
        np.testing.assert_allclose(sample_patch(image, 2.0, 3.0, 4), image[3:7, 2:6])

    def test_patch_size_is_respected(self):
        self.assertEqual(sample_patch(np.zeros((20, 20)), 1.5, 1.5, 7).shape, (7, 7))

    def test_rejects_non_positive_size(self):
        with self.assertRaises(ValueError):
            sample_patch(np.zeros((4, 4)), 0.0, 0.0, 0)


if __name__ == "__main__":
    unittest.main()

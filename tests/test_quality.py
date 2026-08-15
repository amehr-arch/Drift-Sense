"""Spectral quality estimate, and the preprocessing arbitration built on top of it.

The estimator itself is sound and is tested as such. The gate that was built on
it is *not* shipped, because it was refuted -- see ``TestWhyTheGateWasAbandoned``
at the end, which pins the reasoning in place so nobody re-derives the same dead
end from the fact that ``quality.py`` exists.
"""

from __future__ import annotations

import unittest

import numpy as np

from driftsense.imaging import gaussian_blur as _gaussian_blur
from driftsense.locate import LocalisationConfig, locate
from driftsense.preprocess import PreprocessConfig
from driftsense.quality import (
    QualityConfig,
    QualityEstimate,
    estimate_quality,
    radial_power_spectrum,
)


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Isotropic convenience wrapper; the imaging model takes a sigma per axis."""
    return _gaussian_blur(image, sigma, sigma)


def striped(size: int = 256, period: float = 8.0, amplitude: float = 60.0) -> np.ndarray:
    """A periodic layout stand-in: vertical stripes at a known pitch."""
    x = np.arange(size, dtype=np.float64)
    row = 128.0 + amplitude * np.sin(2.0 * np.pi * x / period)
    return np.tile(row, (size, 1))


def noisy(image: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return image + rng.normal(0.0, sigma, image.shape)


class TestRadialPowerSpectrum(unittest.TestCase):
    def test_returns_matching_lengths(self):
        freqs, power = radial_power_spectrum(striped(), n_bins=32)
        self.assertEqual(len(freqs), len(power))

    def test_frequencies_are_increasing_and_bounded(self):
        freqs, _ = radial_power_spectrum(striped(), n_bins=32)
        self.assertTrue(np.all(np.diff(freqs) > 0))
        self.assertGreater(freqs[0], 0.0)
        self.assertLessEqual(freqs[-1], 1.0)

    def test_dc_is_excluded(self):
        """A constant image has all its power at DC, which must not be reported."""
        _, power = radial_power_spectrum(np.full((128, 128), 200.0), n_bins=32)
        self.assertLess(float(power.max()), 1e-6)

    def test_a_brightness_offset_changes_nothing(self):
        a = radial_power_spectrum(striped(), n_bins=32)[1]
        b = radial_power_spectrum(striped() + 50.0, n_bins=32)[1]
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)

    def test_power_peaks_near_the_stripe_frequency(self):
        """Period 8 px means 2/8 = 0.25 of Nyquist."""
        freqs, power = radial_power_spectrum(striped(period=8.0), n_bins=64, window=True)
        self.assertAlmostEqual(float(freqs[int(np.argmax(power))]), 0.25, delta=0.05)

    def test_a_finer_stripe_peaks_higher(self):
        def peak(period):
            f, p = radial_power_spectrum(striped(period=period), n_bins=64)
            return f[int(np.argmax(p))]

        self.assertGreater(peak(4.0), peak(16.0))

    def test_rejects_a_non_2d_input(self):
        with self.assertRaises(ValueError):
            radial_power_spectrum(np.zeros((4, 4, 3)))

    def test_rejects_a_tiny_image(self):
        with self.assertRaises(ValueError):
            radial_power_spectrum(np.zeros((3, 3)))

    def test_handles_a_non_square_image(self):
        freqs, power = radial_power_spectrum(striped(64)[:32, :], n_bins=16)
        self.assertTrue(np.all(np.isfinite(power)))
        self.assertEqual(len(freqs), 15)


class TestEstimateQuality(unittest.TestCase):
    def test_a_sharp_image_scores_higher_than_a_blurred_one(self):
        sharp = noisy(striped(period=6.0), 3.0)
        blurred = noisy(gaussian_blur(striped(period=6.0), 4.0), 3.0)
        self.assertGreater(
            estimate_quality(sharp).cutoff_fraction,
            estimate_quality(blurred).cutoff_fraction,
        )

    def test_the_estimate_falls_monotonically_with_blur(self):
        """The property the whole idea rests on. It does hold."""
        base = striped(period=6.0)
        cutoffs = [
            estimate_quality(noisy(gaussian_blur(base, s), 3.0)).cutoff_fraction
            for s in (0.5, 1.5, 3.0, 6.0)
        ]
        for earlier, later in zip(cutoffs, cutoffs[1:]):
            self.assertGreaterEqual(earlier, later)

    def test_more_noise_lowers_the_estimate(self):
        base = striped(period=6.0)
        self.assertGreaterEqual(
            estimate_quality(noisy(base, 2.0)).cutoff_fraction,
            estimate_quality(noisy(base, 40.0)).cutoff_fraction,
        )

    def test_pure_noise_scores_low(self):
        rng = np.random.default_rng(3)
        estimate = estimate_quality(rng.normal(128.0, 20.0, (256, 256)))
        self.assertLess(estimate.cutoff_fraction, 0.6)

    def test_a_flat_image_reports_no_structure(self):
        estimate = estimate_quality(np.full((128, 128), 100.0))
        self.assertEqual(estimate.cutoff_fraction, 0.0)
        self.assertEqual(estimate.n_bins_above, 0)

    def test_the_result_is_finite_and_bounded(self):
        estimate = estimate_quality(noisy(striped(), 5.0))
        self.assertTrue(0.0 <= estimate.cutoff_fraction <= 1.0)
        self.assertTrue(np.isfinite(estimate.peak_over_floor))

    def test_it_returns_a_quality_estimate(self):
        self.assertIsInstance(estimate_quality(noisy(striped(), 5.0)), QualityEstimate)

    def test_as_dict_is_json_ready(self):
        payload = estimate_quality(noisy(striped(), 5.0)).as_dict()
        self.assertEqual(
            set(payload), {"cutoff_fraction", "noise_floor", "peak_over_floor", "n_bins_above"}
        )

    def test_uint8_input_is_accepted(self):
        image = np.clip(noisy(striped(), 5.0), 0, 255).astype(np.uint8)
        self.assertTrue(np.isfinite(estimate_quality(image).cutoff_fraction))

    def test_the_window_matters(self):
        """Without windowing the edge discontinuity inflates every bin.

        This is why ``window`` defaults to true: unwindowed, a blurred image
        reports a broadband tail it does not have.
        """
        blurred = noisy(gaussian_blur(striped(period=6.0), 6.0), 3.0)
        windowed = estimate_quality(blurred, QualityConfig(window=True))
        raw = estimate_quality(blurred, QualityConfig(window=False))
        self.assertLessEqual(windowed.cutoff_fraction, raw.cutoff_fraction)


class TestQualityConfig(unittest.TestCase):
    def test_rejects_too_few_bins(self):
        with self.assertRaises(ValueError):
            QualityConfig(n_bins=4)

    def test_rejects_a_noise_band_outside_the_unit_interval(self):
        with self.assertRaises(ValueError):
            QualityConfig(noise_band_start=1.5)

    def test_rejects_a_detection_factor_of_one(self):
        """At a factor of 1 the noise floor detects itself and every bin passes."""
        with self.assertRaises(ValueError):
            QualityConfig(detection_factor=1.0)

    def test_a_higher_detection_factor_is_stricter(self):
        image = noisy(striped(period=6.0), 8.0)
        lenient = estimate_quality(image, QualityConfig(detection_factor=1.5))
        strict = estimate_quality(image, QualityConfig(detection_factor=20.0))
        self.assertGreaterEqual(lenient.n_bins_above, strict.n_bins_above)


class TestArbitration(unittest.TestCase):
    """The rule that actually shipped."""

    def setUp(self):
        rng = np.random.default_rng(11)
        self.search = np.clip(
            noisy(striped(size=300, period=9.0), 6.0) + rng.normal(0, 2, (300, 300)), 0, 255
        ).astype(np.uint8)
        patch = self.search[110:150, 130:170]
        self.reference = np.kron(patch, np.ones((10, 10))).astype(np.uint8)

    def test_arbitration_is_on_by_default(self):
        self.assertTrue(LocalisationConfig().arbitrate_preprocessing)

    def test_it_reports_which_pass_won(self):
        result = locate(self.reference, self.search)
        self.assertIsInstance(result.preprocessed, bool)

    def test_it_reports_both_margins(self):
        result = locate(self.reference, self.search)
        self.assertIsNotNone(result.arbitration_margin)
        self.assertEqual(len(result.arbitration_margin), 2)

    def test_the_winning_margin_is_the_larger_one(self):
        won, lost = locate(self.reference, self.search).arbitration_margin
        self.assertGreaterEqual(won, lost)

    def test_disabling_it_runs_a_single_pass(self):
        result = locate(
            self.reference, self.search,
            LocalisationConfig(arbitrate_preprocessing=False),
        )
        self.assertIsNone(result.arbitration_margin)

    def test_a_single_pass_with_no_config_does_not_preprocess(self):
        result = locate(
            self.reference, self.search,
            LocalisationConfig(arbitrate_preprocessing=False),
        )
        self.assertFalse(result.preprocessed)

    def test_a_single_pass_with_a_config_does_preprocess(self):
        result = locate(
            self.reference, self.search,
            LocalisationConfig(arbitrate_preprocessing=False, preprocess=PreprocessConfig()),
        )
        self.assertTrue(result.preprocessed)

    def test_the_reported_time_covers_both_passes(self):
        both = locate(self.reference, self.search).elapsed_s
        one = locate(
            self.reference, self.search,
            LocalisationConfig(arbitrate_preprocessing=False),
        ).elapsed_s
        self.assertGreater(both, one)

    def test_the_answer_stays_inside_the_image(self):
        result = locate(self.reference, self.search)
        self.assertTrue(0 <= result.x <= self.search.shape[1])
        self.assertTrue(0 <= result.y <= self.search.shape[0])

    def test_as_dict_exposes_the_arbitration(self):
        payload = locate(self.reference, self.search).as_dict()
        self.assertIn("preprocessed", payload)
        self.assertIn("arbitration_margin", payload)

    def test_it_is_deterministic(self):
        a = locate(self.reference, self.search)
        b = locate(self.reference, self.search)
        self.assertAlmostEqual(a.x, b.x, places=9)
        self.assertEqual(a.preprocessed, b.preprocessed)


class TestSmallInputs(unittest.TestCase):
    """A small reference used to crash inside the coarse hypothesis pass.

    A 30 px reference reduces to a 3 px template at a zoom ratio of 10, and the
    coarse pass then tried to reduce that by another factor of 4. The guard meant
    to catch it sat *below* the call that raised, so it could never fire. The
    error surfaced as "factor 4.0 is too large for length 3", which tells a user
    nothing about what they did wrong.
    """

    def setUp(self):
        rng = np.random.default_rng(0)
        self.search = (rng.random((200, 200)) * 200).astype(np.uint8)
        self.tiny = (rng.random((30, 30)) * 200).astype(np.uint8)

    def test_a_small_reference_does_not_crash(self):
        result = locate(self.tiny, self.search)
        self.assertTrue(np.isfinite(result.x) and np.isfinite(result.y))

    def test_a_small_reference_without_arbitration(self):
        result = locate(
            self.tiny, self.search, LocalisationConfig(arbitrate_preprocessing=False)
        )
        self.assertTrue(np.isfinite(result.x))

    def test_arbitration_and_single_pass_agree_on_a_small_reference(self):
        a = locate(self.tiny, self.search)
        b = locate(self.tiny, self.search, LocalisationConfig(arbitrate_preprocessing=False))
        self.assertAlmostEqual(a.x, b.x, places=6)

    def test_a_flat_search_image_does_not_crash(self):
        result = locate(self.tiny, np.full((200, 200), 128, np.uint8))
        self.assertTrue(np.isfinite(result.x))

    def test_float_input_matches_uint8_input(self):
        a = locate(self.tiny, self.search)
        b = locate(self.tiny.astype(np.float64), self.search.astype(np.float64))
        self.assertAlmostEqual(a.x, b.x, places=6)

    def test_a_template_the_size_of_the_search_image_still_works(self):
        rng = np.random.default_rng(1)
        big = (rng.random((2000, 2000)) * 200).astype(np.uint8)
        self.assertTrue(np.isfinite(locate(big, self.search).x))

    def test_a_template_larger_than_the_search_image_is_rejected_clearly(self):
        rng = np.random.default_rng(2)
        bigger = (rng.random((3000, 3000)) * 200).astype(np.uint8)
        with self.assertRaises(ValueError) as ctx:
            locate(bigger, self.search)
        self.assertIn("does not fit inside", str(ctx.exception))

    def test_a_flat_reference_is_rejected_clearly(self):
        with self.assertRaises(ValueError) as ctx:
            locate(np.full((300, 300), 100, np.uint8), self.search)
        self.assertIn("contrast", str(ctx.exception))

    def test_rgb_input_is_reduced_to_luminance(self):
        rng = np.random.default_rng(3)
        rgb = (rng.random((300, 300, 3)) * 200).astype(np.uint8)
        self.assertTrue(np.isfinite(locate(rgb, self.search).x))


class TestWhyTheGateWasAbandoned(unittest.TestCase):
    """Pinning down a refuted hypothesis so it does not get rebuilt.

    ``quality.py`` exists and works, which makes it inviting to wire it back in
    as a preprocessing gate. It was tried. On the held-out set the pair that
    preprocessing damaged worst carried the *highest* quality score of all
    thirteen, so no threshold separates the safe cases from the unsafe ones.
    """

    def test_the_estimator_is_kept_as_a_diagnostic_not_a_gate(self):
        config = LocalisationConfig()
        self.assertFalse(hasattr(config, "quality_gate"))
        self.assertFalse(hasattr(config, "min_quality"))

    def test_quality_does_not_import_the_locator(self):
        """It runs at inference time, so it must not create a cycle."""
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1] / "driftsense" / "quality.py"
        ).read_text(encoding="utf-8")
        for forbidden in ("from .locate", "from .preprocess", "from .layouts"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()

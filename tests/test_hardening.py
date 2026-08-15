"""an earlier revision hardening: preprocessing, hypothesis search, uniqueness weighting."""

from __future__ import annotations

import unittest

import numpy as np

from driftsense import DramLayout, DramParams, ImagingGeometry, Placement, generate_pair
from driftsense.correlate import (
    normalised_cross_correlation,
    weighted_normalised_cross_correlation,
)
from driftsense.imaging import CaptureParams
from driftsense.locate import (
    Hypothesis,
    LocalisationConfig,
    locate,
    refine_alignment,
    uniqueness_weights,
    warp_template,
)
from driftsense.preprocess import (
    PreprocessConfig,
    bandpass,
    box_blur,
    local_contrast_normalise,
    preprocess,
)

GEOMETRY = ImagingGeometry()
ANCHORED = 2100.0


def make_pair(rotation_deg: float = 0.0, seed: int = 1):
    layout = DramLayout(
        DramParams(feature_size_nm=35.0, mat_size_nm=2600.0, strip_width_nm=320.0)
    )
    return generate_pair(
        layout,
        Placement(ANCHORED, ANCHORED),
        GEOMETRY,
        reference_capture=CaptureParams(rotation_deg=rotation_deg),
        search_capture=CaptureParams(dose=200.0, detector_noise_sigma=5.0, spot_size_nm=12.0),
        seed=seed,
    )


class TestBoxBlur(unittest.TestCase):
    def test_constant_image_is_unchanged(self):
        np.testing.assert_allclose(box_blur(np.full((32, 32), 3.0), 4), 3.0)

    def test_averages_over_the_window(self):
        image = np.zeros((11, 11))
        image[5, 5] = 121.0
        blurred = box_blur(image, 5)
        self.assertAlmostEqual(float(blurred[5, 5]), 1.0, places=6)

    def test_preserves_the_mean_of_a_smooth_field(self):
        rng = np.random.default_rng(0)
        image = rng.random((64, 64))
        self.assertAlmostEqual(float(box_blur(image, 3).mean()), float(image.mean()), delta=0.02)

    def test_matches_an_explicit_window_mean_in_the_interior(self):
        rng = np.random.default_rng(1)
        image = rng.random((20, 20))
        blurred = box_blur(image, 2)
        self.assertAlmostEqual(float(blurred[10, 10]), float(image[8:13, 8:13].mean()), places=9)

    def test_oversized_radius_returns_a_nearly_flat_field(self):
        """With a window larger than the image, everything averages together."""
        image = np.random.default_rng(0).random((8, 8))
        blurred = box_blur(image, 50)
        self.assertTrue(np.all(np.isfinite(blurred)))
        self.assertLess(float(blurred.std()), float(image.std()) * 0.05)


class TestBandpass(unittest.TestCase):
    def test_highpass_removes_a_constant_offset(self):
        rng = np.random.default_rng(0)
        image = rng.random((64, 64))
        plain = bandpass(image, 8.0, 0.0)
        offset = bandpass(image + 100.0, 8.0, 0.0)
        np.testing.assert_allclose(plain, offset, atol=1e-9)

    def test_highpass_removes_a_smooth_gradient(self):
        """Exactly the vignetting and charging case preprocessing exists for."""
        rows, cols = np.mgrid[0:64, 0:64]
        gradient = rows * 0.5 + cols * 0.3
        filtered = bandpass(gradient, 6.0, 0.0)
        self.assertLess(float(np.abs(filtered).max()), float(np.abs(gradient).max()) * 0.2)

    def test_lowpass_attenuates_fine_detail(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(0.0, 1.0, (64, 64))
        self.assertLess(bandpass(noise, 0.0, 2.0).std(), noise.std())

    def test_disabled_filters_are_a_no_op(self):
        image = np.random.default_rng(0).random((16, 16))
        np.testing.assert_allclose(bandpass(image, 0.0, 0.0), image)


class TestLocalContrastNormalise(unittest.TestCase):
    def test_equalises_two_regions_of_different_contrast(self):
        rng = np.random.default_rng(0)
        image = np.zeros((64, 128))
        image[:, :64] = rng.normal(0, 1.0, (64, 64))
        image[:, 64:] = rng.normal(0, 20.0, (64, 64))
        out = local_contrast_normalise(image, 8.0)
        faint = float(out[:, 10:54].std())
        strong = float(out[:, 74:118].std())
        self.assertAlmostEqual(faint / max(strong, 1e-9), 1.0, delta=0.4)

    def test_zero_sigma_is_a_no_op(self):
        image = np.random.default_rng(0).random((16, 16))
        np.testing.assert_allclose(local_contrast_normalise(image, 0.0), image)

    def test_flat_input_does_not_blow_up(self):
        out = local_contrast_normalise(np.full((32, 32), 5.0), 6.0)
        self.assertTrue(np.all(np.isfinite(out)))


class TestPreprocessConfig(unittest.TestCase):
    def test_rejects_an_empty_band(self):
        with self.assertRaises(ValueError):
            PreprocessConfig(highpass_sigma_px=2.0, lowpass_sigma_px=5.0)

    def test_rejects_negative_scales(self):
        with self.assertRaises(ValueError):
            PreprocessConfig(highpass_sigma_px=-1.0)

    def test_scaled_for_shrinks_with_the_template(self):
        """The defect this fixed: absolute sigmas erased a small template."""
        base = PreprocessConfig()
        small = base.scaled_for(20.0)
        self.assertLess(small.highpass_sigma_px, base.highpass_sigma_px)
        self.assertLess(small.contrast_sigma_px, base.contrast_sigma_px)

    def test_scaled_for_is_identity_at_the_reference_size(self):
        base = PreprocessConfig()
        same = base.scaled_for(base.reference_template_px)
        self.assertAlmostEqual(same.highpass_sigma_px, base.highpass_sigma_px)

    def test_scaled_for_leaves_the_lowpass_alone(self):
        """It models the beam, which is a property of the microscope."""
        base = PreprocessConfig()
        self.assertAlmostEqual(base.scaled_for(20.0).lowpass_sigma_px, base.lowpass_sigma_px)

    def test_scaled_for_keeps_a_usable_floor(self):
        tiny = PreprocessConfig().scaled_for(1.0)
        self.assertGreaterEqual(tiny.highpass_sigma_px, 2.0)

    def test_scaled_for_keeps_disabled_filters_disabled(self):
        """A zero sigma means "off", and rescaling must not switch it back on.

        Applying the minimum floor unconditionally re-enabled a deliberately
        disabled high-pass, which then failed the empty-band validation and
        crashed the locator.
        """
        config = PreprocessConfig(
            highpass_sigma_px=0.0, lowpass_sigma_px=3.0, contrast_sigma_px=0.0
        ).scaled_for(100.0)
        self.assertEqual(config.highpass_sigma_px, 0.0)
        self.assertEqual(config.contrast_sigma_px, 0.0)
        self.assertAlmostEqual(config.lowpass_sigma_px, 3.0)

    def test_rejects_a_non_positive_template_size(self):
        with self.assertRaises(ValueError):
            PreprocessConfig().scaled_for(0.0)

    def test_preprocess_runs_end_to_end(self):
        out = preprocess(np.random.default_rng(0).random((64, 64)))
        self.assertEqual(out.shape, (64, 64))
        self.assertTrue(np.all(np.isfinite(out)))


class TestWarpTemplate(unittest.TestCase):
    def test_identity_is_a_no_op(self):
        image = np.random.default_rng(0).random((32, 32))
        np.testing.assert_allclose(warp_template(image, 1.0, 0.0), image)

    def test_output_keeps_its_shape(self):
        self.assertEqual(warp_template(np.zeros((40, 40)), 1.1, 5.0).shape, (40, 40))

    def test_rotation_preserves_the_centre(self):
        image = np.zeros((41, 41))
        image[20, 20] = 1.0
        warped = warp_template(image, 1.0, 7.0)
        self.assertEqual(np.unravel_index(int(np.argmax(warped)), warped.shape), (20, 20))

    def test_rotating_by_the_inverse_recovers_the_original(self):
        rng = np.random.default_rng(2)
        image = box_blur(rng.random((51, 51)), 3)  # smooth, so resampling is faithful
        there_and_back = warp_template(warp_template(image, 1.0, 6.0), 1.0, -6.0)
        interior = (slice(12, 39), slice(12, 39))
        self.assertGreater(
            float(np.corrcoef(image[interior].ravel(), there_and_back[interior].ravel())[0, 1]),
            0.97,
        )

    def test_non_positive_scale_is_rejected(self):
        with self.assertRaises(ValueError):
            warp_template(np.zeros((8, 8)), 0.0, 0.0)


class TestWeightedCorrelation(unittest.TestCase):
    def test_uniform_weights_reproduce_plain_ncc(self):
        rng = np.random.default_rng(0)
        search = rng.random((40, 44))
        template = search[7:19, 5:17].copy()
        plain = normalised_cross_correlation(search, template)
        weighted = weighted_normalised_cross_correlation(
            search, template, np.ones_like(template)
        )
        np.testing.assert_allclose(plain, weighted, atol=1e-10)

    def test_exact_match_still_scores_one(self):
        rng = np.random.default_rng(3)
        search = rng.random((32, 32))
        template = search[8:20, 8:20].copy()
        weights = rng.random(template.shape) + 0.1
        surface = weighted_normalised_cross_correlation(search, template, weights)
        self.assertAlmostEqual(float(surface.max()), 1.0, places=9)
        self.assertEqual(np.unravel_index(int(np.argmax(surface)), surface.shape), (8, 8))

    def test_weights_redirect_the_match(self):
        """Zero-weighted template content must stop influencing the score."""
        rng = np.random.default_rng(5)
        search = rng.random((60, 60))
        template = search[10:30, 10:30].copy()
        # Corrupt half the template, then weight that half out.
        template[:, 10:] = rng.random((20, 10))
        weights = np.ones_like(template)
        weights[:, 10:] = 0.0
        surface = weighted_normalised_cross_correlation(search, template, weights)
        self.assertEqual(np.unravel_index(int(np.argmax(surface)), surface.shape), (10, 10))

    def test_rejects_mismatched_weight_shape(self):
        with self.assertRaises(ValueError):
            weighted_normalised_cross_correlation(
                np.zeros((20, 20)), np.zeros((5, 5)), np.zeros((4, 4))
            )

    def test_rejects_negative_weights(self):
        with self.assertRaises(ValueError):
            weighted_normalised_cross_correlation(
                np.random.default_rng(0).random((20, 20)), np.ones((5, 5)), -np.ones((5, 5))
            )

    def test_rejects_zero_total_weight(self):
        with self.assertRaises(ValueError):
            weighted_normalised_cross_correlation(
                np.random.default_rng(0).random((20, 20)), np.ones((5, 5)), np.zeros((5, 5))
            )


class TestUniquenessWeights(unittest.TestCase):
    def test_a_mat_boundary_outweighs_the_periodic_interior(self):
        """The whole point: the repeating interior cannot resolve position.

        Tested on a real generated template rather than a synthetic lattice. The
        crop origin is chosen so a mat boundary crosses the template at its
        midpoint, and the weight there must exceed the weight in the uniform
        array to either side.
        """
        from driftsense.preprocess import PreprocessConfig, preprocess
        from driftsense.resample import area_average_reduce

        pair = make_pair()
        template = area_average_reduce(pair.reference, GEOMETRY.zoom_ratio)
        prepared = preprocess(template, PreprocessConfig().scaled_for(template.shape[0]))
        weights = uniqueness_weights(prepared)

        boundary = float(weights[:, 46:54].mean())
        interior = float(weights[:, 10:40].mean())
        self.assertGreater(boundary, interior)

    def test_a_featureless_lattice_produces_structured_weights(self):
        """Documents the weighting produced on a perfectly periodic template.

        On such a template there is nothing anywhere to discriminate on, and the
        lag search selects a multiple of the period rather than the fundamental,
        so the weight map carries structure rather than being flat. The weights
        stay finite, positive and bounded, which is what the rest of the pipeline
        requires of them. Real generated templates contain aperiodic structure and
        do not reach this case.
        """
        rows, cols = np.mgrid[0:100, 0:100]
        lattice = np.sin(rows * 2 * np.pi / 10.0) + np.sin(cols * 2 * np.pi / 10.0)
        weights = uniqueness_weights(lattice)
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertGreaterEqual(float(weights.min()), 0.0)
        self.assertGreater(float(weights.mean()), 0.0)
        self.assertGreaterEqual(float(weights.std()), 0.25 * float(weights.mean()))

    def test_weights_stay_within_the_floor_and_one(self):
        weights = uniqueness_weights(make_pair().reference[:200, :200].astype(float))
        self.assertGreaterEqual(float(weights.min()), 0.0)
        self.assertLessEqual(float(weights.max()), 1.0)

    def test_a_floor_is_retained_so_the_interior_still_contributes(self):
        rows, cols = np.mgrid[0:80, 0:80]
        lattice = np.sin(rows * 2 * np.pi / 8.0) + np.sin(cols * 2 * np.pi / 8.0)
        self.assertGreater(float(uniqueness_weights(lattice).min()), 0.0)

    def test_flat_input_returns_uniform_weights(self):
        np.testing.assert_allclose(uniqueness_weights(np.zeros((32, 32))), 1.0)

    def test_shape_is_preserved(self):
        self.assertEqual(uniqueness_weights(np.random.default_rng(0).random((48, 48))).shape, (48, 48))


class TestHypothesisSearch(unittest.TestCase):
    def test_config_rejects_invalid_search_settings(self):
        for kwargs in (
            {"rotations_deg": ()},
            {"scales": ()},
            {"scales": (0.0,)},
            {"coarse_factor": 0},
            {"coarse_keep": 0},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    LocalisationConfig(**kwargs)

    def test_the_selected_hypothesis_is_reported(self):
        result = locate(*(lambda p: (p.reference, p.search))(make_pair(rotation_deg=2.0)))
        self.assertIsNotNone(result.hypothesis)
        self.assertGreater(result.n_hypotheses, 1)

    def test_a_clear_tilt_selects_a_counter_rotation(self):
        """At 3 degrees the evidence is strong enough to clear the gate.

        At smaller tilts the gate may legitimately keep the identity hypothesis --
        that conservatism is the point of it, and accuracy stays acceptable
        either way.
        """
        pair = make_pair(rotation_deg=3.0)
        result = locate(pair.reference, pair.search)
        self.assertLess(result.hypothesis.rotation_deg, 0.0)

    def test_rotation_search_beats_no_search_at_three_degrees(self):
        pair = make_pair(rotation_deg=3.0)
        gt = pair.ground_truth

        def error(config):
            outcome = locate(pair.reference, pair.search, config)
            return float(np.hypot(outcome.x - gt.x, outcome.y - gt.y))

        without = error(
            LocalisationConfig(rotations_deg=(0.0,), scales=(1.0,), preprocess=None)
        )
        self.assertLess(error(LocalisationConfig()), without)

    def test_gating_keeps_the_identity_hypothesis_when_unrotated(self):
        """A warped hypothesis must earn its place against the untransformed one.

        Selecting on raw peak height alone measured worse than no search at all
        on the full dataset; the margin is what prevents that.
        """
        pair = make_pair(rotation_deg=0.0)
        # Refinement is switched off here: it deliberately moves the hypothesis
        # off the discrete grid afterwards, so leaving it on would test the
        # refinement rather than the gate.
        result = locate(
            pair.reference,
            pair.search,
            LocalisationConfig(hypothesis_margin=10.0, refine=False),
        )
        self.assertEqual(result.hypothesis, Hypothesis(scale=1.0, rotation_deg=0.0))

    def test_result_serialises_the_hypothesis(self):
        payload = locate(*(_ := (make_pair().reference, make_pair().search))).as_dict()
        self.assertIn("hypothesis", payload)
        self.assertIn("n_hypotheses", payload)

    def test_uniqueness_weighting_can_be_disabled(self):
        pair = make_pair()
        plain = locate(pair.reference, pair.search, LocalisationConfig(uniqueness_weighting=False))
        weighted = locate(pair.reference, pair.search, LocalisationConfig())
        for outcome in (plain, weighted):
            self.assertTrue(np.isfinite(outcome.x))
            self.assertTrue(np.isfinite(outcome.y))


class TestAlignmentRefinement(unittest.TestCase):
    """The fifth increment: polishing off the discrete search grid."""

    def _prepared(self, rotation_deg=0.0):
        from driftsense.preprocess import PreprocessConfig, preprocess
        from driftsense.resample import area_average_reduce

        pair = make_pair(rotation_deg=rotation_deg)
        template = area_average_reduce(pair.reference, GEOMETRY.zoom_ratio)
        config = PreprocessConfig().scaled_for(template.shape[0])
        return pair, preprocess(pair.search, config), preprocess(template, config)

    def test_refinement_pulls_a_displaced_start_back(self):
        pair, search, template = self._prepared()
        gt = pair.ground_truth
        x, y, _, _ = refine_alignment(
            search, template, gt.x + 0.7, gt.y - 0.6, Hypothesis(1.0, 0.0)
        )
        before = float(np.hypot(0.7, 0.6))
        after = float(np.hypot(x - gt.x, y - gt.y))
        self.assertLess(after, before)

    def test_refinement_does_not_move_an_already_correct_answer_far(self):
        pair, search, template = self._prepared()
        gt = pair.ground_truth
        x, y, _, _ = refine_alignment(search, template, gt.x, gt.y, Hypothesis(1.0, 0.0))
        self.assertLess(float(np.hypot(x - gt.x, y - gt.y)), 0.6)

    def test_refinement_reports_a_score_in_range(self):
        pair, search, template = self._prepared()
        gt = pair.ground_truth
        *_, score = refine_alignment(search, template, gt.x, gt.y, Hypothesis(1.0, 0.0))
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)

    def test_refinement_never_returns_a_non_positive_scale(self):
        pair, search, template = self._prepared()
        gt = pair.ground_truth
        *_, hypothesis, _ = refine_alignment(
            search, template, gt.x, gt.y, Hypothesis(0.02, 0.0), iterations=6
        )
        self.assertGreater(hypothesis.scale, 0.0)

    def test_refinement_improves_accuracy_end_to_end(self):
        pair = make_pair(rotation_deg=1.5)
        gt = pair.ground_truth

        def error(refine: bool) -> float:
            out = locate(pair.reference, pair.search, LocalisationConfig(refine=refine))
            return float(np.hypot(out.x - gt.x, out.y - gt.y))

        self.assertLessEqual(error(True), error(False) + 1e-9)

    def test_refinement_can_be_disabled(self):
        pair = make_pair()
        out = locate(pair.reference, pair.search, LocalisationConfig(refine=False))
        self.assertTrue(np.isfinite(out.x) and np.isfinite(out.y))


if __name__ == "__main__":
    unittest.main()

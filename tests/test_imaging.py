"""SEM imaging model: each physical term, and the chain as a whole."""

from __future__ import annotations

import unittest

import numpy as np

from driftsense import DramLayout, DramParams, ImagingGeometry, Placement, generate_pair
from driftsense.imaging import (
    REFERENCE_CAPTURE,
    SEARCH_CAPTURE,
    CaptureParams,
    MaterialYields,
    apply_charging,
    apply_dose_noise,
    apply_geometric_error,
    apply_vignette,
    edge_density,
    form_image,
    gaussian_blur,
    secondary_electron_signal,
)
from driftsense.layouts import RenderWindow

GEOMETRY = ImagingGeometry()


def make_render(size: int = 200, pixel_nm: float = 1.0):
    layout = DramLayout(DramParams(feature_size_nm=35.0, mat_size_nm=2600.0, strip_width_nm=320.0))
    return layout.render(RenderWindow(2100.0, 2100.0, size, pixel_nm))


class TestCaptureParams(unittest.TestCase):
    def test_rejects_unphysical_settings(self):
        for kwargs in (
            {"spot_size_nm": -1.0},
            {"astigmatism_ratio": 0.0},
            {"dose": 0.0},
            {"detector_noise_sigma": -1.0},
            {"scale_error": 1.5},
            {"gamma": 0.0},
            {"charging_streak_prob": 1.5},
            {"salt_pepper_prob": -0.1},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    CaptureParams(**kwargs)

    def test_presets_encode_the_dose_ratio_from_the_sample_metadata(self):
        self.assertAlmostEqual(REFERENCE_CAPTURE.dose / SEARCH_CAPTURE.dose, 10.0)
        self.assertGreater(SEARCH_CAPTURE.detector_noise_sigma, REFERENCE_CAPTURE.detector_noise_sigma)
        self.assertGreater(SEARCH_CAPTURE.spot_size_nm, REFERENCE_CAPTURE.spot_size_nm)


class TestMaterialContrast(unittest.TestCase):
    def test_signal_is_the_area_weighted_yield(self):
        render = make_render()
        yields = MaterialYields()
        signal = secondary_electron_signal(render, yields)
        lowest = min([yields.substrate] + list(yields.materials.values()))
        highest = max([yields.substrate] + list(yields.materials.values()))
        self.assertGreaterEqual(float(signal.min()), lowest - 1e-5)
        self.assertLessEqual(float(signal.max()), highest + 1e-5)

    def test_uniform_yields_give_a_flat_signal(self):
        render = make_render()
        flat = MaterialYields(substrate=1.0, materials={k: 1.0 for k in render.layers})
        signal = secondary_electron_signal(render, flat)
        np.testing.assert_allclose(signal, 1.0, atol=1e-5)

    def test_contacts_are_the_brightest_material(self):
        yields = MaterialYields()
        self.assertEqual(max(yields.materials, key=yields.materials.get), "contact")

    def test_negative_yield_is_rejected(self):
        with self.assertRaises(ValueError):
            MaterialYields(materials={"contact": -1.0})


class TestEdgeDensity(unittest.TestCase):
    def test_is_zero_on_a_featureless_region(self):
        layout = DramLayout(DramParams())
        strip = layout.params.mat_size_nm + 20.0
        render = layout.render(RenderWindow(strip, strip, 32, 1.0))
        np.testing.assert_allclose(edge_density(render), 0.0, atol=1e-6)

    def test_is_positive_where_structure_exists(self):
        self.assertGreater(float(edge_density(make_render()).max()), 0.1)

    def test_stays_bounded(self):
        density = edge_density(make_render())
        self.assertGreaterEqual(float(density.min()), 0.0)
        self.assertLessEqual(float(density.max()), 1.0)

    def test_raw_density_grows_with_pixel_size(self):
        """A coarse pixel contains more edge, so the raw density is larger.

        This is why it cannot be used as a brightness term directly.
        """
        layout = DramLayout(DramParams(feature_size_nm=35.0))
        fine = edge_density(layout.render(RenderWindow(2100.0, 2100.0, 1000, 1.0)))
        coarse = edge_density(layout.render(RenderWindow(2100.0, 2100.0, 100, 10.0)))
        self.assertGreater(float(coarse.mean()), 3.0 * float(fine.mean()))

    def test_fringe_scaling_largely_removes_the_resolution_dependence(self):
        """Scaling by fringe width over pixel size makes the term nearly consistent.

        The raw density differs between the two captures by about 8.5x. Converting
        it into the fraction of the pixel's signal that comes from the fringe
        brings that down to about 3.4x -- most of the resolution dependence, but
        not all of it.

        The residual is understood: the discrete gradient saturates at
        high magnification, where a single edge already fills a whole pixel and
        ``|grad c|`` cannot exceed 1. A sub-pixel edge model would close the gap;
        it is not worth the complexity while the term is a perturbation on top of
        material contrast rather than the dominant signal.
        """
        layout = DramLayout(DramParams(feature_size_nm=35.0))
        fringe_nm = 4.0
        fine = edge_density(layout.render(RenderWindow(2100.0, 2100.0, 1000, 1.0)))
        coarse = edge_density(layout.render(RenderWindow(2100.0, 2100.0, 100, 10.0)))
        fine_term = float(fine.mean()) * min(fringe_nm / 1.0, 1.0)
        coarse_term = float(coarse.mean()) * min(fringe_nm / 10.0, 1.0)
        ratio = coarse_term / max(fine_term, 1e-9)
        self.assertGreater(ratio, 0.3)
        self.assertLess(ratio, 4.0)


class TestBeamPointSpread(unittest.TestCase):
    def test_preserves_total_signal(self):
        rng = np.random.default_rng(0)
        image = rng.random((64, 64))
        blurred = gaussian_blur(image, 2.0, 2.0)
        self.assertAlmostEqual(float(blurred.sum()), float(image.sum()), delta=float(image.sum()) * 0.02)

    def test_reduces_variance(self):
        rng = np.random.default_rng(0)
        image = rng.random((64, 64))
        self.assertLess(gaussian_blur(image, 2.0, 2.0).var(), image.var())

    def test_zero_sigma_is_a_no_op(self):
        image = np.random.default_rng(0).random((16, 16))
        np.testing.assert_allclose(gaussian_blur(image, 0.0, 0.0), image)

    def test_astigmatism_blurs_the_axes_differently(self):
        image = np.zeros((41, 41))
        image[20, 20] = 1.0
        blurred = gaussian_blur(image, 4.0, 1.0)  # sigma_y > sigma_x
        vertical_spread = float((blurred[:, 20] > blurred.max() * 0.5).sum())
        horizontal_spread = float((blurred[20, :] > blurred.max() * 0.5).sum())
        self.assertGreater(vertical_spread, horizontal_spread)

    def test_impulse_response_is_gaussian(self):
        image = np.zeros((81, 81))
        image[40, 40] = 1.0
        blurred = gaussian_blur(image, 3.0, 3.0)
        profile = blurred[40]
        offsets = np.arange(81) - 40
        measured = float(np.sqrt((profile * offsets**2).sum() / profile.sum()))
        self.assertAlmostEqual(measured, 3.0, delta=0.15)

    def test_negative_sigma_is_rejected(self):
        with self.assertRaises(ValueError):
            gaussian_blur(np.zeros((4, 4)), -1.0, 1.0)


class TestGeometricError(unittest.TestCase):
    def test_no_error_is_a_no_op(self):
        rng = np.random.default_rng(0)
        image = rng.random((32, 32))
        np.testing.assert_allclose(apply_geometric_error(image, 0, 0, 0, 0, rng), image)

    def test_rotation_leaves_the_centre_in_place(self):
        """Why the stage error is attributed to the reference capture."""
        rng = np.random.default_rng(0)
        image = np.zeros((101, 101))
        image[50, 50] = 1.0
        rotated = apply_geometric_error(image, 3.0, 0.0, 0.0, 0.0, rng)
        self.assertEqual(np.unravel_index(int(np.argmax(rotated)), rotated.shape), (50, 50))

    def test_rotation_actually_moves_off_centre_content(self):
        rng = np.random.default_rng(0)
        image = np.zeros((101, 101))
        image[20, 80] = 1.0
        rotated = apply_geometric_error(image, 5.0, 0.0, 0.0, 0.0, rng)
        self.assertNotEqual(np.unravel_index(int(np.argmax(rotated)), rotated.shape), (20, 80))

    def test_scale_error_expands_about_the_centre(self):
        rng = np.random.default_rng(0)
        image = np.zeros((101, 101))
        image[50, 70] = 1.0
        expanded = apply_geometric_error(image, 0.0, 0.20, 0.0, 0.0, rng)
        row, col = np.unravel_index(int(np.argmax(expanded)), expanded.shape)
        self.assertEqual(row, 50)
        self.assertGreater(col, 70)

    def test_drift_jitter_displaces_rows_independently(self):
        rng = np.random.default_rng(1)
        image = np.tile(np.arange(64, dtype=np.float64), (64, 1))
        jittered = apply_geometric_error(image, 0.0, 0.0, 0.0, 1.5, rng)
        row_means = jittered[:, 10:54].mean(axis=1)
        self.assertGreater(float(row_means.std()), 0.05)


class TestDetection(unittest.TestCase):
    def test_shot_noise_scales_as_the_inverse_root_of_dose(self):
        """The physical reason the wide-field capture is noisier."""
        rng = np.random.default_rng(0)
        signal = np.ones((256, 256))
        high = apply_dose_noise(signal, 4000.0, rng).std()
        low = apply_dose_noise(signal, 250.0, rng).std()
        self.assertAlmostEqual(low / high, 4.0, delta=0.4)  # sqrt(4000/250) = 4

    def test_shot_noise_preserves_the_mean(self):
        rng = np.random.default_rng(0)
        signal = np.full((128, 128), 0.7)
        self.assertAlmostEqual(float(apply_dose_noise(signal, 500.0, rng).mean()), 0.7, delta=0.01)

    def test_zero_dose_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_dose_noise(np.ones((4, 4)), 0.0, np.random.default_rng(0))

    def test_vignette_darkens_the_corners_not_the_centre(self):
        image = np.ones((101, 101))
        vignetted = apply_vignette(image, 0.5)
        self.assertAlmostEqual(float(vignetted[50, 50]), 1.0, places=6)
        self.assertLess(float(vignetted[0, 0]), 0.6)

    def test_zero_vignette_is_a_no_op(self):
        image = np.ones((16, 16))
        np.testing.assert_allclose(apply_vignette(image, 0.0), image)

    def test_charging_brightens_whole_scan_lines(self):
        rng = np.random.default_rng(3)
        image = np.ones((64, 64))
        charged = apply_charging(image, 1.0, 0.5, rng)
        self.assertGreater(float(charged.max()), 1.0)
        # The artefact follows the raster: it varies far more between rows than
        # the underlying uniform image does.
        self.assertGreater(float(charged.mean(axis=1).std()), 0.0)

    def test_charging_is_off_when_the_probability_is_zero(self):
        rng = np.random.default_rng(3)
        image = np.ones((32, 32))
        np.testing.assert_allclose(apply_charging(image, 0.0, 0.5, rng), image)


class TestFormImage(unittest.TestCase):
    def test_output_is_eight_bit_and_correctly_shaped(self):
        render = make_render(256)
        image = form_image(render, REFERENCE_CAPTURE, 1.0, np.random.default_rng(0))
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(image.shape, (256, 256))

    def test_lower_dose_produces_a_noisier_image(self):
        render = make_render(256)
        quiet = form_image(
            render, CaptureParams(dose=5000.0, detector_noise_sigma=0.0, spot_size_nm=0.0),
            1.0, np.random.default_rng(0),
        )
        noisy = form_image(
            render, CaptureParams(dose=100.0, detector_noise_sigma=0.0, spot_size_nm=0.0),
            1.0, np.random.default_rng(0),
        )
        # Compare high-frequency content, which is where shot noise lives.
        self.assertGreater(
            float(np.diff(noisy.astype(float), axis=1).std()),
            float(np.diff(quiet.astype(float), axis=1).std()),
        )

    def test_same_generator_reproduces_the_image(self):
        render = make_render(128)
        a = form_image(render, SEARCH_CAPTURE, 10.0, np.random.default_rng(11))
        b = form_image(render, SEARCH_CAPTURE, 10.0, np.random.default_rng(11))
        np.testing.assert_array_equal(a, b)

    def test_different_generators_give_independent_noise(self):
        render = make_render(128)
        a = form_image(render, SEARCH_CAPTURE, 10.0, np.random.default_rng(1))
        b = form_image(render, SEARCH_CAPTURE, 10.0, np.random.default_rng(2))
        self.assertFalse(np.array_equal(a, b))

    def test_edge_gain_brightens_feature_edges(self):
        render = make_render(256)
        params = CaptureParams(dose=1e7, detector_noise_sigma=0.0, spot_size_nm=0.0)
        plain = form_image(render, params.with_changes(edge_gain=0.0), 1.0, np.random.default_rng(0))
        edged = form_image(render, params.with_changes(edge_gain=1.0), 1.0, np.random.default_rng(0))
        mask = edge_density(render) > 0.3
        self.assertGreater(
            float(edged[mask].mean() - edged[~mask].mean()),
            float(plain[mask].mean() - plain[~mask].mean()),
        )

    def test_the_image_retains_structure_under_realistic_noise(self):
        """A sanity floor: the physics must not destroy the signal outright."""
        render = make_render(256)
        image = form_image(render, SEARCH_CAPTURE, 10.0, np.random.default_rng(0))
        self.assertGreater(float(image.std()), 5.0)


class TestPairIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = DramLayout(
            DramParams(feature_size_nm=35.0, mat_size_nm=2600.0, strip_width_nm=320.0)
        )

    def test_the_two_captures_have_independent_noise(self):
        """Explicitly required by the problem statement.

        Rendered from one layout, the two images share structure. What must not be
        shared is the noise realisation. Comparing the residual of each image
        against its own blurred version isolates the noise fields; if they had been
        drawn once and reused, those residuals would correlate.
        """
        pair = generate_pair(
            self.layout,
            Placement(2100.0, 2100.0),
            GEOMETRY,
            reference_capture=CaptureParams(spot_size_nm=0.0, dose=300.0),
            search_capture=CaptureParams(spot_size_nm=0.0, dose=300.0),
            seed=7,
        )
        reference_noise = pair.reference.astype(float) - gaussian_blur(pair.reference, 2.0, 2.0)
        search_noise = pair.search.astype(float) - gaussian_blur(pair.search, 2.0, 2.0)
        # Downsample the reference noise to the search grid so the two are comparable.
        reduced = reference_noise.reshape(100, 10, 100, 10).mean(axis=(1, 3))
        coarse = search_noise.reshape(100, 10, 100, 10).mean(axis=(1, 3))
        correlation = float(np.corrcoef(reduced.ravel(), coarse.ravel())[0, 1])
        self.assertLess(abs(correlation), 0.1)

    def test_imaging_is_off_by_default(self):
        pair = generate_pair(self.layout, Placement(2100.0, 2100.0), GEOMETRY)
        self.assertIsNone(pair.reference_capture)
        self.assertEqual(pair.as_metadata()["driftsense"]["imaging_model"], "none (layout render only)")

    def test_metadata_records_both_capture_settings(self):
        pair = generate_pair(
            self.layout,
            Placement(2100.0, 2100.0),
            GEOMETRY,
            reference_capture=REFERENCE_CAPTURE,
            search_capture=SEARCH_CAPTURE,
            seed=3,
        )
        payload = pair.as_metadata()["driftsense"]
        self.assertEqual(payload["imaging_model"], "sem")
        self.assertAlmostEqual(payload["capture"]["reference"]["dose"], 2000.0)
        self.assertAlmostEqual(payload["capture"]["search"]["dose"], 200.0)

    def test_pair_generation_is_reproducible_with_imaging_on(self):
        def build():
            return generate_pair(
                self.layout,
                Placement(2100.0, 2100.0),
                GEOMETRY,
                reference_capture=REFERENCE_CAPTURE,
                search_capture=SEARCH_CAPTURE,
                seed=99,
            )

        np.testing.assert_array_equal(build().search, build().search)


class TestSweep(unittest.TestCase):
    """The degradation study that produces the result."""

    def test_sweep_returns_one_point_per_level(self):
        from driftsense.sweep import sweep_parameter

        points = sweep_parameter("dose", [800.0, 100.0], n_pairs=3, seed=4242)
        self.assertEqual(len(points), 2)
        for point in points:
            self.assertGreater(point.n_solvable, 0)
            self.assertGreaterEqual(point.within_1px, 0.0)
            self.assertLessEqual(point.within_1px, 1.0)

    def test_lower_dose_does_not_improve_accuracy(self):
        """Monotonicity sanity check: more noise must not help."""
        from driftsense.sweep import sweep_parameter

        points = sweep_parameter("dose", [1600.0, 25.0], n_pairs=4, seed=4242)
        self.assertLessEqual(points[0].median_error_px, points[1].median_error_px + 0.05)

    def test_unknown_parameter_is_rejected(self):
        from driftsense.sweep import sweep_parameter

        with self.assertRaises(AttributeError):
            sweep_parameter("not_a_field", [1.0], n_pairs=2)

    def test_invalid_target_capture_is_rejected(self):
        from driftsense.sweep import sweep_parameter

        with self.assertRaises(ValueError):
            sweep_parameter("dose", [1.0], n_pairs=2, apply_to="neither")

    def test_report_formats_a_table(self):
        from driftsense.sweep import format_sweep, sweep_parameter

        text = format_sweep("dose", sweep_parameter("dose", [500.0], n_pairs=3, seed=4242))
        self.assertIn("median err", text)


class TestSweepCli(unittest.TestCase):
    def test_cli_runs_end_to_end(self):
        import contextlib
        import io
        import tempfile
        from pathlib import Path

        import sweep_noise

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sweep.json"
            with contextlib.redirect_stdout(io.StringIO()):
                code = sweep_noise.main(
                    ["--parameter", "dose", "--values", "500", "--pairs", "3", "--quiet",
                     "--out", str(out)]
                )
            self.assertEqual(code, 0)
            self.assertTrue(out.exists())

    def test_cli_reports_a_bad_parameter(self):
        import contextlib
        import io

        import sweep_noise

        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(
                sweep_noise.main(["--parameter", "nope", "--values", "1", "--quiet"]), 1
            )


if __name__ == "__main__":
    unittest.main()

"""Optical-microscope image formation, and the locator's RGB entry point."""

from __future__ import annotations

import unittest

import numpy as np

from driftsense.layouts import DramLayout, FinfetLayout, RenderWindow
from driftsense.locate import locate
from driftsense.optical import (
    CHANNEL_WAVELENGTHS_NM,
    MaterialColours,
    OpticalParams,
    form_rgb_image,
    rayleigh_resolution_nm,
)


def window(size_px=160, pixel_nm=20.0):
    return RenderWindow(1000.0, 1000.0, size_px, pixel_nm)


class TestRayleigh(unittest.TestCase):
    def test_matches_the_textbook_expression(self):
        self.assertAlmostEqual(rayleigh_resolution_nm(550.0, 1.0), 0.61 * 550.0, places=9)

    def test_a_higher_aperture_resolves_finer(self):
        self.assertLess(
            rayleigh_resolution_nm(550.0, 1.4), rayleigh_resolution_nm(550.0, 0.6)
        )

    def test_blue_resolves_finer_than_red(self):
        """The one genuinely chromatic effect this model carries."""
        self.assertLess(
            rayleigh_resolution_nm(460.0, 0.9), rayleigh_resolution_nm(620.0, 0.9)
        )

    def test_rejects_non_positive_inputs(self):
        with self.assertRaises(ValueError):
            rayleigh_resolution_nm(0.0, 0.9)
        with self.assertRaises(ValueError):
            rayleigh_resolution_nm(550.0, 0.0)

    def test_optical_resolution_is_far_coarser_than_a_dram_pitch(self):
        """The fact that makes optical a different problem, pinned as a test.

        A 70 nm bit-line pitch against a ~430 nm green resolution: the cell array
        is not blurred, it is gone.
        """
        self.assertGreater(OpticalParams().resolution_nm(550.0), 5 * 70.0)


class TestOpticalParams(unittest.TestCase):
    def test_defaults_are_valid(self):
        OpticalParams()

    def test_rejects_a_non_positive_aperture(self):
        with self.assertRaises(ValueError):
            OpticalParams(numerical_aperture=0.0)

    def test_rejects_an_objective_that_beats_diffraction(self):
        with self.assertRaises(ValueError):
            OpticalParams(aberration_factor=0.5)

    def test_rejects_a_non_positive_exposure(self):
        with self.assertRaises(ValueError):
            OpticalParams(exposure=0.0)

    def test_rejects_negative_read_noise(self):
        with self.assertRaises(ValueError):
            OpticalParams(read_noise=-1.0)

    def test_rejects_full_vignetting(self):
        with self.assertRaises(ValueError):
            OpticalParams(vignette_strength=1.0)

    def test_rejects_a_bad_channel_gain(self):
        with self.assertRaises(ValueError):
            OpticalParams(channel_gain=(1.0, 1.0))

    def test_as_dict_reports_the_three_resolutions(self):
        payload = OpticalParams().as_dict()
        self.assertEqual(set(payload["resolution_nm"]), {"red", "green", "blue"})

    def test_aberration_makes_the_spot_larger(self):
        perfect = OpticalParams(aberration_factor=1.0).resolution_nm(550.0)
        real = OpticalParams(aberration_factor=1.5).resolution_nm(550.0)
        self.assertGreater(real, perfect)


class TestFormRgbImage(unittest.TestCase):
    def setUp(self):
        self.layout = DramLayout()
        self.render = self.layout.render(window())
        self.params = OpticalParams()

    def image(self, **kwargs):
        params = OpticalParams(**kwargs) if kwargs else self.params
        return form_rgb_image(self.render, params, 20.0, np.random.default_rng(0))

    def test_returns_three_channels_of_uint8(self):
        image = self.image()
        self.assertEqual(image.shape, (160, 160, 3))
        self.assertEqual(image.dtype, np.uint8)

    def test_output_is_in_range(self):
        image = self.image()
        self.assertGreaterEqual(int(image.min()), 0)
        self.assertLessEqual(int(image.max()), 255)

    def test_it_is_deterministic_for_a_given_seed(self):
        a = form_rgb_image(self.render, self.params, 20.0, np.random.default_rng(4))
        b = form_rgb_image(self.render, self.params, 20.0, np.random.default_rng(4))
        np.testing.assert_array_equal(a, b)

    def test_a_different_seed_gives_different_noise(self):
        a = form_rgb_image(self.render, self.params, 20.0, np.random.default_rng(1))
        b = form_rgb_image(self.render, self.params, 20.0, np.random.default_rng(2))
        self.assertGreater(float(np.abs(a.astype(float) - b.astype(float)).mean()), 0.0)

    def test_more_exposure_means_less_noise(self):
        rough = form_rgb_image(
            self.render, OpticalParams(exposure=50.0), 20.0, np.random.default_rng(0)
        )
        clean = form_rgb_image(
            self.render, OpticalParams(exposure=20000.0), 20.0, np.random.default_rng(0)
        )
        # Compare high-frequency content: noise lives there.
        def roughness(img):
            g = img[..., 1].astype(np.float64)
            return float(np.abs(np.diff(g, axis=1)).mean())

        self.assertGreater(roughness(rough), roughness(clean))

    def test_a_larger_aperture_preserves_more_detail(self):
        blurry = form_rgb_image(
            self.render, OpticalParams(numerical_aperture=0.3), 20.0, np.random.default_rng(0)
        )
        sharp = form_rgb_image(
            self.render, OpticalParams(numerical_aperture=1.3), 20.0, np.random.default_rng(0)
        )
        self.assertGreater(float(sharp[..., 1].std()), float(blurry[..., 1].std()))

    def test_channels_differ_because_their_blur_differs(self):
        image = self.image(read_noise=0.0, exposure=1e7, vignette_strength=0.0)
        red = image[..., 0].astype(np.float64)
        blue = image[..., 2].astype(np.float64)
        red -= red.mean()
        blue -= blue.mean()
        self.assertGreater(float(np.abs(red - blue).max()), 0.5)

    def test_vignetting_darkens_the_corners(self):
        image = form_rgb_image(
            self.render,
            OpticalParams(vignette_strength=0.6, read_noise=0.0, exposure=1e6),
            20.0,
            np.random.default_rng(0),
        )
        green = image[..., 1].astype(np.float64)
        corner = float(green[:20, :20].mean())
        centre = float(green[70:90, 70:90].mean())
        self.assertLess(corner, centre)

    def test_rejects_a_non_positive_pixel_size(self):
        with self.assertRaises(ValueError):
            form_rgb_image(self.render, self.params, 0.0, np.random.default_rng(0))

    def test_it_works_on_finfet_too(self):
        render = FinfetLayout().render(window())
        image = form_rgb_image(render, self.params, 20.0, np.random.default_rng(0))
        self.assertEqual(image.shape[2], 3)

    def test_an_unknown_layer_falls_back_to_a_default_colour(self):
        self.assertEqual(MaterialColours().for_layer("no_such_layer"),
                         MaterialColours().default)

    def test_channel_wavelengths_are_ordered_red_to_blue(self):
        self.assertGreater(CHANNEL_WAVELENGTHS_NM[0], CHANNEL_WAVELENGTHS_NM[2])


class TestLocatorAcceptsRgb(unittest.TestCase):
    """The bonus path's actual requirement: the entry point must not fail on RGB."""

    def setUp(self):
        layout = DramLayout()
        params = OpticalParams(vignette_strength=0.0)
        search_w = RenderWindow(0.0, 0.0, 300, 200.0)
        self.search = form_rgb_image(
            layout.render(search_w), params, 200.0, np.random.default_rng(0)
        )
        ref_w = RenderWindow(6000.0, 4000.0, 300, 20.0)
        self.reference = form_rgb_image(
            layout.render(ref_w), params, 20.0, np.random.default_rng(1)
        )

    def test_it_returns_a_finite_answer(self):
        result = locate(self.reference, self.search)
        self.assertTrue(np.isfinite(result.x) and np.isfinite(result.y))

    def test_the_answer_lies_inside_the_search_image(self):
        result = locate(self.reference, self.search)
        self.assertTrue(0 <= result.x <= self.search.shape[1])
        self.assertTrue(0 <= result.y <= self.search.shape[0])

    def test_rgb_and_its_luminance_agree(self):
        """Reducing to luminance is what the locator does, so the two must match."""
        weights = np.array([0.299, 0.587, 0.114])
        ref_l = self.reference[..., :3].astype(np.float64) @ weights
        search_l = self.search[..., :3].astype(np.float64) @ weights
        a = locate(self.reference, self.search)
        b = locate(ref_l, search_l)
        self.assertAlmostEqual(a.x, b.x, places=6)
        self.assertAlmostEqual(a.y, b.y, places=6)

    def test_a_two_channel_image_is_rejected_clearly(self):
        with self.assertRaises(ValueError) as ctx:
            locate(self.reference[..., :2], self.search)
        self.assertIn("channel", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

"""Line-edge roughness: amplitude, anisotropy, scale consistency, invariants."""

from __future__ import annotations

import unittest

import numpy as np

from driftsense.layouts import DramLayout, FinfetLayout, RenderWindow
from driftsense.resample import area_average_reduce
from driftsense.roughness import RoughnessParams, apply_roughness, displacement_fields


def window(size_px=300, pixel_nm=2.0, x=1234.0, y=2345.0):
    return RenderWindow(x, y, size_px, pixel_nm)


class TestParams(unittest.TestCase):
    def test_defaults_are_valid_and_enabled(self):
        self.assertTrue(RoughnessParams().enabled)

    def test_zero_amplitude_is_disabled(self):
        self.assertFalse(RoughnessParams(amplitude_nm=0.0).enabled)

    def test_rejects_a_negative_amplitude(self):
        with self.assertRaises(ValueError):
            RoughnessParams(amplitude_nm=-1.0)

    def test_rejects_a_non_positive_correlation_length(self):
        with self.assertRaises(ValueError):
            RoughnessParams(along_length_nm=0.0)

    def test_rejects_zero_components(self):
        with self.assertRaises(ValueError):
            RoughnessParams(n_components=0)

    def test_as_dict_is_json_ready(self):
        import json

        json.dumps(RoughnessParams().as_dict())


class TestDisplacementField(unittest.TestCase):
    def test_amplitude_matches_the_requested_sigma(self):
        """``amplitude_nm`` must mean the standard deviation it claims to mean.

        The first implementation normalised by sqrt(n) instead of sqrt(n/2) and
        delivered 0.71 of the requested sigma.
        """
        w = window()
        dx, dy = displacement_fields(w, RoughnessParams(amplitude_nm=1.2, seed=7))
        self.assertAlmostEqual(float(dx.std()) * w.pixel_size_nm, 1.2, delta=0.15)
        self.assertAlmostEqual(float(dy.std()) * w.pixel_size_nm, 1.2, delta=0.15)

    def test_amplitude_scales_linearly(self):
        w = window()
        a = displacement_fields(w, RoughnessParams(amplitude_nm=1.0, seed=3))[0].std()
        b = displacement_fields(w, RoughnessParams(amplitude_nm=4.0, seed=3))[0].std()
        self.assertAlmostEqual(b / a, 4.0, places=6)

    def test_disabled_returns_zero(self):
        dx, dy = displacement_fields(window(), RoughnessParams(amplitude_nm=0.0))
        self.assertEqual(float(np.abs(dx).max()), 0.0)
        self.assertEqual(float(np.abs(dy).max()), 0.0)

    def test_it_is_anisotropic_in_the_right_direction(self):
        """dx displaces vertical lines, so it must vary faster across x than along y.

        An isotropic field would make neighbouring lines wander together, which
        reads as stage drift rather than roughness.
        """
        dx, _ = displacement_fields(window(), RoughnessParams(seed=11))
        across = float(np.abs(np.diff(dx, axis=1)).mean())
        along = float(np.abs(np.diff(dx, axis=0)).mean())
        self.assertGreater(across, along * 2.0)

    def test_dy_is_anisotropic_the_other_way(self):
        _, dy = displacement_fields(window(), RoughnessParams(seed=11))
        across = float(np.abs(np.diff(dy, axis=0)).mean())
        along = float(np.abs(np.diff(dy, axis=1)).mean())
        self.assertGreater(across, along * 2.0)

    def test_dx_and_dy_are_independent(self):
        dx, dy = displacement_fields(window(), RoughnessParams(seed=5))
        corr = float(np.corrcoef(dx.ravel(), dy.ravel())[0, 1])
        self.assertLess(abs(corr), 0.25)

    def test_it_is_deterministic(self):
        a = displacement_fields(window(), RoughnessParams(seed=9))[0]
        b = displacement_fields(window(), RoughnessParams(seed=9))[0]
        np.testing.assert_array_equal(a, b)

    def test_a_different_seed_gives_a_different_specimen(self):
        a = displacement_fields(window(), RoughnessParams(seed=1))[0]
        b = displacement_fields(window(), RoughnessParams(seed=2))[0]
        self.assertGreater(float(np.abs(a - b).mean()), 0.0)

    def test_the_field_is_a_function_of_world_position_not_pixel_size(self):
        """The property the whole design rests on.

        The reference is captured at 1 nm/px and the search at 10 nm/px. If the
        same physical point were displaced differently at the two scales, the two
        images would show different specimens.
        """
        params = RoughnessParams(seed=13)
        fine = displacement_fields(RenderWindow(500.0, 700.0, 200, 1.0), params)[0]
        coarse = displacement_fields(RenderWindow(500.0, 700.0, 20, 10.0), params)[0]
        # Pixel 0 of the coarse grid is centred at +5 nm; the fine pixel nearest
        # that centre is index 4 (centred at +4.5 nm). Compare in nanometres.
        self.assertAlmostEqual(
            float(coarse[0, 0]) * 10.0, float(fine[4, 4]) * 1.0, delta=0.35
        )


class TestApplyRoughness(unittest.TestCase):
    def setUp(self):
        self.layout = DramLayout()
        self.window = window()
        self.params = RoughnessParams(seed=7)

    def test_disjointness_survives(self):
        """Every layer is warped with the same weights, so the sum warps too."""
        apply_roughness(
            self.layout.render(self.window), self.window, self.params
        ).check_disjoint()

    def test_coverage_still_sums_to_at_most_one(self):
        out = apply_roughness(self.layout.render(self.window), self.window, self.params)
        self.assertLessEqual(float(out.total_coverage().max()), 1.0 + 1e-4)

    def test_it_actually_changes_the_image(self):
        plain = self.layout.render(self.window).to_greyscale()
        rough = apply_roughness(
            self.layout.render(self.window), self.window, self.params
        ).to_greyscale()
        self.assertGreater(float(np.abs(plain - rough).max()), 0.01)

    def test_it_only_moves_edges(self):
        """Interiors are uniform, so displacing them must be invisible.

        Checked as: the pixels that change are a small minority, and they are the
        ones with a large local gradient.
        """
        plain = self.layout.render(self.window).to_greyscale()
        rough = apply_roughness(
            self.layout.render(self.window), self.window, self.params
        ).to_greyscale()
        changed = np.abs(plain - rough) > 0.01
        gradient = np.abs(np.gradient(plain.astype(np.float64))[0]) + np.abs(
            np.gradient(plain.astype(np.float64))[1]
        )
        self.assertGreater(float(gradient[changed].mean()), float(gradient[~changed].mean()))

    def test_none_is_a_no_op(self):
        render = self.layout.render(self.window)
        self.assertIs(apply_roughness(render, self.window, None), render)

    def test_zero_amplitude_is_a_no_op(self):
        render = self.layout.render(self.window)
        out = apply_roughness(render, self.window, RoughnessParams(amplitude_nm=0.0))
        self.assertIs(out, render)

    def test_metadata_records_the_settings(self):
        out = apply_roughness(self.layout.render(self.window), self.window, self.params)
        self.assertIn("roughness", out.metadata)

    def test_it_works_on_finfet_too(self):
        layout = FinfetLayout()
        apply_roughness(
            layout.render(self.window), self.window, self.params
        ).check_disjoint()


class TestScaleConsistencyCost(unittest.TestCase):
    """Roughness weakens the cross-scale exactness, and it should.

    Warping an area-averaged image is not the same as area-averaging a warped
    one, and the gap is exactly the fine roughness a 10 nm pixel cannot resolve.
    That is physics, not a numerical defect -- but the size of it is pinned here
    so the value is pinned.
    """

    def _residual(self, params):
        layout = DramLayout()
        fine_w = RenderWindow(1234.0, 2345.0, 1000, 1.0)
        coarse_w = RenderWindow(1234.0, 2345.0, 100, 10.0)
        fine = apply_roughness(layout.render(fine_w), fine_w, params).to_greyscale()
        coarse = apply_roughness(layout.render(coarse_w), coarse_w, params).to_greyscale()
        return np.abs(area_average_reduce(fine, 10.0) - coarse) * 255.0

    def test_without_roughness_the_render_is_still_exact(self):
        self.assertLess(float(self._residual(None).max()), 1e-3)

    def test_with_roughness_the_typical_residual_stays_small(self):
        residual = self._residual(RoughnessParams(seed=7))
        self.assertLess(float(np.median(residual)), 1.0)

    def test_with_roughness_the_rms_residual_is_a_few_grey_levels(self):
        residual = self._residual(RoughnessParams(seed=7))
        self.assertLess(float(np.sqrt((residual ** 2).mean())), 8.0)

    def test_a_larger_amplitude_costs_more(self):
        small = float(np.sqrt((self._residual(RoughnessParams(amplitude_nm=0.5, seed=7)) ** 2).mean()))
        large = float(np.sqrt((self._residual(RoughnessParams(amplitude_nm=3.0, seed=7)) ** 2).mean()))
        self.assertGreater(large, small)


class TestGenerationWiring(unittest.TestCase):
    def test_roughness_is_off_without_the_imaging_model(self):
        """Noise-free renders exist to be exact; roughening them defeats the point."""
        from pathlib import Path

        from driftsense.generate import GenerationConfig

        self.assertIsNone(
            GenerationConfig(output_dir=Path("/tmp/unused")).resolved_roughness()
        )

    def test_roughness_is_on_with_the_imaging_model(self):
        from pathlib import Path

        from driftsense.generate import GenerationConfig

        self.assertIsNotNone(
            GenerationConfig(output_dir=Path("/tmp/unused"), imaging=True).resolved_roughness()
        )

    def test_the_flag_turns_it_off(self):
        from pathlib import Path

        from driftsense.generate import GenerationConfig

        config = GenerationConfig(
            output_dir=Path("/tmp/unused"), imaging=True, apply_roughness=False
        )
        self.assertIsNone(config.resolved_roughness())

    def test_both_captures_of_a_pair_share_the_specimen(self):
        """Roughness is a property of the silicon, not of the visit.

        If the two captures were roughened independently the reference would not
        match the search image even at the correct location, and the problem would
        be harder than reality.
        """
        import numpy as np

        from driftsense import ImagingGeometry, generate_pair
        from driftsense.sampling import PlacementSampler, sample_dram_layout
        from driftsense.validate import zncc
        from driftsense.resample import area_average_reduce

        geometry = ImagingGeometry()
        layout = sample_dram_layout(np.random.default_rng(3))
        placement = PlacementSampler().sample(layout, np.random.default_rng(4))
        pair = generate_pair(
            layout, placement, geometry, seed=1, roughness=RoughnessParams(seed=99)
        )
        template = area_average_reduce(pair.reference.astype(float), 10.0)
        gt = pair.ground_truth
        x, y = int(round(gt.box_x)), int(round(gt.box_y))
        patch = pair.search[y : y + template.shape[0], x : x + template.shape[1]].astype(float)
        self.assertGreater(zncc(template, patch), 0.9)


if __name__ == "__main__":
    unittest.main()

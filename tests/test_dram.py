"""DRAM layout tests, centred on the scale-consistency invariant."""

from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from driftsense.layouts import DramLayout, DramParams, RenderWindow
from driftsense.validate import block_reduce, zncc


def make_params(**overrides) -> DramParams:
    base = dict(
        feature_size_nm=35.0,
        cell_architecture="6F2",
        mat_size_nm=2600.0,
        strip_width_nm=320.0,
    )
    base.update(overrides)
    return DramParams(**base)


class TestDramParams(unittest.TestCase):
    def setUp(self) -> None:
        self.params = make_params()

    def test_six_f_squared_pitches_follow_the_cell_architecture(self):
        self.assertAlmostEqual(self.params.resolved_bitline_pitch_nm, 70.0)  # 2F
        self.assertAlmostEqual(self.params.resolved_wordline_pitch_nm, 105.0)  # 3F

    def test_eight_f_squared_uses_a_taller_cell(self):
        p = make_params(cell_architecture="8F2")
        self.assertAlmostEqual(p.resolved_wordline_pitch_nm, 140.0)  # 4F

    def test_explicit_pitches_override_the_architecture(self):
        p = make_params(bitline_pitch_nm=64.0, wordline_pitch_nm=96.0)
        self.assertAlmostEqual(p.resolved_bitline_pitch_nm, 64.0)
        self.assertAlmostEqual(p.resolved_wordline_pitch_nm, 96.0)

    def test_mat_period_is_array_plus_periphery(self):
        self.assertAlmostEqual(self.params.mat_period_nm, 2920.0)

    def test_positive_etch_bias_grows_features(self):
        widened = dataclasses.replace(self.params, linewidth_bias_nm=4.0)
        self.assertAlmostEqual(widened.bitline_width_nm, self.params.bitline_width_nm + 4.0)

    def test_etch_bias_cannot_exceed_the_pitch(self):
        absurd = dataclasses.replace(self.params, linewidth_bias_nm=10_000.0)
        self.assertAlmostEqual(absurd.bitline_width_nm, absurd.resolved_bitline_pitch_nm)

    def test_negative_bias_cannot_produce_a_negative_width(self):
        absurd = dataclasses.replace(self.params, linewidth_bias_nm=-10_000.0)
        self.assertAlmostEqual(absurd.bitline_width_nm, 0.0)

    def test_invalid_parameters_are_rejected(self):
        for kwargs in (
            {"feature_size_nm": 0.0},
            {"cell_architecture": "12F2"},
            {"mat_size_nm": -10.0},
            {"strip_width_nm": -1.0},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    make_params(**kwargs)

    def test_describe_round_trips_through_json_types(self):
        payload = DramLayout(self.params).describe()
        self.assertEqual(payload["architecture"], "dram")
        self.assertAlmostEqual(payload["mat_size_nm"], 2600.0)
        self.assertIn("bitline_width_nm", payload)


class TestRendering(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = DramLayout(make_params())

    def test_render_shape_and_range(self):
        render = self.layout.render(RenderWindow(0.0, 0.0, 256, 1.0))
        self.assertEqual(render.shape, (256, 256))
        for name, field in render.layers.items():
            with self.subTest(layer=name):
                self.assertEqual(field.dtype, np.float32)
                self.assertGreaterEqual(float(field.min()), 0.0)
                self.assertLessEqual(float(field.max()), 1.0)

    def test_greyscale_and_uint8_agree(self):
        render = self.layout.render(RenderWindow(0.0, 0.0, 128, 1.0))
        grey, quantised = render.to_greyscale(), render.to_uint8()
        self.assertEqual(quantised.dtype, np.uint8)
        self.assertLessEqual(float(np.max(np.abs(grey * 255.0 - quantised))), 0.5 + 1e-4)

    def test_layers_are_disjoint_and_sum_to_the_covered_fraction(self):
        """The invariant the exact composite depends on."""
        render = self.layout.render(RenderWindow(0.0, 0.0, 512, 1.0))
        render.check_disjoint()
        total = render.total_coverage()
        self.assertLessEqual(float(total.max()), 1.0 + 1e-5)
        self.assertGreaterEqual(float(total.min()), 0.0)

    def test_no_layer_has_negative_coverage(self):
        """Occlusion subtraction must never go the wrong way.

        Exercised at parameter extremes, where a wide contact on a narrow line
        would break containment if the contact clamp were missing.
        """
        for kwargs in (
            {"contact_size_ratio": 0.70, "bitline_width_ratio": 0.40, "feature_size_nm": 48.0},
            {"linewidth_bias_nm": -6.0},
            {"linewidth_bias_nm": 6.0},
            {"mat_pitch_variation": 0.30, "signature_seed": 99},
        ):
            with self.subTest(**kwargs):
                render = DramLayout(make_params(**kwargs)).render(
                    RenderWindow(0.0, 0.0, 400, 1.0)
                )
                render.check_disjoint()
                for name, field in render.layers.items():
                    self.assertGreaterEqual(float(field.min()), -1e-6, name)

    def test_periphery_strips_contain_no_array_structure(self):
        """A window placed entirely inside a strip must render as flat substrate."""
        p = self.layout.params
        start = p.mat_size_nm + 5.0
        render = self.layout.render(RenderWindow(start, start, 24, 1.0))
        np.testing.assert_allclose(render.layers["array_field"], 0.0, atol=1e-6)
        np.testing.assert_allclose(render.to_greyscale(), p.intensity_substrate, atol=1e-5)

    def test_lattice_period_appears_in_the_rendered_image(self):
        """The rendered bit-line period must match this mat's pitch.

        The expectation is the *per-mat* pitch rather than the nominal one, since
        each mat carries its own cell pitch; the window used here lies inside mat
        index 0.
        """
        row = self.layout.render(RenderWindow(100.0, 100.0, 700, 1.0)).to_greyscale()[0]
        spectrum = np.abs(np.fft.rfft(row - row.mean()))
        peak_bin = int(np.argmax(spectrum[1:]) + 1)
        measured_period = len(row) / peak_bin
        expected = self.layout.params.mat_pitch_nm("x", 0)
        self.assertAlmostEqual(measured_period, expected, delta=0.05 * expected)

    def test_rendering_is_deterministic(self):
        window = RenderWindow(321.0, 654.0, 200, 1.0)
        self.assertTrue(
            np.array_equal(
                self.layout.render(window).to_uint8(), self.layout.render(window).to_uint8()
            )
        )

    def test_with_params_returns_an_independent_copy(self):
        other = self.layout.with_params(feature_size_nm=20.0)
        self.assertAlmostEqual(other.params.feature_size_nm, 20.0)
        self.assertAlmostEqual(self.layout.params.feature_size_nm, 35.0)

    def test_invalid_windows_are_rejected(self):
        for kwargs in ({"size_px": 0}, {"pixel_size_nm": 0.0}):
            with self.subTest(**kwargs):
                args = {"origin_x_nm": 0.0, "origin_y_nm": 0.0, "size_px": 8, "pixel_size_nm": 1.0}
                args.update(kwargs)
                with self.assertRaises(ValueError):
                    RenderWindow(**args)


class TestScaleConsistency(unittest.TestCase):
    """The core invariant: one world, rendered at two magnifications, agrees."""

    def setUp(self) -> None:
        self.layout = DramLayout(make_params())

    def test_coarse_render_matches_block_reduced_fine_render(self):
        fine = self.layout.render(RenderWindow(0.0, 0.0, 1000, 1.0)).to_greyscale() * 255.0
        coarse = self.layout.render(RenderWindow(0.0, 0.0, 100, 10.0)).to_greyscale() * 255.0
        reduced = block_reduce(fine, 10.0)

        self.assertEqual(reduced.shape, coarse.shape)
        self.assertGreater(zncc(reduced, coarse), 0.99999)
        # With occlusion resolved in the layout model and coverage integrated in
        # closed form, this is exact up to float32 rounding -- not merely close.
        # The bound is three orders of magnitude below one 8-bit grey level.
        self.assertLess(float(np.max(np.abs(reduced - coarse))), 1e-3)

    def test_invariant_holds_at_a_non_aligned_origin(self):
        origin = 1234.0
        fine = self.layout.render(RenderWindow(origin, origin, 500, 1.0)).to_greyscale() * 255.0
        coarse = self.layout.render(RenderWindow(origin, origin, 50, 10.0)).to_greyscale() * 255.0
        self.assertLess(float(np.max(np.abs(block_reduce(fine, 10.0) - coarse))), 1e-3)

    def test_invariant_holds_under_mat_phase_and_etch_bias(self):
        layout = DramLayout(
            make_params(mat_phase_x_nm=137.0, mat_phase_y_nm=911.0, linewidth_bias_nm=-1.5)
        )
        fine = layout.render(RenderWindow(0.0, 0.0, 800, 1.0)).to_greyscale() * 255.0
        coarse = layout.render(RenderWindow(0.0, 0.0, 80, 10.0)).to_greyscale() * 255.0
        self.assertLess(float(np.max(np.abs(block_reduce(fine, 10.0) - coarse))), 1e-3)

    def test_window_origin_shifts_content_by_the_expected_amount(self):
        base = self.layout.render(RenderWindow(0.0, 0.0, 200, 1.0)).to_greyscale()
        shifted = self.layout.render(RenderWindow(25.0, 0.0, 200, 1.0)).to_greyscale()
        np.testing.assert_allclose(base[:, 25:], shifted[:, :-25], atol=1e-5)


class TestBoundaryCoordinates(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = DramLayout(make_params())

    def test_reports_both_edges_of_each_mat(self):
        edges = self.layout.boundary_coordinates_nm("x", 0.0, 10_000.0)
        self.assertGreaterEqual(len(edges), 6)
        period, size = self.layout.params.mat_period_nm, self.layout.params.mat_size_nm
        for edge in edges:
            offset = edge % period
            with self.subTest(edge=edge):
                self.assertTrue(
                    abs(offset) < 1e-6 or abs(offset - size) < 1e-6,
                    f"edge {edge} is not on a mat boundary (offset {offset})",
                )

    def test_edges_are_sorted_and_within_the_requested_span(self):
        edges = self.layout.boundary_coordinates_nm("y", 1000.0, 6000.0)
        self.assertEqual(list(edges), sorted(edges))
        self.assertTrue(all(1000.0 <= edge <= 6000.0 for edge in edges))

    def test_unknown_axis_is_rejected(self):
        with self.assertRaises(ValueError):
            self.layout.boundary_coordinates_nm("z", 0.0, 1.0)

    def test_phase_offsets_the_mat_grid(self):
        shifted = DramLayout(make_params(mat_phase_x_nm=500.0))
        edges = shifted.boundary_coordinates_nm("x", 0.0, 3000.0)
        self.assertTrue(any(abs(edge - 500.0) < 1e-6 for edge in edges))


if __name__ == "__main__":
    unittest.main()

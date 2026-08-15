"""FinFET layout: geometry, occlusion, scale consistency and anchors."""

from __future__ import annotations

import unittest

import numpy as np

from driftsense.layouts import (
    FinfetLayout,
    FinfetParams,
    RenderWindow,
    available_architectures,
    get_layout_class,
)
from driftsense.layouts.base import COVERAGE_TOLERANCE
from driftsense.resample import area_average_reduce
from driftsense.sampling import FinfetParamRanges, sample_finfet_layout


def window(size_px=200, pixel_nm=5.0, x=1000.0, y=1000.0):
    return RenderWindow(x, y, size_px, pixel_nm)


class TestParams(unittest.TestCase):
    def test_defaults_are_valid(self):
        FinfetParams()

    def test_rejects_a_non_positive_fin_pitch(self):
        with self.assertRaises(ValueError):
            FinfetParams(fin_pitch_nm=0.0)

    def test_rejects_fins_wider_than_their_pitch(self):
        """Fins that touch are a sheet, not a fin grating."""
        with self.assertRaises(ValueError):
            FinfetParams(fin_pitch_nm=30.0, fin_width_nm=40.0)

    def test_rejects_gates_longer_than_their_pitch(self):
        with self.assertRaises(ValueError):
            FinfetParams(gate_pitch_nm=100.0, gate_length_nm=120.0)

    def test_rejects_a_negative_clearance(self):
        with self.assertRaises(ValueError):
            FinfetParams(contact_gate_clearance_nm=-1.0)

    def test_rejects_block_variation_at_one(self):
        with self.assertRaises(ValueError):
            FinfetParams(block_pitch_variation=1.0)

    def test_contact_is_clamped_away_from_the_gates(self):
        """The disjointness the separable occlusion arithmetic depends on."""
        p = FinfetParams(
            gate_pitch_nm=100.0, gate_length_nm=30.0,
            contact_length_nm=500.0, contact_gate_clearance_nm=5.0,
        )
        # gap is 70, minus 2x5 clearance leaves 60
        self.assertAlmostEqual(p.resolved_contact_length_nm, 60.0, places=6)

    def test_a_clearance_wider_than_the_gap_gives_no_contact(self):
        p = FinfetParams(
            gate_pitch_nm=100.0, gate_length_nm=90.0, contact_gate_clearance_nm=40.0
        )
        self.assertEqual(p.resolved_contact_length_nm, 0.0)

    def test_etch_bias_grows_the_fin(self):
        plain = FinfetParams().resolved_fin_width_nm
        biased = FinfetParams(linewidth_bias_nm=3.0).resolved_fin_width_nm
        self.assertAlmostEqual(biased - plain, 3.0, places=6)

    def test_etch_bias_cannot_exceed_the_pitch(self):
        p = FinfetParams(fin_pitch_nm=30.0, fin_width_nm=10.0, linewidth_bias_nm=500.0)
        self.assertLessEqual(p.resolved_fin_width_nm, 30.0)

    def test_block_pitch_varies_between_blocks(self):
        p = FinfetParams(signature_seed=7)
        pitches = {round(p.block_pitch_nm("x", i), 6) for i in range(6)}
        self.assertGreater(len(pitches), 1)

    def test_block_pitch_is_deterministic(self):
        p = FinfetParams(signature_seed=7)
        self.assertEqual(p.block_pitch_nm("x", 3), p.block_pitch_nm("x", 3))

    def test_zero_variation_makes_every_block_identical(self):
        """Retained deliberately: it is the genuinely unsolvable case."""
        p = FinfetParams(block_pitch_variation=0.0)
        self.assertEqual(p.block_pitch_nm("y", 1), p.block_pitch_nm("y", 9))

    def test_as_dict_reports_resolved_values(self):
        payload = FinfetParams().as_dict()
        self.assertIn("fin_pitch_nm", payload)
        self.assertIn("contact_length_nm", payload)
        self.assertNotIn("intensity_gate", payload)


class TestRender(unittest.TestCase):
    def setUp(self):
        self.layout = FinfetLayout()

    def test_layers_are_the_expected_four(self):
        render = self.layout.render(window())
        self.assertEqual(set(render.layers), {"field", "fin", "contact", "gate"})

    def test_layers_are_disjoint(self):
        self.layout.render(window()).check_disjoint()

    def test_coverage_never_exceeds_one(self):
        total = self.layout.render(window()).total_coverage()
        self.assertLessEqual(float(total.max()), 1.0 + COVERAGE_TOLERANCE)

    def test_no_layer_is_negative(self):
        for coverage in self.layout.render(window()).layers.values():
            self.assertGreaterEqual(float(coverage.min()), -COVERAGE_TOLERANCE)

    def test_every_layer_is_actually_present(self):
        """A layer that never appears would mean the occlusion has eaten it."""
        for name, coverage in self.layout.render(window(300, 4.0)).layers.items():
            self.assertGreater(float(coverage.max()), 0.0, f"layer {name!r} is empty")

    def test_greyscale_is_bounded(self):
        image = self.layout.render(window()).to_greyscale()
        self.assertGreaterEqual(float(image.min()), 0.0)
        self.assertLessEqual(float(image.max()), 1.0)

    def test_uint8_render_has_the_right_dtype_and_shape(self):
        image = self.layout.render(window(128, 8.0)).to_uint8()
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(image.shape, (128, 128))

    def test_rendering_is_deterministic(self):
        a = self.layout.render(window()).to_greyscale()
        b = self.layout.render(window()).to_greyscale()
        np.testing.assert_array_equal(a, b)

    def test_the_same_world_region_renders_the_same_at_two_pixel_sizes(self):
        """The property the whole dataset rests on.

        The reference is captured at 1 nm/px and the search at 10 nm/px. If those
        two disagreed about the same physical region, the ground truth would be
        arithmetic fiction.
        """
        fine = self.layout.render(RenderWindow(1234.0, 2345.0, 500, 1.0)).to_greyscale()
        coarse = self.layout.render(RenderWindow(1234.0, 2345.0, 50, 10.0)).to_greyscale()
        residual = float(np.abs(area_average_reduce(fine, 10.0) - coarse).max()) * 255.0
        self.assertLess(residual, 1e-3, f"residual {residual:.2e} grey levels")

    def test_a_translated_window_renders_translated_content(self):
        pixel = 5.0
        a = self.layout.render(RenderWindow(1000.0, 1000.0, 120, pixel)).to_greyscale()
        b = self.layout.render(RenderWindow(1000.0 + 20 * pixel, 1000.0, 120, pixel)).to_greyscale()
        np.testing.assert_allclose(a[:, 20:], b[:, :-20], atol=1e-6)

    def test_gates_span_the_full_block_width(self):
        """A gate crosses every fin, so its row is uniform across the block."""
        render = self.layout.render(RenderWindow(200.0, 200.0, 160, 4.0))
        gate = render.layers["gate"]
        strongest = int(np.argmax(gate.sum(axis=1)))
        row = gate[strongest]
        interior = row[10:-10]
        self.assertGreater(float(interior.min()), 0.0)

    def test_contacts_only_appear_on_fins(self):
        """Contact coverage must be nested inside fin columns, never off them."""
        render = self.layout.render(RenderWindow(200.0, 200.0, 160, 4.0))
        contact_columns = render.layers["contact"].sum(axis=0)
        fin_columns = (
            render.layers["fin"].sum(axis=0) + render.layers["contact"].sum(axis=0)
        )
        self.assertTrue(np.all(contact_columns <= fin_columns + 1e-6))

    def test_a_zero_break_still_renders(self):
        layout = FinfetLayout(FinfetParams(break_width_nm=0.0))
        layout.render(window()).check_disjoint()

    def test_extreme_bias_still_renders_disjoint_layers(self):
        layout = FinfetLayout(FinfetParams(linewidth_bias_nm=25.0))
        layout.render(window()).check_disjoint()

    def test_a_single_pixel_window_is_allowed(self):
        self.assertEqual(self.layout.render(RenderWindow(0.0, 0.0, 1, 10.0)).shape, (1, 1))


class TestAnchors(unittest.TestCase):
    def setUp(self):
        self.layout = FinfetLayout(FinfetParams(block_size_nm=1800.0, break_width_nm=200.0))

    def test_reports_block_edges(self):
        edges = self.layout.boundary_coordinates_nm("x", 0.0, 4200.0)
        self.assertIn(1800.0, edges)
        self.assertIn(2000.0, edges)

    def test_edges_are_sorted(self):
        edges = self.layout.boundary_coordinates_nm("y", 0.0, 8000.0)
        self.assertEqual(list(edges), sorted(edges))

    def test_edges_lie_inside_the_requested_span(self):
        for edge in self.layout.boundary_coordinates_nm("x", 500.0, 3000.0):
            self.assertTrue(500.0 <= edge <= 3000.0)

    def test_an_empty_span_reports_nothing(self):
        self.assertEqual(self.layout.boundary_coordinates_nm("x", 100.0, 200.0), ())

    def test_rejects_an_unknown_axis(self):
        with self.assertRaises(ValueError):
            self.layout.boundary_coordinates_nm("z", 0.0, 100.0)

    def test_the_phase_shifts_the_edges(self):
        shifted = FinfetLayout(
            FinfetParams(block_size_nm=1800.0, break_width_nm=200.0, block_phase_x_nm=50.0)
        )
        self.assertIn(1850.0, shifted.boundary_coordinates_nm("x", 0.0, 4200.0))


class TestRegistryAndSampling(unittest.TestCase):
    def test_finfet_is_registered(self):
        self.assertIn("finfet", available_architectures())

    def test_the_registry_returns_the_class(self):
        self.assertIs(get_layout_class("finfet"), FinfetLayout)

    def test_sampling_is_deterministic(self):
        a = sample_finfet_layout(np.random.default_rng(5)).describe()
        b = sample_finfet_layout(np.random.default_rng(5)).describe()
        self.assertEqual(a, b)

    def test_sampled_layouts_differ(self):
        a = sample_finfet_layout(np.random.default_rng(1)).describe()
        b = sample_finfet_layout(np.random.default_rng(2)).describe()
        self.assertNotEqual(a, b)

    def test_sampled_layouts_render_disjoint_layers(self):
        for seed in range(6):
            layout = sample_finfet_layout(np.random.default_rng(seed))
            layout.render(window(96, 10.0)).check_disjoint()

    def test_sampled_pitches_stay_resolvable_at_ten_nanometre_pixels(self):
        """Below about 3 px per pitch the grating is not sampled at all.

        The range deliberately stops short of the finest published ground rules;
        see ``FinfetParamRanges``.
        """
        for seed in range(20):
            layout = sample_finfet_layout(np.random.default_rng(seed))
            self.assertGreaterEqual(layout.params.fin_pitch_nm / 10.0, 3.0)

    def test_rejects_an_inverted_range(self):
        with self.assertRaises(ValueError):
            FinfetParamRanges(fin_pitch_nm=(50.0, 30.0))

    def test_describe_is_json_serialisable(self):
        import json

        json.dumps(sample_finfet_layout(np.random.default_rng(3)).describe())

    def test_with_params_replaces_selected_fields(self):
        layout = FinfetLayout().with_params(fin_pitch_nm=60.0)
        self.assertEqual(layout.params.fin_pitch_nm, 60.0)
        self.assertEqual(layout.params.gate_pitch_nm, FinfetParams().gate_pitch_nm)


class TestFinfetIsTheHarderArchitecture(unittest.TestCase):
    """Recording why FinFET needs its own validation floor.

    Applying DRAM's 0.55 ground-truth correlation threshold rejected genuine
    FinFET pairs. It is not a coordinate bug: the same layouts validate at 0.95
    with imaging off. A fin grating is finer relative to the beam, so less of the
    reference survives into the wide-search capture.
    """

    def test_the_imaging_floor_is_lower_for_finfet(self):
        from driftsense.validate import ValidationThresholds

        self.assertLess(
            ValidationThresholds.for_imaging("finfet").min_zncc,
            ValidationThresholds.for_imaging("dram").min_zncc,
        )

    def test_an_unknown_architecture_gets_the_default_floor(self):
        from driftsense.validate import ValidationThresholds

        self.assertEqual(
            ValidationThresholds.for_imaging("something_else").min_zncc,
            ValidationThresholds.for_imaging("dram").min_zncc,
        )

    def test_the_floor_still_leaves_room_to_catch_a_coordinate_bug(self):
        """A displaced ground truth scores near zero, far under any floor here."""
        from driftsense.validate import ValidationThresholds

        self.assertGreater(ValidationThresholds.for_imaging("finfet").min_zncc, 0.2)


if __name__ == "__main__":
    unittest.main()

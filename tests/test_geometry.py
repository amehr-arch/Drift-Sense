"""Geometry and ground-truth tests, including a regression against organiser data."""

from __future__ import annotations

import unittest

from driftsense.geometry import (
    ImagingGeometry,
    Placement,
    ground_truth_for,
    search_px_to_world_nm,
    world_nm_to_reference_px,
    world_nm_to_search_px,
)


class TestImagingGeometry(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = ImagingGeometry()

    def test_problem_statement_defaults(self):
        g = self.geometry
        self.assertEqual(g.image_size_px, 1000)
        self.assertAlmostEqual(g.reference_pixel_size_nm, 1.0)
        self.assertAlmostEqual(g.search_pixel_size_nm, 10.0)
        self.assertAlmostEqual(g.reference_fov_nm, 1000.0)
        self.assertAlmostEqual(g.search_fov_nm, 10000.0)

    def test_template_size_is_one_hundred_pixels(self):
        """The reference must occupy a 100x100 patch of the search image."""
        self.assertAlmostEqual(self.geometry.template_size_px, 100.0)

    def test_max_origin_keeps_the_window_inside_the_field(self):
        self.assertAlmostEqual(self.geometry.max_origin_nm, 9000.0)

    def test_invalid_configurations_are_rejected(self):
        for kwargs in (
            {"image_size_px": 0},
            {"reference_pixel_size_nm": 0.0},
            {"reference_pixel_size_nm": -1.0},
            {"zoom_ratio": 1.0},
            {"zoom_ratio": 0.5},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    ImagingGeometry(**kwargs)

    def test_as_dict_is_complete(self):
        payload = self.geometry.as_dict()
        for key in ("image_size_px", "search_pixel_size_nm", "template_size_px"):
            self.assertIn(key, payload)


class TestCoordinateTransforms(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = ImagingGeometry()

    def test_world_to_search_round_trip(self):
        for nm in (0.0, 1.0, 2491.0, 9999.9):
            with self.subTest(nm=nm):
                px = world_nm_to_search_px(nm, self.geometry)
                self.assertAlmostEqual(search_px_to_world_nm(px, self.geometry), nm, places=9)

    def test_search_pixel_spans_ten_nanometres(self):
        self.assertAlmostEqual(world_nm_to_search_px(10.0, self.geometry), 1.0)

    def test_reference_pixels_are_relative_to_the_crop_origin(self):
        self.assertAlmostEqual(world_nm_to_reference_px(2500.0, 2491.0, self.geometry), 9.0)


class TestGroundTruth(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = ImagingGeometry()

    def test_matches_organiser_sample_metadata(self):
        """Regression against the metadata released with the organisers' sample pair.

        The published block gives gt_x = 299.1, gt_y = 618.5 and
        gt_box = [249.1, 568.5, 100, 100]. Reproducing those exactly confirms our
        ground truth is expressed in the frame the submission will be scored in.
        """
        gt = ground_truth_for(Placement(2491.0, 5685.0), self.geometry)
        self.assertAlmostEqual(gt.x, 299.1, places=9)
        self.assertAlmostEqual(gt.y, 618.5, places=9)
        self.assertAlmostEqual(gt.box_x, 249.1, places=9)
        self.assertAlmostEqual(gt.box_y, 568.5, places=9)
        self.assertAlmostEqual(gt.box_w, 100.0, places=9)
        self.assertAlmostEqual(gt.box_h, 100.0, places=9)

    def test_centre_is_the_box_centre(self):
        gt = ground_truth_for(Placement(1234.0, 5678.0), self.geometry)
        self.assertAlmostEqual(gt.x, gt.box_x + gt.box_w / 2)
        self.assertAlmostEqual(gt.y, gt.box_y + gt.box_h / 2)

    def test_origin_placement_sits_at_the_frame_corner(self):
        gt = ground_truth_for(Placement(0.0, 0.0), self.geometry)
        self.assertAlmostEqual(gt.box_x, 0.0)
        self.assertAlmostEqual(gt.box_y, 0.0)
        self.assertAlmostEqual(gt.x, 50.0)
        self.assertAlmostEqual(gt.y, 50.0)

    def test_extreme_placement_touches_the_far_edge(self):
        gt = ground_truth_for(Placement(9000.0, 9000.0), self.geometry)
        self.assertAlmostEqual(gt.box_x + gt.box_w, float(self.geometry.image_size_px))
        self.assertAlmostEqual(gt.x, 950.0)
        self.assertAlmostEqual(gt.y, 950.0)

    def test_no_half_pixel_offset_is_introduced(self):
        """A shift of one search pixel in the world must move the answer by exactly one."""
        a = ground_truth_for(Placement(1000.0, 1000.0), self.geometry)
        b = ground_truth_for(Placement(1010.0, 1000.0), self.geometry)
        self.assertAlmostEqual(b.x - a.x, 1.0, places=9)
        self.assertAlmostEqual(b.y, a.y, places=9)

    def test_sub_nanometre_shifts_survive(self):
        a = ground_truth_for(Placement(1000.0, 0.0), self.geometry)
        b = ground_truth_for(Placement(1000.5, 0.0), self.geometry)
        self.assertAlmostEqual(b.x - a.x, 0.05, places=9)

    def test_serialisation_uses_the_organiser_key_names(self):
        payload = ground_truth_for(Placement(2491.0, 5685.0), self.geometry).as_dict()
        self.assertEqual(set(payload), {"gt_x", "gt_y", "gt_box"})
        for value, expected in zip(payload["gt_box"], [249.1, 568.5, 100.0, 100.0]):
            self.assertAlmostEqual(value, expected, places=9)


if __name__ == "__main__":
    unittest.main()

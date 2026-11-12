"""Unit tests for sun geometry and cloud shadow projection."""

import unittest
import numpy as np

from sat_pipeline.physics.sun_geometry import (
    calculate_shadow_offset_m,
    calculate_shadow_shift_pixels,
    estimate_cloud_height,
    project_cloud_mask,
    project_cloud_shadow,
)


class TestSunGeometry(unittest.TestCase):
    """Test sun geometry mathematical formulas."""

    def test_shadow_offset_formula(self):
        # At 45 degrees elevation, tan(45) = 1.0 -> offset == height
        offset = calculate_shadow_offset_m(cloud_height_m=1000.0, sun_elevation_deg=45.0)
        self.assertAlmostEqual(offset, 1000.0, places=4)

        # At 30 degrees, tan(30) = 1/sqrt(3) -> offset = height * sqrt(3) ~ 1732.05m
        offset_30 = calculate_shadow_offset_m(cloud_height_m=1000.0, sun_elevation_deg=30.0)
        self.assertAlmostEqual(offset_30, 1000.0 * np.sqrt(3.0), places=2)

        # Invalid angles raise ValueError
        with self.assertRaises(ValueError):
            calculate_shadow_offset_m(1000.0, 0.0)
        with self.assertRaises(ValueError):
            calculate_shadow_offset_m(1000.0, -10.0)

    def test_shadow_shift_directions(self):
        # Case 1: Sun in South (azimuth = 180). Shadow points North (azimuth + 180 = 360/0 deg).
        # North is negative row direction.
        row_shift, col_shift = calculate_shadow_shift_pixels(
            offset_m=100.0,
            sun_azimuth_deg=180.0,
            pixel_size_m=10.0,
            north_is_up=True,
        )
        self.assertAlmostEqual(row_shift, -10.0, places=4)
        self.assertAlmostEqual(col_shift, 0.0, places=4)

        # Case 2: Sun in East (azimuth = 90). Shadow points West (azimuth + 180 = 270 deg).
        # West is negative col direction.
        row_shift, col_shift = calculate_shadow_shift_pixels(
            offset_m=100.0,
            sun_azimuth_deg=90.0,
            pixel_size_m=10.0,
            north_is_up=True,
        )
        self.assertAlmostEqual(row_shift, 0.0, places=4)
        self.assertAlmostEqual(col_shift, -10.0, places=4)

    def test_cloud_height_search_recovery(self):
        # Create a synthetic cloud at center (50, 50) of a 100x100 grid
        h, w = 100, 100
        cloud_mask = np.zeros((h, w), dtype=np.uint8)
        cloud_mask[45:55, 45:55] = 1

        # Ground truth cloud height = 1000m
        # Sun: elevation 45 deg, azimuth 180 deg (South)
        # Expected shadow shift: offset = 1000m / tan(45) = 1000m -> 100 pixels North (row -100)
        # Let's use elevation 45 deg, azimuth 135 deg (SE), height 500m -> offset 500m -> 50 pixels NW
        true_height = 400.0
        sun_elev = 45.0
        sun_azim = 180.0  # shadow points North (row -40)

        # Place dark pixels at the exact ground truth projected shadow location
        gt_shadow = project_cloud_shadow(
            cloud_mask=cloud_mask,
            cloud_height_m=true_height,
            sun_elevation_deg=sun_elev,
            sun_azimuth_deg=sun_azim,
            pixel_size_m=10.0,
        )

        brightness = np.full((h, w), 0.35, dtype=np.float32)  # clear background
        brightness[gt_shadow > 0] = 0.08                     # dark shadow pixels (< 0.15)
        brightness[cloud_mask > 0] = 0.70                    # bright cloud pixels

        best_h, pred_shadow, overlap_scores = estimate_cloud_height(
            cloud_mask=cloud_mask,
            brightness=brightness,
            sun_elevation_deg=sun_elev,
            sun_azimuth_deg=sun_azim,
            pixel_size_m=10.0,
            min_height_m=200.0,
            max_height_m=2000.0,
            step_m=100.0,
            dark_threshold=0.15,
        )

        self.assertEqual(best_h, true_height)
        self.assertGreater(overlap_scores[best_h], 0)
        self.assertGreaterEqual(overlap_scores[best_h], max(overlap_scores.values()))


if __name__ == "__main__":
    unittest.main()
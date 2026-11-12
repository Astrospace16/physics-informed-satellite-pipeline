"""Unit tests for DEM gradient and hillshade illumination."""

import unittest
import numpy as np

from sat_pipeline.physics.hillshade import (
    compute_dem_gradients,
    compute_hillshade_mask,
    compute_slope_and_aspect,
    compute_solar_illumination,
)


class TestHillshade(unittest.TestCase):
    """Test DEM gradient, slope, aspect, and hillshade calculations."""

    def test_flat_surface(self):
        # Flat surface (100m elevation everywhere)
        dem = np.full((50, 50), 100.0, dtype=np.float32)
        dz_dx, dz_dy = compute_dem_gradients(dem, pixel_size_m=10.0)
        slope, aspect = compute_slope_and_aspect(dz_dx, dz_dy)

        # Gradient and slope should be zero everywhere
        np.testing.assert_allclose(dz_dx, 0.0, atol=1e-5)
        np.testing.assert_allclose(dz_dy, 0.0, atol=1e-5)
        np.testing.assert_allclose(slope, 0.0, atol=1e-5)

        # Solar illumination on flat surface equals cos(zenith) = sin(elevation)
        illum = compute_solar_illumination(slope, aspect, sun_elevation_deg=45.0, sun_azimuth_deg=180.0)
        expected_illum = np.sin(np.radians(45.0))
        np.testing.assert_allclose(illum, expected_illum, atol=1e-4)

    def test_sloped_surface_illumination(self):
        # Create an East-facing slope: elevation increases toward West (decreasing x)
        # dz/dx is negative (lower toward East)
        y, x = np.mgrid[0:50, 0:50]
        dem = 500.0 - (x * 10.0).astype(np.float32)  # 10m drop per 10m pixel = 45 degree slope

        dz_dx, dz_dy = compute_dem_gradients(dem, pixel_size_m=10.0)
        slope, aspect = compute_slope_and_aspect(dz_dx, dz_dy)

        # Sun in the East (azimuth = 90 deg) directly illuminates East-facing slope
        illum_facing = compute_solar_illumination(slope, aspect, sun_elevation_deg=30.0, sun_azimuth_deg=90.0)

        # Sun in the West (azimuth = 270 deg) is behind the slope (self-shadowing)
        illum_away = compute_solar_illumination(slope, aspect, sun_elevation_deg=30.0, sun_azimuth_deg=270.0)

        # Facing illumination should be much higher than away illumination
        self.assertGreater(np.mean(illum_facing[5:-5, 5:-5]), np.mean(illum_away[5:-5, 5:-5]))

        # Away illumination should produce a hillshade mask
        mask_away, _, _, _ = compute_hillshade_mask(
            dem=dem,
            sun_elevation_deg=30.0,
            sun_azimuth_deg=270.0,
            pixel_size_m=10.0,
            illumination_threshold=0.15,
        )
        self.assertTrue(np.any(mask_away[5:-5, 5:-5] > 0))


if __name__ == "__main__":
    unittest.main()
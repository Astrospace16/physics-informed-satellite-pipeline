"""Unit tests for spectral indices and spectral detection checks."""

import unittest
import numpy as np

from sat_pipeline.physics.spectral import (
    compute_brightness,
    compute_ndvi,
    compute_ndsi,
    compute_ndwi,
    detect_spectral_clouds,
    detect_spectral_snow,
    detect_spectral_water,
)


class TestSpectral(unittest.TestCase):
    """Test spectral index formulas and threshold conditions."""

    def test_ndsi_snow_rule(self):
        # Snow: Green (B3) = 0.8, SWIR (B11) = 0.2
        # NDSI = (0.8 - 0.2) / (0.8 + 0.2) = 0.6 / 1.0 = 0.6 > 0.4 -> Snow
        b3_snow = np.array([[0.8]], dtype=np.float32)
        b11_snow = np.array([[0.2]], dtype=np.float32)
        ndsi = compute_ndsi(b3_snow, b11_snow)
        self.assertAlmostEqual(float(ndsi[0, 0]), 0.6, places=4)

        mask, _ = detect_spectral_snow(b3_snow, b11_snow, ndsi_threshold=0.40)
        self.assertEqual(int(mask[0, 0]), 1)

        # Cloud: Green (B3) = 0.7, SWIR (B11) = 0.6
        # NDSI = (0.7 - 0.6) / (0.7 + 0.6) = 0.1 / 1.3 = 0.0769 <= 0.4 -> NOT Snow
        b3_cloud = np.array([[0.7]], dtype=np.float32)
        b11_cloud = np.array([[0.6]], dtype=np.float32)
        mask_cloud, ndsi_cloud = detect_spectral_snow(b3_cloud, b11_cloud, ndsi_threshold=0.40)
        self.assertEqual(int(mask_cloud[0, 0]), 0)
        self.assertLess(float(ndsi_cloud[0, 0]), 0.40)

    def test_ndwi_water_rule(self):
        # Water: Green (B3) = 0.15, NIR (B8) = 0.03
        # NDWI = (0.15 - 0.03) / (0.15 + 0.03) = 0.12 / 0.18 = 0.666 > 0.3 AND B8 < 0.1 -> Water
        b3_water = np.array([[0.15]], dtype=np.float32)
        b8_water = np.array([[0.03]], dtype=np.float32)
        ndwi = compute_ndwi(b3_water, b8_water)
        self.assertAlmostEqual(float(ndwi[0, 0]), 0.12 / 0.18, places=4)

        mask, _ = detect_spectral_water(b3_water, b8_water, ndwi_threshold=0.30, b8_threshold=0.10)
        self.assertEqual(int(mask[0, 0]), 1)

        # Vegetation: Green (B3) = 0.08, NIR (B8) = 0.40 -> NDWI negative
        b3_veg = np.array([[0.08]], dtype=np.float32)
        b8_veg = np.array([[0.40]], dtype=np.float32)
        mask_veg, _ = detect_spectral_water(b3_veg, b8_veg, ndwi_threshold=0.30, b8_threshold=0.10)
        self.assertEqual(int(mask_veg[0, 0]), 0)

    def test_cirrus_cloud_rule(self):
        # Cirrus: B10 > 0.03 combined with high visible brightness -> cloud
        b2 = np.array([[0.25]], dtype=np.float32)
        b3 = np.array([[0.25]], dtype=np.float32)
        b4 = np.array([[0.25]], dtype=np.float32)
        b8 = np.array([[0.28]], dtype=np.float32)
        b10_cirrus = np.array([[0.045]], dtype=np.float32)  # > 0.03
        b11 = np.array([[0.18]], dtype=np.float32)

        cloud_mask, cirrus_mask, brightness = detect_spectral_clouds(
            b2, b3, b4, b8, b10_cirrus, b11, cirrus_b10_threshold=0.03, visible_brightness_threshold=0.20
        )
        self.assertEqual(int(cirrus_mask[0, 0]), 1)
        self.assertEqual(int(cloud_mask[0, 0]), 1)

        # Clear air without cirrus (B10 = 0.005)
        b10_clear = np.array([[0.005]], dtype=np.float32)
        _, cirrus_mask_clear, _ = detect_spectral_clouds(
            b2, b3, b4, b8, b10_clear, b11, cirrus_b10_threshold=0.03, visible_brightness_threshold=0.20
        )
        self.assertEqual(int(cirrus_mask_clear[0, 0]), 0)


if __name__ == "__main__":
    unittest.main()
"""Unit tests for radiative transfer, sensor noise filtering, seasonal phenology, and flood/fire/crop land_change."""

import unittest
import numpy as np

from sat_pipeline.core.constants import (
    REASON_LAND_CHANGE_CROP_DAMAGE,
    REASON_LAND_CHANGE_FIRE,
    REASON_LAND_CHANGE_FLOOD,
    Label,
    LandChangeSubtype,
)
from sat_pipeline.core.types import PipelineConfig, TileMetadata
from sat_pipeline.physics.radiative_transfer import (
    apply_c_correction_topography,
    apply_dark_object_subtraction,
    apply_solar_zenith_correction,
    detect_and_filter_sensor_noise,
    run_radiative_transfer_pipeline,
)
from sat_pipeline.physics.spectral import (
    compute_bsi,
    compute_nbr,
    compute_ndmi,
    compute_seasonal_phenology,
    detect_land_change_disturbances,
)
from sat_pipeline.pipeline import SatellitePipeline


class TestRadiativeAndDisturbances(unittest.TestCase):
    """Test radiative transfer, sensor noise filtering, seasonal modulation, and flood/fire/crop physics."""

    def test_radiative_transfer_functions_active(self):
        # 1. Dark object subtraction removes excess atmospheric haze floor
        hazy_band = np.full((16, 16), 0.04, dtype=np.float32)
        hazy_band[8:, 8:] = 0.25
        dos_band = apply_dark_object_subtraction(hazy_band, percentile=1.0)
        self.assertLess(float(np.min(dos_band)), float(np.min(hazy_band)))

        # 2. Solar zenith normalization boosts low-sun scenes relative to 45 deg reference
        low_sun = apply_solar_zenith_correction(
            np.full((4, 4), 0.20, dtype=np.float32),
            sun_elevation_deg=30.0,
            reference_elevation_deg=45.0,
        )
        self.assertGreater(float(low_sun[0, 0]), 0.20)

        # 3. Topographic C-correction adjusts moderately illuminated slopes while preserving deep hillshade (<0.10)
        illum = np.array([[0.05, 0.35]], dtype=np.float32)
        slope = np.array([[0.40, 0.40]], dtype=np.float32)
        band = np.array([[0.08, 0.20]], dtype=np.float32)
        topo = apply_c_correction_topography(
            band, illum, sun_elevation_deg=45.0, slope_rad=slope, hillshade_illumination_threshold=0.10
        )
        # Deep hillshade pixel (illum 0.05 < 0.10) is preserved untouched
        self.assertAlmostEqual(float(topo[0, 0]), 0.08, places=5)
        # Moderately shaded slope pixel (illum 0.35) is normalized upward
        self.assertGreater(float(topo[0, 1]), 0.20)

    def test_sensor_noise_despeckling(self):
        b2 = np.full((16, 16), 0.10, dtype=np.float32)
        # Inject single-pixel impulse spike
        b2[8, 8] = 0.95
        bands = {"B2": b2}
        filtered, noise_mask = detect_and_filter_sensor_noise(bands, zscore_threshold=3.5)
        self.assertEqual(int(noise_mask[8, 8]), 1)
        self.assertAlmostEqual(float(filtered["B2"][8, 8]), 0.10, places=4)

    def test_seasonal_phenology_modulation(self):
        # Summer solstice (DOY 172) in Northern Hemisphere has higher expected NDVI than Winter (DOY 15)
        ndvi_summer, _ = compute_seasonal_phenology(day_of_year=172, latitude_deg=35.0)
        ndvi_winter, _ = compute_seasonal_phenology(day_of_year=15, latitude_deg=35.0)
        self.assertGreater(ndvi_summer, 0.55)
        self.assertLess(ndvi_winter, 0.30)

    def test_flood_fire_crop_damage_detection_in_pipeline(self):
        # Build a 30x30 test tile with 3 distinct disturbance zones:
        # Row 2..8: Flood inundation
        # Row 12..18: Fire burn scar
        # Row 22..28: Crop damage / severe senescence
        h, w = 30, 30
        b2 = np.full((h, w), 0.05, dtype=np.float32)
        b3 = np.full((h, w), 0.08, dtype=np.float32)
        b4 = np.full((h, w), 0.06, dtype=np.float32)
        b8 = np.full((h, w), 0.40, dtype=np.float32)  # healthy vegetation background
        b11 = np.full((h, w), 0.15, dtype=np.float32)
        b10 = np.full((h, w), 0.005, dtype=np.float32)

        # Zone 1: Flood inundation (rows 2..7, cols 2..7)
        b2[2:7, 2:7] = 0.08
        b3[2:7, 2:7] = 0.11
        b4[2:7, 2:7] = 0.07
        b8[2:7, 2:7] = 0.075
        b11[2:7, 2:7] = 0.04

        # Zone 2: Fire burn scar (rows 12..17, cols 12..17)
        b2[12:17, 12:17] = 0.05
        b3[12:17, 12:17] = 0.05
        b4[12:17, 12:17] = 0.06
        b8[12:17, 12:17] = 0.06
        b11[12:17, 12:17] = 0.16

        # Zone 3: Crop damage (rows 22..27, cols 22..27)
        b2[22:27, 22:27] = 0.08
        b3[22:27, 22:27] = 0.10
        b4[22:27, 22:27] = 0.15
        b8[22:27, 22:27] = 0.16
        b11[22:27, 22:27] = 0.26

        bands = {"B2": b2, "B3": b3, "B4": b4, "B8": b8, "B11": b11, "B10": b10}
        dem = np.full((h, w), 300.0, dtype=np.float32)
        meta = TileMetadata(
            sun_elevation_deg=45.0,
            sun_azimuth_deg=135.0,
            acquisition_date="2026-06-15",
        )

        pipeline = SatellitePipeline(config=PipelineConfig(model_type="rule_based"))
        outputs = pipeline.run(bands_or_path=bands, dem_or_path=dem, metadata_or_path=meta)

        # Verify radiative transfer steps were executed and logged
        self.assertGreaterEqual(len(outputs.change_log.radiative_corrections_applied), 3)

        # Verify Flood pixel (4, 4)
        self.assertEqual(int(outputs.labels[4, 4]), Label.LAND_CHANGE)
        self.assertEqual(
            int(outputs.land_change_subtypes[4, 4]), LandChangeSubtype.FLOOD_INUNDATION
        )
        self.assertEqual(str(outputs.reason_codes[4, 4]), REASON_LAND_CHANGE_FLOOD)
        self.assertAlmostEqual(float(outputs.trust_scores[4, 4]), 0.88, places=2)

        # Verify Fire pixel (14, 14)
        self.assertEqual(int(outputs.labels[14, 14]), Label.LAND_CHANGE)
        self.assertEqual(
            int(outputs.land_change_subtypes[14, 14]), LandChangeSubtype.FIRE_BURN_SCAR
        )
        self.assertEqual(str(outputs.reason_codes[14, 14]), REASON_LAND_CHANGE_FIRE)
        self.assertAlmostEqual(float(outputs.trust_scores[14, 14]), 0.88, places=2)

        # Verify Crop Damage pixel (24, 24)
        self.assertEqual(int(outputs.labels[24, 24]), Label.LAND_CHANGE)
        self.assertEqual(
            int(outputs.land_change_subtypes[24, 24]),
            LandChangeSubtype.CROP_DAMAGE_SENESCENCE,
        )
        self.assertEqual(str(outputs.reason_codes[24, 24]), REASON_LAND_CHANGE_CROP_DAMAGE)
        self.assertAlmostEqual(float(outputs.trust_scores[24, 24]), 0.88, places=2)


if __name__ == "__main__":
    unittest.main()
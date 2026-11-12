"""Unit tests for the 5 physics guardrails in fusion."""

import unittest
import numpy as np

from sat_pipeline.core.constants import (
    REASON_GUARDRAIL_HILLSHADE,
    REASON_GUARDRAIL_SHADOW_TO_LAND_CHANGE,
    REASON_GUARDRAIL_SNOW_TO_CLOUD,
    REASON_GUARDRAIL_WATER_TO_SHADOW,
    REASON_PHYSICS_AI_SHADOW_CONSENSUS,
    Label,
)
from sat_pipeline.core.types import PhysicsFeatures, PipelineConfig
from sat_pipeline.fusion.guardrails import PhysicsBoundedFusion


def create_mock_physics_features(shape=(1, 1), **overrides) -> PhysicsFeatures:
    """Helper to create dummy PhysicsFeatures."""
    defaults = {
        "predicted_shadow_mask": np.zeros(shape, dtype=np.uint8),
        "hillshade_mask": np.zeros(shape, dtype=np.uint8),
        "ndsi": np.zeros(shape, dtype=np.float32),
        "ndwi": np.zeros(shape, dtype=np.float32),
        "ndvi": np.zeros(shape, dtype=np.float32),
        "brightness": np.full(shape, 0.30, dtype=np.float32),
        "cirrus_mask": np.zeros(shape, dtype=np.uint8),
        "spectral_cloud_mask": np.zeros(shape, dtype=np.uint8),
        "spectral_water_mask": np.zeros(shape, dtype=np.uint8),
        "spectral_snow_mask": np.zeros(shape, dtype=np.uint8),
        "estimated_cloud_height_m": 1000.0,
        "shadow_overlap_score": 0.85,
        "illumination": np.full(shape, 0.70, dtype=np.float32),
        "slope_rad": np.zeros(shape, dtype=np.float32),
        "aspect_rad": np.zeros(shape, dtype=np.float32),
    }
    defaults.update(overrides)
    return PhysicsFeatures(**defaults)


class TestFusionGuardrails(unittest.TestCase):
    """Test every guardrail rule specified in the pitch deck."""

    def setUp(self):
        self.config = PipelineConfig()
        self.fusion = PhysicsBoundedFusion(self.config)

    def test_guardrail_1_shadow_disagreement_to_land_change(self):
        # Rule 1: If AI says shadow but geometric projection disagrees -> relabel as land_change with trust 0.55
        ai_probs = np.zeros((1, 1, 7), dtype=np.float32)
        ai_probs[0, 0, Label.CLOUD_SHADOW] = 0.90  # AI predicts shadow

        physics = create_mock_physics_features(
            predicted_shadow_mask=np.zeros((1, 1), dtype=np.uint8),  # geometric projection disagrees (0)
            brightness=np.array([[0.10]], dtype=np.float32),         # dark pixel
        )

        labels, names, trust, reasons, cl = self.fusion.fuse(ai_probs, physics)
        self.assertEqual(int(labels[0, 0]), Label.LAND_CHANGE)
        self.assertEqual(str(names[0, 0]), "land_change")
        self.assertAlmostEqual(float(trust[0, 0]), 0.55, places=3)
        self.assertIn("GUARDRAIL_LAND_CHANGE", str(reasons[0, 0]))
        self.assertEqual(cl.overrides["AI shadow rejected (relabelled as land_change)"], 1)

    def test_guardrail_2_water_overridden_by_predicted_shadow(self):
        # Rule 2: If AI says water but a geometric shadow is predicted -> shadow
        ai_probs = np.zeros((1, 1, 7), dtype=np.float32)
        ai_probs[0, 0, Label.WATER] = 0.85  # AI predicts water

        physics = create_mock_physics_features(
            predicted_shadow_mask=np.array([[1]], dtype=np.uint8),  # geometric shadow predicted
            brightness=np.array([[0.08]], dtype=np.float32),
        )

        labels, names, trust, reasons, cl = self.fusion.fuse(ai_probs, physics)
        self.assertEqual(int(labels[0, 0]), Label.CLOUD_SHADOW)
        self.assertEqual(str(names[0, 0]), "cloud_shadow")
        self.assertAlmostEqual(float(trust[0, 0]), 0.85, places=3)
        self.assertIn("GUARDRAIL_WATER_TO_SHADOW", str(reasons[0, 0]))
        self.assertEqual(cl.overrides["Water false-positive overridden by geometric shadow"], 1)

    def test_guardrail_3_snow_with_low_ndsi_to_cloud(self):
        # Rule 3: If AI says snow but NDSI is low -> cloud
        ai_probs = np.zeros((1, 1, 7), dtype=np.float32)
        ai_probs[0, 0, Label.SNOW] = 0.88  # AI predicts snow

        physics = create_mock_physics_features(
            ndsi=np.array([[0.15]], dtype=np.float32),  # low NDSI <= 0.40
            brightness=np.array([[0.75]], dtype=np.float32),
        )

        labels, names, trust, reasons, cl = self.fusion.fuse(ai_probs, physics)
        self.assertEqual(int(labels[0, 0]), Label.CLOUD)
        self.assertEqual(str(names[0, 0]), "cloud")
        self.assertAlmostEqual(float(trust[0, 0]), 0.80, places=3)
        self.assertIn("GUARDRAIL_SNOW_TO_CLOUD", str(reasons[0, 0]))
        self.assertEqual(cl.overrides["Low-NDSI snow overridden to cloud"], 1)

    def test_guardrail_4_shadow_consensus(self):
        # Rule 4: If physics confirms shadow and AI agrees -> trust 0.95
        ai_probs = np.zeros((1, 1, 7), dtype=np.float32)
        ai_probs[0, 0, Label.CLOUD_SHADOW] = 0.92  # AI predicts shadow

        physics = create_mock_physics_features(
            predicted_shadow_mask=np.array([[1]], dtype=np.uint8),  # physics confirms shadow
            brightness=np.array([[0.08]], dtype=np.float32),
        )

        labels, names, trust, reasons, cl = self.fusion.fuse(ai_probs, physics)
        self.assertEqual(int(labels[0, 0]), Label.CLOUD_SHADOW)
        self.assertAlmostEqual(float(trust[0, 0]), 0.95, places=3)
        self.assertIn("PHYSICS_AI_SHADOW_CONSENSUS", str(reasons[0, 0]))
        self.assertEqual(cl.overrides["Physics and AI confirmed shadow consensus"], 1)

    def test_guardrail_5_hillshade_explains_darkening(self):
        # Rule 5: If hillshade explains the darkening -> hillshade with trust 0.90
        ai_probs = np.zeros((1, 1, 7), dtype=np.float32)
        ai_probs[0, 0, Label.CLOUD_SHADOW] = 0.70  # AI thought shadow or other

        physics = create_mock_physics_features(
            hillshade_mask=np.array([[1]], dtype=np.uint8),  # terrain slope facing away
            brightness=np.array([[0.08]], dtype=np.float32), # dark pixel (< 0.15)
        )

        labels, names, trust, reasons, cl = self.fusion.fuse(ai_probs, physics)
        self.assertEqual(int(labels[0, 0]), Label.HILLSHADE)
        self.assertEqual(str(names[0, 0]), "hillshade")
        self.assertAlmostEqual(float(trust[0, 0]), 0.90, places=3)
        self.assertIn("GUARDRAIL_HILLSHADE", str(reasons[0, 0]))
        self.assertEqual(cl.overrides["Darkness explained by terrain hillshade"], 1)


if __name__ == "__main__":
    unittest.main()
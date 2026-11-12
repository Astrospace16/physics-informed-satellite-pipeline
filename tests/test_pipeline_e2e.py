"""End-to-end integration tests for the full satellite pipeline."""

import tempfile
import unittest
from pathlib import Path
import numpy as np

from sat_pipeline.core.constants import Label
from sat_pipeline.core.types import PipelineConfig
from sat_pipeline.ingest.synthetic import generate_synthetic_sentinel2_tile
from sat_pipeline.pipeline import SatellitePipeline


class TestPipelineE2E(unittest.TestCase):
    """End-to-end integration test on synthetic Sentinel-2 L2A tile."""

    def test_end_to_end_execution(self):
        # 1. Generate realistic synthetic scene with ground truth
        bands, dem, metadata, gt = generate_synthetic_sentinel2_tile(
            height=128,
            width=128,
            sun_elevation_deg=45.0,
            sun_azimuth_deg=135.0,
            cloud_height_m=1200.0,
            pixel_size_m=10.0,
            seed=42,
        )

        # 2. Run pipeline in demo rule-based mode
        config = PipelineConfig(model_type="rule_based")
        pipeline = SatellitePipeline(config=config)

        outputs = pipeline.run(
            bands_or_path=bands,
            dem_or_path=dem,
            metadata_or_path=metadata,
        )

        # Verify output shapes and types
        self.assertEqual(outputs.labels.shape, (128, 128))
        self.assertEqual(outputs.trust_scores.shape, (128, 128))
        self.assertEqual(outputs.reason_codes.shape, (128, 128))
        self.assertEqual(outputs.probabilities.shape, (128, 128, 7))

        # Trust scores must strictly lie in [0.0, 1.0]
        self.assertTrue(np.all((outputs.trust_scores >= 0.0) & (outputs.trust_scores <= 1.0)))

        # Verify estimated cloud height is within candidate search range [200, 2000]
        est_h = outputs.physics_features.estimated_cloud_height_m
        self.assertTrue(200.0 <= est_h <= 2000.0)

        # Verify change log has populated statistics
        cl = outputs.change_log
        self.assertEqual(cl.total_pixels, 128 * 128)
        self.assertIn("clear", cl.class_counts)
        self.assertIn("cloud", cl.class_counts)
        self.assertIn("cloud_shadow", cl.class_counts)

        # 3. Test export of deliverables to disk
        with tempfile.TemporaryDirectory() as tmp_dir:
            exported = pipeline.export_results(outputs, output_dir=tmp_dir)
            self.assertIn("labels", exported)
            self.assertIn("colorized", exported)
            self.assertIn("trust", exported)
            self.assertIn("changelog_json", exported)
            self.assertIn("changelog_md", exported)
            self.assertIn("reasons_summary", exported)

            for key, file_path in exported.items():
                self.assertTrue(Path(file_path).is_file())
                self.assertGreater(Path(file_path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
"""Unit tests for residual models (RuleBased, RandomForest, UNet) and factory."""

import tempfile
import unittest
from pathlib import Path
import numpy as np

from sat_pipeline.core.constants import NUM_CLASSES
from sat_pipeline.models.factory import ModelFactory
from sat_pipeline.models.random_forest import RandomForestResidualModel
from sat_pipeline.models.rule_based import RuleBasedPseudoProbabilityModel
from sat_pipeline.models.unet import UNetResidualModel


class TestResidualModels(unittest.TestCase):
    """Test model creation, fitting, prediction shapes, and serialization."""

    def test_factory_creation(self):
        m_rule = ModelFactory.create("rule_based")
        self.assertIsInstance(m_rule, RuleBasedPseudoProbabilityModel)

        m_rf = ModelFactory.create("random_forest")
        self.assertIsInstance(m_rf, RandomForestResidualModel)

        m_unet = ModelFactory.create("unet")
        self.assertIsInstance(m_unet, UNetResidualModel)

    def test_rule_based_pseudo_probability(self):
        model = ModelFactory.create("rule_based")
        # Dummy 12-channel feature map (10x10)
        feats = np.random.rand(10, 10, 12).astype(np.float32)
        probs = model.predict_proba(feats)

        self.assertEqual(probs.shape, (10, 10, NUM_CLASSES))
        # Sum of probabilities across classes should equal 1.0 everywhere
        row_sums = np.sum(probs, axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)

        preds = model.predict(feats)
        self.assertEqual(preds.shape, (10, 10))
        self.assertTrue(np.all((preds >= 0) & (preds < NUM_CLASSES)))

    def test_random_forest_fit_predict(self):
        model = RandomForestResidualModel(n_estimators=10, max_depth=6)
        x = np.random.rand(100, 12).astype(np.float32)
        y = np.random.randint(0, NUM_CLASSES, size=(100,)).astype(np.int32)

        model.fit(x, y)
        self.assertTrue(model.is_fitted)

        test_feats = np.random.rand(8, 8, 12).astype(np.float32)
        probs = model.predict_proba(test_feats)
        self.assertEqual(probs.shape, (8, 8, NUM_CLASSES))
        row_sums = np.sum(probs, axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)

    def test_unet_fit_predict(self):
        model = UNetResidualModel()
        feats = np.random.rand(16, 16, 12).astype(np.float32)
        labels = np.random.randint(0, NUM_CLASSES, size=(16, 16)).astype(np.int32)

        model.fit(feats, labels, epochs=2)
        probs = model.predict_proba(feats)
        self.assertEqual(probs.shape, (16, 16, NUM_CLASSES))

        row_sums = np.sum(probs, axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
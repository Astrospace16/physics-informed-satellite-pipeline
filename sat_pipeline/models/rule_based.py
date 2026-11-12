"""Fallback rule-based pseudo-probability model for demo mode without trained weights."""

from pathlib import Path
from typing import Union
import numpy as np

from sat_pipeline.core.constants import NUM_CLASSES, Label
from sat_pipeline.models.base import BaseResidualModel


class RuleBasedPseudoProbabilityModel(BaseResidualModel):
    """Rule-based pseudo-probability generator.

    Provides a fallback path so the pipeline runs with NO trained model (demo mode).
    Uses physical features, radiatively corrected bands, and disturbance indices
    (NBR, NDWI, NDVI, BSI) to synthesize calibrated pseudo-probabilities across the
    7 core classes, including active detection of flood, fire/burn scars, and crop damage.
    """

    def __init__(self):
        self.is_fitted = True

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        **kwargs,
    ) -> "RuleBasedPseudoProbabilityModel":
        """No-op for rule-based model."""
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Synthesize per-pixel pseudo-probabilities across the 7 classes.

        Feature layout (12 core channels; optional extra channels supported):
          0: B2, 1: B3, 2: B4, 3: B8, 4: B11, 5: B10,
          6: predicted_shadow_mask,
          7: hillshade_mask,
          8: ndsi,
          9: ndwi,
          10: ndvi,
          11: brightness
        """
        if features.ndim != 3 or features.shape[2] < 12:
            raise ValueError(f"Expected features shape (H, W, >=12), got {features.shape}")

        h, w, _ = features.shape
        b2 = features[:, :, 0]
        b3 = features[:, :, 1]
        b4 = features[:, :, 2]
        b8 = features[:, :, 3]
        b11 = features[:, :, 4]
        b10 = features[:, :, 5]
        predicted_shadow = features[:, :, 6]
        hillshade = features[:, :, 7]
        ndsi = features[:, :, 8]
        ndwi = features[:, :, 9]
        ndvi = features[:, :, 10]
        brightness = features[:, :, 11]

        # Derived physical disturbance indicators from the 12-channel stack
        eps = 1e-6
        nbr = (b8 - b11) / (b8 + b11 + eps)
        bsi = ((b11 + b4) - (b8 + b2)) / (b11 + b4 + b8 + b2 + eps)

        # Logits array for 7 classes
        logits = np.zeros((h, w, NUM_CLASSES), dtype=np.float32)

        # Base clear ground prior (healthy vegetation / normal bare soil)
        logits[:, :, Label.CLEAR] = 1.2 + 2.5 * np.clip(ndvi, 0.0, 1.0)

        # Cloud: Brightness > 0.20, Cirrus B10 > 0.03, opaque cloud
        cloud_signal = np.maximum(
            (brightness - 0.20) * 10.0,
            (b10 - 0.02) * 40.0,
        )
        # Suppress cloud if snow signature is dominant (NDSI > 0.4)
        cloud_signal = np.where(ndsi > 0.40, cloud_signal - 3.0, cloud_signal)
        logits[:, :, Label.CLOUD] = np.clip(cloud_signal, -3.0, 6.0)

        # Cloud Shadow: Dark pixels (< 0.15) boosted if in predicted shadow path
        darkness = np.maximum(0.0, 0.15 - brightness) / 0.15  # [0, 1]
        shadow_signal = (darkness * 4.0) + (predicted_shadow * 3.5)
        # Suppress shadow if hillshade is present
        shadow_signal = np.where(hillshade > 0.5, shadow_signal - 3.0, shadow_signal)
        logits[:, :, Label.CLOUD_SHADOW] = np.clip(shadow_signal, -3.0, 6.0)

        # Water: Permanent deep water (NDWI > 0.30 and B8 < 0.10)
        water_signal = ((ndwi - 0.25) * 9.0) + ((0.10 - b8) * 15.0)
        logits[:, :, Label.WATER] = np.clip(water_signal, -3.0, 6.0)

        # Snow: NDSI > 0.40 and high visible reflectance
        snow_signal = ((ndsi - 0.35) * 12.0) + ((brightness - 0.20) * 5.0)
        logits[:, :, Label.SNOW] = np.clip(snow_signal, -3.0, 6.0)

        # Hillshade: Low illumination on slopes
        hillshade_signal = hillshade * 5.0 + darkness * 2.0
        logits[:, :, Label.HILLSHADE] = np.clip(hillshade_signal, -3.0, 6.0)

        # Land Change: Direct physical signatures for Fire (burn scar), Flood (turbid inundation),
        # and Crop Damage (canopy collapse / bare soil stress), plus unshadowed dark anomalies
        fire_signal = np.where((nbr < -0.15) & (b11 > b8) & (brightness < 0.15), 5.0, 0.0)
        flood_signal = np.where(
            (ndwi >= 0.10) & (ndwi <= 0.30) & (b8 < 0.14) & (b11 < 0.09) & (brightness < 0.16),
            4.8,
            0.0,
        )
        crop_signal = np.where(
            (ndvi < 0.22) & (bsi > 0.08) & (b11 > 0.16) & (b4 > b3 * 1.05) & (brightness <= 0.22),
            4.5,
            0.0,
        )
        disturbance_signal = np.maximum(np.maximum(fire_signal, flood_signal), crop_signal)

        land_change_signal = (
            np.maximum(darkness * 3.5, disturbance_signal)
            - (predicted_shadow * 4.5)
            - (hillshade * 4.5)
        )
        logits[:, :, Label.LAND_CHANGE] = np.clip(land_change_signal, -3.0, 6.0)

        # Numerically stable softmax over classes
        max_logits = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probabilities = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-7)

        return probabilities.astype(np.float32)

    def save(self, model_path: Union[str, Path]) -> None:
        """Rule-based model has no learned weights; creates a marker file."""
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("RuleBasedPseudoProbabilityModel(v2.0)", encoding="utf-8")

    @classmethod
    def load(cls, model_path: Union[str, Path]) -> "RuleBasedPseudoProbabilityModel":
        return cls()
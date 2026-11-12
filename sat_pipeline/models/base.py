"""Abstract base class for physics-informed AI residual models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np


class BaseResidualModel(ABC):
    """Abstract interface for residual models in the satellite pipeline."""

    FEATURE_NAMES: List[str] = [
        "B2", "B3", "B4", "B8", "B11", "B10",
        "predicted_shadow_mask",
        "hillshade_mask",
        "ndsi",
        "ndwi",
        "ndvi",
        "brightness",
    ]

    @abstractmethod
    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        **kwargs,
    ) -> "BaseResidualModel":
        """Train the model on extracted feature stack and ground truth labels.

        Args:
            features: 3D array (H, W, C) or 2D array (N, C) of input features.
            labels: 2D array (H, W) or 1D array (N,) of ground truth class indices.
        """
        pass

    @abstractmethod
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict class probabilities for the feature stack.

        Args:
            features: 3D array of shape (H, W, C).

        Returns:
            probabilities: 3D array of shape (H, W, num_classes) where sum over classes = 1.
        """
        pass

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict most probable class index per pixel.

        Args:
            features: 3D array of shape (H, W, C).

        Returns:
            labels: 2D array of shape (H, W) with class indices [0..6].
        """
        proba = self.predict_proba(features)
        return np.argmax(proba, axis=-1).astype(np.int32)

    @abstractmethod
    def save(self, model_path: Union[str, Path]) -> None:
        """Serialize model weights/parameters to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, model_path: Union[str, Path]) -> "BaseResidualModel":
        """Load trained model from disk."""
        pass
"""RandomForest residual model implementation supporting OpenCV RTrees and scikit-learn."""

import os
from pathlib import Path
import pickle
from typing import Optional, Union
import numpy as np

from sat_pipeline.core.constants import NUM_CLASSES
from sat_pipeline.models.base import BaseResidualModel

# Check for scikit-learn availability
try:
    from sklearn.ensemble import RandomForestClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Check for OpenCV availability
try:
    import cv2
    HAS_OPENCV = hasattr(cv2, "ml") and hasattr(cv2.ml, "RTrees_create")
except ImportError:
    HAS_OPENCV = False


class RandomForestResidualModel(BaseResidualModel):
    """RandomForest residual classifier for satellite pixel classification.

    Supports scikit-learn RandomForestClassifier when available, with automatic
    high-performance fallback to OpenCV cv2.ml.RTrees.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 12,
        min_samples_split: int = 4,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.backend = "sklearn" if HAS_SKLEARN else ("opencv" if HAS_OPENCV else "numpy")
        self.model = None
        self.is_fitted = False

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        subsample_limit: int = 50000,
        **kwargs,
    ) -> "RandomForestResidualModel":
        """Fit the random forest on pixel features.

        Args:
            features: (H, W, C) or (N, C) float32 array.
            labels: (H, W) or (N,) int32 array.
            subsample_limit: Max pixel samples used to speed up training.
        """
        # Reshape to 2D tabular (N, C)
        if features.ndim == 3:
            h, w, c = features.shape
            x_flat = features.reshape(-1, c)
            y_flat = labels.reshape(-1)
        else:
            x_flat = features
            y_flat = labels.reshape(-1)

        # Remove NaN or Inf if any
        valid_mask = np.isfinite(x_flat).all(axis=1) & np.isfinite(y_flat)
        x_valid = x_flat[valid_mask].astype(np.float32)
        y_valid = y_flat[valid_mask].astype(np.int32)

        # Subsample if dataset is very large
        n_samples = x_valid.shape[0]
        if n_samples > subsample_limit:
            rng = np.random.default_rng(self.random_state)
            indices = rng.choice(n_samples, size=subsample_limit, replace=False)
            x_train = x_valid[indices]
            y_train = y_valid[indices]
        else:
            x_train = x_valid
            y_train = y_valid

        if self.backend == "sklearn":
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self.model.fit(x_train, y_train)
        elif self.backend == "opencv":
            rt = cv2.ml.RTrees_create()
            rt.setMaxDepth(self.max_depth)
            rt.setMinSampleCount(self.min_samples_split)
            rt.setTermCriteria(
                (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, self.n_estimators, 0.01)
            )
            train_data = cv2.ml.TrainData_create(
                x_train,
                cv2.ml.ROW_SAMPLE,
                y_train.reshape(-1, 1),
            )
            rt.train(train_data)
            self.model = rt
        else:
            raise RuntimeError("Neither scikit-learn nor OpenCV is available for RandomForest.")

        self.is_fitted = True
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict class probabilities for feature stack."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() or load() first.")

        orig_ndim = features.ndim
        if orig_ndim == 3:
            h, w, c = features.shape
            x_flat = features.reshape(-1, c).astype(np.float32)
        else:
            x_flat = features.astype(np.float32)
            h, w = x_flat.shape[0], 1

        n_samples = x_flat.shape[0]
        proba = np.zeros((n_samples, NUM_CLASSES), dtype=np.float32)

        if self.backend == "sklearn":
            # Scikit-learn predict_proba
            raw_proba = self.model.predict_proba(x_flat)
            # Map model classes to full 7-class range
            for idx, cls_id in enumerate(self.model.classes_):
                if 0 <= cls_id < NUM_CLASSES:
                    proba[:, cls_id] = raw_proba[:, idx]
        elif self.backend == "opencv":
            # OpenCV RTrees predicts class IDs
            _, preds = self.model.predict(x_flat)
            preds = np.clip(preds.ravel().astype(np.int32), 0, NUM_CLASSES - 1)

            # Build calibrated probabilities with high confidence on predicted class
            # and uniform distribution on background
            for i in range(NUM_CLASSES):
                proba[:, i] = 0.02  # smoothing baseline
            proba[np.arange(n_samples), preds] = 0.88
            # Normalize row sums to 1.0
            proba = proba / np.sum(proba, axis=-1, keepdims=True)

        if orig_ndim == 3:
            return proba.reshape(h, w, NUM_CLASSES)
        return proba

    def save(self, model_path: Union[str, Path]) -> None:
        """Serialize model to file."""
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "backend": self.backend,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "random_state": self.random_state,
            "is_fitted": self.is_fitted,
        }

        if self.backend == "sklearn":
            with open(path, "wb") as f:
                pickle.dump({"meta": meta, "model": self.model}, f)
        elif self.backend == "opencv":
            xml_path = str(path.with_suffix(".xml"))
            self.model.save(xml_path)
            with open(path, "wb") as f:
                pickle.dump({"meta": meta, "xml_path": os.path.basename(xml_path)}, f)

    @classmethod
    def load(cls, model_path: Union[str, Path]) -> "RandomForestResidualModel":
        """Load serialized model."""
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)

        meta = data["meta"]
        instance = cls(
            n_estimators=meta["n_estimators"],
            max_depth=meta["max_depth"],
            min_samples_split=meta["min_samples_split"],
            random_state=meta["random_state"],
        )
        instance.backend = meta["backend"]
        instance.is_fitted = meta["is_fitted"]

        if meta["backend"] == "sklearn":
            instance.model = data["model"]
        elif meta["backend"] == "opencv":
            xml_name = data.get("xml_path", path.with_suffix(".xml").name)
            xml_full_path = str(path.parent / xml_name)
            rt = cv2.ml.RTrees_load(xml_full_path)
            instance.model = rt

        return instance
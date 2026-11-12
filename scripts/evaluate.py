#!/usr/bin/env python3
"""Evaluation utility comparing pipeline output against ground-truth classification."""

import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sat_pipeline.core.constants import LABEL_NAMES, NUM_CLASSES, Label
from sat_pipeline.core.types import PipelineConfig
from sat_pipeline.ingest.synthetic import generate_synthetic_sentinel2_tile
from sat_pipeline.pipeline import SatellitePipeline


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    trust_scores: np.ndarray,
):
    """Compute per-class Precision, Recall, F1, and trust calibration."""
    y_t = y_true.ravel()
    y_p = y_pred.ravel()
    trust = trust_scores.ravel()

    # Confusion Matrix
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(y_t, y_p):
        if 0 <= t < NUM_CLASSES and 0 <= p < NUM_CLASSES:
            cm[t, p] += 1

    overall_acc = float(np.sum(np.diag(cm))) / float(len(y_t)) * 100.0

    print("\n" + "=" * 75)
    print("PHYSICS-BOUNDED PIPELINE EVALUATION REPORT")
    print("=" * 75)
    print(f"Overall Accuracy: {overall_acc:.2f}%")
    print("\n| Class ID | Class Name | Precision (%) | Recall (%) | F1-Score (%) | Support |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")

    f1_list = []
    for cls_id in range(NUM_CLASSES):
        name = LABEL_NAMES[cls_id]
        tp = cm[cls_id, cls_id]
        fp = np.sum(cm[:, cls_id]) - tp
        fn = np.sum(cm[cls_id, :]) - tp
        support = np.sum(cm[cls_id, :])

        prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        if support > 0:
            f1_list.append(f1)

        print(f"| {cls_id} | `{name:12s}` | {prec:12.2f}% | {rec:10.2f}% | {f1:11.2f}% | {support:7,d} |")

    macro_f1 = np.mean(f1_list) if f1_list else 0.0
    print(f"\nMacro Average F1-Score: {macro_f1:.2f}%")

    # Trust Score Calibration
    correct_mask = (y_t == y_p)
    mean_trust_correct = float(np.mean(trust[correct_mask])) if np.any(correct_mask) else 0.0
    mean_trust_incorrect = float(np.mean(trust[~correct_mask])) if np.any(~correct_mask) else 0.0
    print("\nTrust Score Calibration:")
    print(f"  - Mean Trust on Correct Pixels  : {mean_trust_correct:.4f}")
    print(f"  - Mean Trust on Incorrect Pixels: {mean_trust_incorrect:.4f}")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate pipeline against ground truth.")
    parser.add_argument("--model-type", type=str, default="rule_based")
    args = parser.parse_args()

    # Generate test tile with ground truth
    bands, dem, meta, gt = generate_synthetic_sentinel2_tile(
        height=256,
        width=256,
        sun_elevation_deg=45.0,
        sun_azimuth_deg=135.0,
        cloud_height_m=1200.0,
        seed=123,
    )

    pipeline = SatellitePipeline(config=PipelineConfig(model_type=args.model_type))
    outputs = pipeline.run(bands_or_path=bands, dem_or_path=dem, metadata_or_path=meta)

    evaluate_predictions(gt, outputs.labels, outputs.trust_scores)


if __name__ == "__main__":
    main()
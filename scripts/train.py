#!/usr/bin/env python3
"""Training script for physics-informed AI residual models."""

import argparse
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sat_pipeline.core.constants import LABEL_NAMES, NUM_CLASSES
from sat_pipeline.core.types import PipelineConfig
from sat_pipeline.ingest.synthetic import generate_synthetic_sentinel2_tile
from sat_pipeline.models.factory import ModelFactory
from sat_pipeline.pipeline import SatellitePipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Train residual ML model on labelled satellite tiles.")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["random_forest", "unet"],
        default="random_forest",
        help="Model architecture to train (random_forest or unet).",
    )
    parser.add_argument(
        "--num-tiles",
        type=int,
        default=4,
        help="Number of synthetic tiles to generate for training (default: 4).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="Height and width of training tiles (default: 256).",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="models/trained_residual_model.pkl",
        help="Destination path to save trained model weights.",
    )
    parser.add_argument(
        "--estimators",
        type=int,
        default=40,
        help="Number of trees for RandomForest (default: 40).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Training epochs for U-Net (default: 8).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 65)
    print(f"TRAINING RESIDUAL AI MODEL: {args.model_type.upper()}")
    print("=" * 65)

    pipeline = SatellitePipeline(config=PipelineConfig(model_type=args.model_type))

    all_features = []
    all_labels = []

    print(f"\n[1/3] Generating {args.num_tiles} labelled training scenes...")
    for idx in range(args.num_tiles):
        sun_elev = float(35.0 + idx * 8.0)
        sun_azim = float(120.0 + idx * 25.0)
        cloud_h = float(800.0 + idx * 300.0)

        bands, dem, meta, gt = generate_synthetic_sentinel2_tile(
            height=args.tile_size,
            width=args.tile_size,
            sun_elevation_deg=sun_elev,
            sun_azimuth_deg=sun_azim,
            cloud_height_m=cloud_h,
            seed=100 + idx,
        )

        physics = pipeline.extract_physics_features(bands, dem, meta)
        feats = pipeline.assemble_feature_stack(bands, physics)

        all_features.append(feats)
        all_labels.append(gt)
        print(f"  - Tile {idx+1}: Sun Elev={sun_elev:.1f}°, Azim={sun_azim:.1f}°, Cloud H={cloud_h:.1f}m")

    # Stack features
    features_stack = np.concatenate(all_features, axis=0)
    labels_stack = np.concatenate(all_labels, axis=0)
    print(f"\n[2/3] Training feature tensor assembled: {features_stack.shape}")

    # Instantiate model
    print(f"\n[3/3] Fitting {args.model_type} model...")
    if args.model_type == "random_forest":
        model = ModelFactory.create("random_forest", n_estimators=args.estimators, max_depth=12)
        model.fit(features_stack, labels_stack)
    else:
        model = ModelFactory.create("unet")
        model.fit(features_stack, labels_stack, epochs=args.epochs)

    # Save model
    out_model_path = Path(args.output_model)
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_model_path)
    print(f"\n[Success] Model successfully trained and saved to: {out_model_path}")

    # Validation evaluation on fresh unseen tile
    print("\n[Validation] Evaluating on hold-out test scene...")
    val_bands, val_dem, val_meta, val_gt = generate_synthetic_sentinel2_tile(
        height=args.tile_size,
        width=args.tile_size,
        sun_elevation_deg=48.0,
        sun_azimuth_deg=142.0,
        cloud_height_m=1100.0,
        seed=999,
    )
    val_physics = pipeline.extract_physics_features(val_bands, val_dem, val_meta)
    val_feats = pipeline.assemble_feature_stack(val_bands, val_physics)
    val_preds = model.predict(val_feats)

    acc = np.mean(val_preds == val_gt) * 100.0
    print(f"  Hold-out Pixel Accuracy (Pre-Guardrail AI): {acc:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""CLI runner for the Physics-Informed Satellite Pipeline."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sat_pipeline.core.types import PipelineConfig, TileMetadata
from sat_pipeline.ingest.synthetic import generate_synthetic_sentinel2_tile, save_synthetic_dataset
from sat_pipeline.pipeline import SatellitePipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Physics-Informed Satellite Pipeline on a Sentinel-2 L2A tile."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to multi-band TIFF file (or omit to run on auto-generated demo scene).",
    )
    parser.add_argument(
        "--dem",
        "-d",
        type=str,
        default=None,
        help="Path to DEM elevation TIFF file (optional).",
    )
    parser.add_argument(
        "--metadata",
        "-m",
        type=str,
        default=None,
        help="Path to metadata JSON file (optional).",
    )
    parser.add_argument(
        "--sun-elevation",
        type=float,
        default=45.0,
        help="Sun elevation angle in degrees above horizon (default 45.0).",
    )
    parser.add_argument(
        "--sun-azimuth",
        type=float,
        default=135.0,
        help="Sun azimuth angle in degrees clockwise from North (default 135.0).",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=10.0,
        help="Ground sampling distance in meters (default 10.0m).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["rule_based", "random_forest", "unet"],
        default="rule_based",
        help="AI residual model architecture (default: rule_based demo mode).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to pre-trained model file.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="outputs/demo_run",
        help="Directory to save output rasters, changelog, and diagnostics.",
    )
    parser.add_argument(
        "--generate-demo-data",
        action="store_true",
        help="Explicitly generate synthetic demo data on disk before running.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("PHYSICS-INFORMED SATELLITE PIPELINE (Sentinel-2 L2A)")
    print("Separating Clouds from Shadows via Sun Geometry, Spectral Indices & AI")
    print("=" * 70)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = args.input
    dem_path = args.dem

    if args.generate_demo_data or input_path is None:
        print("\n[Ingest] Generating synthetic Sentinel-2 L2A tile with realistic physics...")
        demo_data_dir = output_dir / "demo_input"
        save_synthetic_dataset(
            output_dir=str(demo_data_dir),
            height=256,
            width=256,
            sun_elevation_deg=args.sun_elevation,
            sun_azimuth_deg=args.sun_azimuth,
            cloud_height_m=1200.0,
        )
        input_path = str(demo_data_dir / "sentinel2_l2a_bands.tif")
        dem_path = str(demo_data_dir / "dem_elevation.tif")
        args.metadata = str(demo_data_dir / "metadata.json")
        print(f"[Ingest] Demo dataset created at: {demo_data_dir}")

    # Build Pipeline Configuration
    config = PipelineConfig(
        model_type=args.model_type,
        model_path=args.model_path,
        default_pixel_size_m=args.pixel_size,
    )

    print(f"\n[Pipeline] Initializing pipeline with model: '{config.model_type}'...")
    pipeline = SatellitePipeline(config=config)

    print(f"[Pipeline] Processing tile from {input_path}...")
    outputs = pipeline.run(
        bands_or_path=input_path,
        dem_or_path=dem_path,
        metadata_or_path=args.metadata,
        sun_elevation_deg=args.sun_elevation,
        sun_azimuth_deg=args.sun_azimuth,
        pixel_size_m=args.pixel_size,
    )

    print("\n[Export] Saving classification maps and change log...")
    exported = pipeline.export_results(outputs, output_dir=output_dir)

    print(f"\n[Success] Processing complete! Exported deliverables:")
    for name, p in exported.items():
        print(f"  - {name:16s}: {p}")

    print("\n" + outputs.change_log.to_markdown())
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
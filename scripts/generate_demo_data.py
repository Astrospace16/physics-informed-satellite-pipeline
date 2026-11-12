#!/usr/bin/env python3
"""CLI utility to generate synthetic Sentinel-2 L2A tiles with ground truth."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sat_pipeline.ingest.synthetic import save_synthetic_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Sentinel-2 L2A tile.")
    parser.add_argument("--output-dir", "-o", type=str, default="data/demo_tile")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--sun-elevation", type=float, default=45.0)
    parser.add_argument("--sun-azimuth", type=float, default=135.0)
    parser.add_argument("--cloud-height", type=float, default=1200.0)
    args = parser.parse_args()

    out_path = save_synthetic_dataset(
        output_dir=args.output_dir,
        height=args.height,
        width=args.width,
        sun_elevation_deg=args.sun_elevation,
        sun_azimuth_deg=args.sun_azimuth,
        cloud_height_m=args.cloud_height,
    )
    print(f"Synthetic Sentinel-2 dataset generated at: {out_path}")


if __name__ == "__main__":
    main()
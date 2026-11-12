"""Synthetic Sentinel-2 L2A tile generator with ground-truth physics features, noise, and disturbances."""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from PIL import Image

from sat_pipeline.core.constants import Label
from sat_pipeline.core.types import TileMetadata
from sat_pipeline.physics.hillshade import compute_hillshade_mask
from sat_pipeline.physics.sun_geometry import project_cloud_shadow


def generate_synthetic_sentinel2_tile(
    height: int = 256,
    width: int = 256,
    sun_elevation_deg: float = 45.0,
    sun_azimuth_deg: float = 135.0,
    cloud_height_m: float = 1200.0,
    pixel_size_m: float = 10.0,
    acquisition_date: str = "2026-05-15",
    inject_sensor_noise: bool = True,
    seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, TileMetadata, np.ndarray]:
    """Generate a realistic synthetic Sentinel-2 L2A tile with known ground truth.

    The scene contains:
      - Mountainous terrain DEM generating real hillshade.
      - Cumulus clouds at `cloud_height_m`.
      - Real physical ground shadows cast at (sun_elevation, sun_azimuth).
      - Permanent water lake with NDWI > 0.3 and NIR < 0.1.
      - Snow cap with NDSI > 0.4.
      - Thin cirrus cloud with B10 > 0.03.
      - Distinct `land_change` disturbance zones (not aligned with any cloud shadow):
          (a) Active Fire / Burn Scar (charcoal/ash: NBR < -0.15, B11 > B8, brightness < 0.09)
          (b) Flood Inundation zone (turbid floodwater: NDWI ~ 0.18, low B8 & B11, outside permanent lake)
          (c) Crop Damage / Severe Canopy Senescence patch (collapsed NDVI, high SWIR B11 & red B4)
      - Realistic salt-and-pepper / impulse sensor noise speckles (`inject_sensor_noise=True`).
      - Clear sunlit vegetation/soil background (brightness in [0.16, 0.185]).

    Returns:
        bands: Dict of 2D float32 reflectance arrays for ['B2', 'B3', 'B4', 'B8', 'B11', 'B10'].
        dem: 2D float32 array of elevation in meters.
        metadata: TileMetadata object.
        ground_truth: 2D int32 array of ground truth Label values.
    """
    rng = np.random.default_rng(seed)

    # 1. Coordinate Grid
    y_coords, x_coords = np.mgrid[0:height, 0:width]

    # 2. Digital Elevation Model (DEM)
    dem = 300.0 + 80.0 * np.sin(x_coords / 40.0) * np.cos(y_coords / 40.0)
    mountain_dist = np.sqrt(
        (x_coords - (width * 0.75)) ** 2 + (y_coords - (height * 0.25)) ** 2
    )
    mountain_cone = np.maximum(0.0, 1500.0 - (mountain_dist * 18.0))
    dem = (dem + mountain_cone).astype(np.float32)

    # Compute hillshade from DEM
    hillshade_mask, illumination, _, _ = compute_hillshade_mask(
        dem=dem,
        sun_elevation_deg=sun_elevation_deg,
        sun_azimuth_deg=sun_azimuth_deg,
        pixel_size_m=pixel_size_m,
        illumination_threshold=0.10,
    )

    # 3. Cloud Mask (Cumulus + Cirrus)
    cloud_mask = np.zeros((height, width), dtype=np.uint8)
    c1_dist = np.sqrt(
        (x_coords - (width * 0.52)) ** 2 + (y_coords - (height * 0.56)) ** 2
    )
    cloud_mask[c1_dist < (width * 0.09)] = 1
    c2_dist = np.sqrt(
        (x_coords - (width * 0.62)) ** 2 + (y_coords - (height * 0.48)) ** 2
    )
    cloud_mask[c2_dist < (width * 0.05)] = 1

    # 4. Snow Peak (high mountain summit)
    snow_mask = ((dem > 1300.0) & (cloud_mask == 0)).astype(np.uint8)

    # 5. Cirrus Filament (high altitude streak in lower-mid right so its NW shadow stays in bounds)
    cirrus_dist = np.abs((y_coords - (height * 0.68)) - 0.20 * (x_coords - width * 0.45))
    cirrus_mask = (
        (cirrus_dist < 4)
        & (x_coords >= width * 0.48)
        & (x_coords <= width * 0.70)
        & (cloud_mask == 0)
        & (snow_mask == 0)
    ).astype(np.uint8)
    cloud_mask[cirrus_mask > 0] = 1

    # 6. Physical Shadow Projection (cast by full cloud mask)
    projected_shadow = project_cloud_shadow(
        cloud_mask=cloud_mask,
        cloud_height_m=cloud_height_m,
        sun_elevation_deg=sun_elevation_deg,
        sun_azimuth_deg=sun_azimuth_deg,
        pixel_size_m=pixel_size_m,
    )
    ground_shadow = (projected_shadow & (cloud_mask == 0)).astype(np.uint8)

    # 7. Permanent Water Body (lake in bottom-left)
    water_dist = np.sqrt(
        (x_coords - (width * 0.20)) ** 2 + (y_coords - (height * 0.80)) ** 2
    )
    water_mask = (water_dist < (width * 0.12)).astype(np.uint8)

    # 8. Land Change Disturbance Zones (Fire Burn Scar, Flood Inundation, Crop Damage)
    unoccluded = (
        (cloud_mask == 0)
        & (ground_shadow == 0)
        & (water_mask == 0)
        & (snow_mask == 0)
        & (hillshade_mask == 0)
    )

    # 8a. Active Fire / Burn Scar (bottom-right circular patch)
    fire_dist = np.sqrt(
        (x_coords - (width * 0.80)) ** 2 + (y_coords - (height * 0.80)) ** 2
    )
    fire_zone = ((fire_dist < (width * 0.065)) & unoccluded).astype(np.uint8)

    # 8b. Flood Inundation Zone (overflow plain near the lake in bottom-center)
    flood_dist = np.sqrt(
        (x_coords - (width * 0.44)) ** 2 + (y_coords - (height * 0.84)) ** 2
    )
    flood_zone = (
        (flood_dist < (width * 0.055)) & unoccluded & (fire_zone == 0)
    ).astype(np.uint8)

    # 8c. Crop Damage / Severe Senescence Field (rectangular agricultural parcel in mid-right)
    crop_rect = (
        (x_coords >= int(width * 0.74))
        & (x_coords < int(width * 0.86))
        & (y_coords >= int(height * 0.52))
        & (y_coords < int(height * 0.64))
    )
    crop_zone = (
        crop_rect & unoccluded & (fire_zone == 0) & (flood_zone == 0)
    ).astype(np.uint8)

    land_change_mask = (fire_zone | flood_zone | crop_zone).astype(np.uint8)

    # 9. Synthesize Spectral Reflectance Bands (B2, B3, B4, B8, B11, B10)
    # Clear sunlit background has visible brightness in [0.160, 0.180] (> 0.15 dark threshold, < 0.20 cloud threshold)
    noise = rng.normal(0.0, 0.003, size=(height, width)).astype(np.float32)
    b2 = np.clip(0.158 + 0.008 * (y_coords / height) + noise, 0.152, 0.175)  # Blue
    b3 = np.clip(0.172 + 0.008 * (y_coords / height) + noise, 0.165, 0.188)  # Green
    b4 = np.clip(0.158 + 0.008 * (y_coords / height) + noise, 0.152, 0.175)  # Red
    b8 = np.clip(0.42 + 0.05 * (x_coords / width) + noise, 0.30, 0.70)       # NIR (healthy vegetation, NDVI ~ 0.45)
    b11 = np.clip(0.22 + 0.02 * (y_coords / height) + noise, 0.16, 0.32)     # SWIR-1
    b10 = np.clip(0.005 + 0.002 * rng.random((height, width)), 0.001, 0.02)

    # Apply Permanent Water Signature: High NDWI (> 0.3), Low B8 (< 0.1)
    w_idx = water_mask > 0
    b2[w_idx] = 0.09
    b3[w_idx] = 0.12
    b4[w_idx] = 0.05
    b8[w_idx] = 0.03
    b11[w_idx] = 0.02

    # Apply Snow Signature: High NDSI (> 0.4), bright visible, low SWIR
    s_idx = snow_mask > 0
    b2[s_idx] = 0.75
    b3[s_idx] = 0.72
    b4[s_idx] = 0.70
    b8[s_idx] = 0.65
    b11[s_idx] = 0.15

    # Apply Opaque Cloud Signature
    c_idx = (cloud_mask > 0) & (cirrus_mask == 0)
    b2[c_idx] = 0.65
    b3[c_idx] = 0.68
    b4[c_idx] = 0.66
    b8[c_idx] = 0.60
    b11[c_idx] = 0.40

    # Apply Cirrus Signature: B10 > 0.03 and elevated visible
    cir_idx = cirrus_mask > 0
    b10[cir_idx] = 0.045
    b2[cir_idx] = 0.28
    b3[cir_idx] = 0.29
    b4[cir_idx] = 0.27
    b8[cir_idx] = 0.32

    # Apply Cloud Shadow Signature: Darkened across all bands (brightness ~ 0.055 < 0.15)
    sh_idx = ground_shadow > 0
    b2[sh_idx] = 0.05
    b3[sh_idx] = 0.06
    b4[sh_idx] = 0.05
    b8[sh_idx] = 0.11
    b11[sh_idx] = 0.07

    # Apply Hillshade Darkness: Terrain self-shadowing on steep slopes
    hs_idx = (hillshade_mask > 0) & (cloud_mask == 0) & (ground_shadow == 0)
    b2[hs_idx] = 0.04
    b3[hs_idx] = 0.05
    b4[hs_idx] = 0.04
    b8[hs_idx] = 0.12
    b11[hs_idx] = 0.07

    # Apply Fire / Burn Scar Signature: Charcoal/ash (NBR < -0.15, B11 > B8, brightness < 0.09)
    f_idx = fire_zone > 0
    b2[f_idx] = 0.05
    b3[f_idx] = 0.05
    b4[f_idx] = 0.06
    b8[f_idx] = 0.06
    b11[f_idx] = 0.16

    # Apply Flood Inundation Signature: Turbid shallow water (NDWI in [0.12, 0.28], low B8 & B11)
    fl_idx = flood_zone > 0
    b2[fl_idx] = 0.08
    b3[fl_idx] = 0.11
    b4[fl_idx] = 0.07
    b8[fl_idx] = 0.075
    b11[fl_idx] = 0.04

    # Apply Crop Damage / Senescence Signature: Destroyed chlorophyll (low NDVI < 0.15, high B11 & B4)
    cr_idx = crop_zone > 0
    b2[cr_idx] = 0.08
    b3[cr_idx] = 0.10
    b4[cr_idx] = 0.15
    b8[cr_idx] = 0.16
    b11[cr_idx] = 0.26

    # Inject isolated single-pixel sensor impulse noise / speckles
    if inject_sensor_noise:
        num_spikes = max(8, (height * width) // 1200)
        spike_rows = rng.integers(2, height - 2, size=num_spikes)
        spike_cols = rng.integers(2, width - 2, size=num_spikes)
        for r, c in zip(spike_rows, spike_cols):
            b2[r, c] = 0.92
            b4[r, c] = 0.001

    # Build Ground Truth Map
    ground_truth = np.full((height, width), Label.CLEAR, dtype=np.int32)
    ground_truth[land_change_mask > 0] = Label.LAND_CHANGE
    ground_truth[snow_mask > 0] = Label.SNOW
    ground_truth[water_mask > 0] = Label.WATER
    ground_truth[hs_idx] = Label.HILLSHADE
    ground_truth[ground_shadow > 0] = Label.CLOUD_SHADOW
    ground_truth[cloud_mask > 0] = Label.CLOUD

    bands = {
        "B2": np.clip(b2, 0.0, 1.0).astype(np.float32),
        "B3": np.clip(b3, 0.0, 1.0).astype(np.float32),
        "B4": np.clip(b4, 0.0, 1.0).astype(np.float32),
        "B8": np.clip(b8, 0.0, 1.0).astype(np.float32),
        "B11": np.clip(b11, 0.0, 1.0).astype(np.float32),
        "B10": np.clip(b10, 0.0, 1.0).astype(np.float32),
    }

    metadata = TileMetadata(
        sun_elevation_deg=sun_elevation_deg,
        sun_azimuth_deg=sun_azimuth_deg,
        pixel_size_m=pixel_size_m,
        tile_id="SYNTHETIC_S2A_TEST_TILE",
        acquisition_date=acquisition_date,
        band_names=["B2", "B3", "B4", "B8", "B11", "B10"],
    )

    return bands, dem, metadata, ground_truth


def save_synthetic_dataset(
    output_dir: str,
    height: int = 256,
    width: int = 256,
    sun_elevation_deg: float = 45.0,
    sun_azimuth_deg: float = 135.0,
    cloud_height_m: float = 1200.0,
) -> Path:
    """Generate and save a synthetic Sentinel-2 tile as GeoTIFF files and metadata."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    bands, dem, metadata, gt = generate_synthetic_sentinel2_tile(
        height=height,
        width=width,
        sun_elevation_deg=sun_elevation_deg,
        sun_azimuth_deg=sun_azimuth_deg,
        cloud_height_m=cloud_height_m,
    )

    band_order = ["B2", "B3", "B4", "B8", "B11", "B10"]
    images = []
    for bname in band_order:
        arr_u16 = np.clip(bands[bname] * 10000.0, 0, 65535).astype(np.uint16)
        images.append(Image.fromarray(arr_u16))

    tif_path = out_path / "sentinel2_l2a_bands.tif"
    images[0].save(str(tif_path), save_all=True, append_images=images[1:])

    dem_im = Image.fromarray(np.clip(dem, 0, 65535).astype(np.uint16))
    dem_path = out_path / "dem_elevation.tif"
    dem_im.save(str(dem_path))

    gt_im = Image.fromarray(gt.astype(np.uint8))
    gt_path = out_path / "ground_truth.png"
    gt_im.save(str(gt_path))

    meta_dict = {
        "tile_id": metadata.tile_id,
        "acquisition_date": metadata.acquisition_date,
        "sun_elevation_deg": metadata.sun_elevation_deg,
        "sun_azimuth_deg": metadata.sun_azimuth_deg,
        "pixel_size_m": metadata.pixel_size_m,
        "bands": band_order,
        "scale_factor": 10000.0,
        "true_cloud_height_m": cloud_height_m,
    }
    meta_path = out_path / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)

    return out_path
"""Ingestion and reader routines for Sentinel-2 L2A and multi-band GeoTIFF data."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from sat_pipeline.core.constants import DEFAULT_BAND_ORDER
from sat_pipeline.core.types import TileMetadata


class TileReader:
    """Reader for Sentinel-2 L2A tiles, multi-band GeoTIFFs, and DEM data."""

    @staticmethod
    def read_multiband_tiff(
        tiff_path: Union[str, Path],
        band_names: Optional[List[str]] = None,
        scale_factor: float = 10000.0,
    ) -> Dict[str, np.ndarray]:
        """Read a multi-frame / multi-band TIFF file into float32 reflectance arrays [0.0, 1.0].

        Supports both PIL multi-frame TIFFs and 3D arrays.
        """
        path = Path(tiff_path)
        if not path.is_file():
            raise FileNotFoundError(f"TIFF file not found: {path}")

        band_names = band_names or DEFAULT_BAND_ORDER
        bands: Dict[str, np.ndarray] = {}

        with Image.open(path) as img:
            n_frames = getattr(img, "n_frames", 1)
            if n_frames > 1:
                for idx in range(min(n_frames, len(band_names))):
                    img.seek(idx)
                    arr = np.array(img).astype(np.float32)
                    # Normalize if digital numbers
                    if np.max(arr) > 1.5:
                        arr = arr / scale_factor
                    bands[band_names[idx]] = np.clip(arr, 0.0, 1.0)
            else:
                arr = np.array(img).astype(np.float32)
                if arr.ndim == 3:
                    # Shape (H, W, C)
                    for idx in range(min(arr.shape[2], len(band_names))):
                        b_arr = arr[:, :, idx]
                        if np.max(b_arr) > 1.5:
                            b_arr = b_arr / scale_factor
                        bands[band_names[idx]] = np.clip(b_arr, 0.0, 1.0)
                else:
                    # Single band
                    if np.max(arr) > 1.5:
                        arr = arr / scale_factor
                    bands[band_names[0]] = np.clip(arr, 0.0, 1.0)

        # Ensure all expected bands are populated
        first_band = next(iter(bands.values()))
        h, w = first_band.shape
        for bname in band_names:
            if bname not in bands:
                bands[bname] = np.zeros((h, w), dtype=np.float32)

        return bands

    @staticmethod
    def read_dem(
        dem_path: Optional[Union[str, Path]],
        reference_shape: Tuple[int, int],
        default_elevation_m: float = 300.0,
    ) -> np.ndarray:
        """Read DEM file (TIFF) or generate a flat reference surface if absent."""
        if dem_path is None or not Path(dem_path).is_file():
            return np.full(reference_shape, default_elevation_m, dtype=np.float32)

        with Image.open(dem_path) as img:
            dem = np.array(img).astype(np.float32)
            if dem.ndim == 3:
                dem = dem[:, :, 0]
            if dem.shape != reference_shape:
                dem_im = Image.fromarray(dem)
                dem_resized = dem_im.resize((reference_shape[1], reference_shape[0]), Image.BILINEAR)
                dem = np.array(dem_resized).astype(np.float32)
            return dem

    @staticmethod
    def read_metadata(
        meta_path: Optional[Union[str, Path]] = None,
        default_sun_elevation: float = 45.0,
        default_sun_azimuth: float = 135.0,
        default_pixel_size: float = 10.0,
    ) -> TileMetadata:
        """Load acquisition metadata from JSON file or apply defaults."""
        if meta_path is not None and Path(meta_path).is_file():
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return TileMetadata(
                sun_elevation_deg=float(data.get("sun_elevation_deg", default_sun_elevation)),
                sun_azimuth_deg=float(data.get("sun_azimuth_deg", default_sun_azimuth)),
                pixel_size_m=float(data.get("pixel_size_m", default_pixel_size)),
                tile_id=str(data.get("tile_id", "SENTINEL2_TILE")),
                acquisition_date=str(data.get("acquisition_date", "2026-05-15")),
                band_names=list(data.get("bands", DEFAULT_BAND_ORDER)),
            )

        return TileMetadata(
            sun_elevation_deg=default_sun_elevation,
            sun_azimuth_deg=default_sun_azimuth,
            pixel_size_m=default_pixel_size,
        )
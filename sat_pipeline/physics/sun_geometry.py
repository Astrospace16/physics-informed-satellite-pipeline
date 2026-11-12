"""Sun geometry and cloud shadow projection calculations."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import ndimage


def calculate_shadow_offset_m(cloud_height_m: float, sun_elevation_deg: float) -> float:
    """Calculate the geometric horizontal ground shadow offset in meters.

    Formula:
        offset_m = cloud_height_m / tan(sun_elevation_deg)

    Args:
        cloud_height_m: Height of the cloud above ground in meters.
        sun_elevation_deg: Solar elevation angle above the horizon in degrees (0, 90].

    Returns:
        Horizontal offset in meters along the solar ray ground projection.
    """
    if sun_elevation_deg <= 0.0 or sun_elevation_deg > 90.0:
        raise ValueError(f"sun_elevation_deg must be in (0, 90], got {sun_elevation_deg}")
    if cloud_height_m < 0.0:
        raise ValueError(f"cloud_height_m cannot be negative, got {cloud_height_m}")

    elev_rad = np.radians(sun_elevation_deg)
    return float(cloud_height_m / np.tan(elev_rad))


def calculate_shadow_shift_pixels(
    offset_m: float,
    sun_azimuth_deg: float,
    pixel_size_m: float = 10.0,
    north_is_up: bool = True,
) -> Tuple[float, float]:
    """Calculate the (row_shift, col_shift) in raster pixel coordinates.

    Rule:
        Slide the cloud mask away from the sun by (offset_m / pixel_size_m)
        pixels using sun_azimuth_deg + 180 degrees.

    Geospatial coordinate mapping:
        - Azimuth is measured clockwise from North (0° = North, 90° = East).
        - Shadow direction = (sun_azimuth_deg + 180°) % 360°.
        - In a standard raster where North is up (row 0 at top):
            row increases southward -> row_shift = -offset_pixels * cos(shadow_azimuth)
            col increases eastward  -> col_shift = offset_pixels * sin(shadow_azimuth)

    Returns:
        (shift_row, shift_col) in pixels.
    """
    if pixel_size_m <= 0:
        raise ValueError(f"pixel_size_m must be positive, got {pixel_size_m}")

    shadow_azimuth_deg = (sun_azimuth_deg + 180.0) % 360.0
    shadow_azimuth_rad = np.radians(shadow_azimuth_deg)
    offset_pixels = offset_m / pixel_size_m

    # Trigonometric projection onto raster grid
    shift_col = offset_pixels * np.sin(shadow_azimuth_rad)
    shift_row = -offset_pixels * np.cos(shadow_azimuth_rad) if north_is_up else offset_pixels * np.cos(shadow_azimuth_rad)

    return float(shift_row), float(shift_col)


def project_cloud_mask(
    cloud_mask: np.ndarray,
    shift_row: float,
    shift_col: float,
    threshold: float = 0.5,
) -> np.ndarray:
    """Slide a binary cloud mask by (shift_row, shift_col) pixels using sub-pixel interpolation.

    Pixels shifted outside the image boundaries are filled with 0.
    """
    if cloud_mask.ndim != 2:
        raise ValueError(f"cloud_mask must be 2D, got shape {cloud_mask.shape}")

    shifted = ndimage.shift(
        cloud_mask.astype(np.float32),
        shift=(shift_row, shift_col),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return (shifted >= threshold).astype(np.uint8)


def project_cloud_shadow(
    cloud_mask: np.ndarray,
    cloud_height_m: float,
    sun_elevation_deg: float,
    sun_azimuth_deg: float,
    pixel_size_m: float = 10.0,
    north_is_up: bool = True,
) -> np.ndarray:
    """Project ground cloud shadow for a specified cloud height and sun angles.

    1. offset_m = cloud_height_m / tan(sun_elevation_deg)
    2. Slide cloud mask away from sun by (offset_m / pixel_size_m) pixels
       using sun_azimuth_deg + 180 degrees.
    """
    offset_m = calculate_shadow_offset_m(cloud_height_m, sun_elevation_deg)
    shift_row, shift_col = calculate_shadow_shift_pixels(
        offset_m=offset_m,
        sun_azimuth_deg=sun_azimuth_deg,
        pixel_size_m=pixel_size_m,
        north_is_up=north_is_up,
    )
    return project_cloud_mask(cloud_mask, shift_row, shift_col)


def estimate_cloud_height(
    cloud_mask: np.ndarray,
    brightness: np.ndarray,
    sun_elevation_deg: float,
    sun_azimuth_deg: float,
    pixel_size_m: float = 10.0,
    min_height_m: float = 200.0,
    max_height_m: float = 2000.0,
    step_m: float = 100.0,
    dark_threshold: float = 0.15,
    north_is_up: bool = True,
) -> Tuple[float, np.ndarray, Dict[float, int]]:
    """Search candidate cloud heights 200 m to 2000 m in 100 m steps and pick the height

    whose projected shadow maximises overlap with dark pixels (brightness < 0.15).

    Args:
        cloud_mask: Binary 2D array indicating cloud presence.
        brightness: 2D array of normalized surface/visible brightness [0, 1].
        sun_elevation_deg: Solar elevation angle in degrees.
        sun_azimuth_deg: Solar azimuth angle in degrees.
        pixel_size_m: Ground sampling distance in meters.
        min_height_m: Lower search bound in meters (default 200m).
        max_height_m: Upper search bound in meters (default 2000m).
        step_m: Search step in meters (default 100m).
        dark_threshold: Darkness brightness threshold (default 0.15).
        north_is_up: Raster orientation.

    Returns:
        best_height_m: Estimated cloud height maximizing shadow-darkness overlap.
        best_projected_shadow: Binary 2D mask of predicted shadow at best height.
        overlap_scores: Mapping from candidate height to overlapping dark pixel count.
    """
    if cloud_mask.shape != brightness.shape:
        raise ValueError(
            f"Shape mismatch: cloud_mask {cloud_mask.shape} vs brightness {brightness.shape}"
        )

    dark_pixels = (brightness < dark_threshold).astype(np.uint8)
    # A pixel cannot be its own ground shadow if it is already covered by cloud
    valid_ground = (cloud_mask == 0)
    dark_ground = (dark_pixels & valid_ground).astype(np.uint8)

    candidates = np.arange(min_height_m, max_height_m + (step_m / 2.0), step_m)
    overlap_scores: Dict[float, int] = {}

    best_height = float(candidates[0])
    max_overlap = -1
    best_shadow = np.zeros_like(cloud_mask, dtype=np.uint8)

    cloud_pixel_count = int(np.sum(cloud_mask > 0))
    if cloud_pixel_count == 0:
        # No clouds detected in scene; return minimum height and zero shadow mask
        for h in candidates:
            overlap_scores[float(h)] = 0
        return float(min_height_m), best_shadow, overlap_scores

    for h in candidates:
        h_float = float(h)
        projected = project_cloud_shadow(
            cloud_mask=cloud_mask,
            cloud_height_m=h_float,
            sun_elevation_deg=sun_elevation_deg,
            sun_azimuth_deg=sun_azimuth_deg,
            pixel_size_m=pixel_size_m,
            north_is_up=north_is_up,
        )
        # Exclude regions occluded by the cloud itself
        projected_on_ground = (projected & valid_ground).astype(np.uint8)
        overlap = int(np.sum((projected_on_ground & dark_ground) > 0))
        overlap_scores[h_float] = overlap

        if overlap > max_overlap:
            max_overlap = overlap
            best_height = h_float
            best_shadow = projected_on_ground

    return best_height, best_shadow, overlap_scores
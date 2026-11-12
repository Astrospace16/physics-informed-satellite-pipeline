"""DEM gradient, slope, aspect, and hillshade illumination calculations."""

from typing import Tuple
import numpy as np
from scipy import ndimage


def compute_dem_gradients(
    dem: np.ndarray,
    pixel_size_m: float = 10.0,
    north_is_up: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute spatial elevation gradients (dz_dx, dz_dy) using Horn's 3x3 kernel.

    Args:
        dem: 2D array of elevations in meters (H, W).
        pixel_size_m: Ground sampling distance in meters.
        north_is_up: True if raster row 0 corresponds to North.

    Returns:
        dz_dx: Elevation gradient in East-West direction (positive = elevation increases East).
        dz_dy: Elevation gradient in North-South direction (positive = elevation increases North).
    """
    if dem.ndim != 2:
        raise ValueError(f"dem must be a 2D array, got shape {dem.shape}")
    if pixel_size_m <= 0:
        raise ValueError(f"pixel_size_m must be positive, got {pixel_size_m}")

    dem_f = dem.astype(np.float32)

    # Horn's 3x3 kernels
    # East-West kernel: [-1, 0, 1] weighting
    kernel_x = np.array([
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0],
    ], dtype=np.float32) / (8.0 * pixel_size_m)

    # North-South kernel: row 0 is top (North). Moving south (increasing row) decreases y_geo.
    # Therefore, dz/dy_north = (north - south) / dy = (row_minus - row_plus) / (2 * dy)
    if north_is_up:
        kernel_y = np.array([
            [1.0, 2.0, 1.0],
            [0.0, 0.0, 0.0],
            [-1.0, -2.0, -1.0],
        ], dtype=np.float32) / (8.0 * pixel_size_m)
    else:
        kernel_y = -np.array([
            [1.0, 2.0, 1.0],
            [0.0, 0.0, 0.0],
            [-1.0, -2.0, -1.0],
        ], dtype=np.float32) / (8.0 * pixel_size_m)

    dz_dx = ndimage.convolve(dem_f, kernel_x, mode="nearest")
    dz_dy = ndimage.convolve(dem_f, kernel_y, mode="nearest")

    return dz_dx, dz_dy


def compute_slope_and_aspect(
    dz_dx: np.ndarray,
    dz_dy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute terrain slope and aspect in radians.

    Slope:
        slope = arctan(sqrt(dz_dx^2 + dz_dy^2))
        Range: [0, pi/2] radians (0 = horizontal flat).

    Aspect:
        Direction of maximum slope, measured clockwise from North (0 rad = North, pi/2 = East).
        Range: [0, 2*pi) radians.
    """
    # Slope calculation
    rise_run = np.sqrt(dz_dx**2 + dz_dy**2)
    slope = np.arctan(rise_run)

    # Aspect calculation in geographic azimuth (clockwise from North)
    # dx is East (+x), dy is North (+y).
    # Geographic aspect is 0 at North, pi/2 at East.
    # Standard formula: aspect = arctan2(dz_dx, dz_dy)
    aspect = np.arctan2(dz_dx, dz_dy)
    # Wrap negative angles to [0, 2*pi)
    aspect = np.where(aspect < 0.0, aspect + 2.0 * np.pi, aspect)

    return slope, aspect


def compute_solar_illumination(
    slope: np.ndarray,
    aspect: np.ndarray,
    sun_elevation_deg: float,
    sun_azimuth_deg: float,
) -> np.ndarray:
    """Compute cosine of solar incidence angle (direct solar illumination).

    Formula:
        zenith = 90 deg - sun_elevation_deg
        cos(i) = cos(zenith) * cos(slope) + sin(zenith) * sin(slope) * cos(sun_azimuth - aspect)

    Values <= 0 represent self-shadowed slopes (facing away from the sun).
    Values in [0, 1] represent degree of direct solar exposure.
    """
    sun_zenith_deg = 90.0 - sun_elevation_deg
    zenith_rad = np.radians(sun_zenith_deg)
    sun_azimuth_rad = np.radians(sun_azimuth_deg)

    # Cosine of incidence angle
    cos_i = (
        np.cos(zenith_rad) * np.cos(slope)
        + np.sin(zenith_rad) * np.sin(slope) * np.cos(sun_azimuth_rad - aspect)
    )

    # Direct solar illumination clipped to [0, 1]
    illumination = np.clip(cos_i, 0.0, 1.0)
    return illumination


def compute_hillshade_mask(
    dem: np.ndarray,
    sun_elevation_deg: float,
    sun_azimuth_deg: float,
    pixel_size_m: float = 10.0,
    illumination_threshold: float = 0.10,
    north_is_up: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the hillshade mask from DEM gradient and illumination threshold.

    Rule:
        Compute from DEM gradient (slope + aspect) and a threshold on illumination;
        use it to rule out terrain darkness.

    Args:
        dem: 2D array of elevations in meters.
        sun_elevation_deg: Sun elevation angle in degrees.
        sun_azimuth_deg: Sun azimuth angle in degrees.
        pixel_size_m: Pixel size in meters.
        illumination_threshold: Threshold below which terrain is shadowed (default 0.10).
        north_is_up: True if row 0 is North.

    Returns:
        hillshade_mask: Binary 2D array (1 where terrain is shadowed, 0 elsewhere).
        illumination: 2D array of normalized illumination values [0, 1].
        slope: 2D array of slope in radians.
        aspect: 2D array of aspect in radians.
    """
    dz_dx, dz_dy = compute_dem_gradients(dem, pixel_size_m, north_is_up=north_is_up)
    slope, aspect = compute_slope_and_aspect(dz_dx, dz_dy)
    illumination = compute_solar_illumination(
        slope=slope,
        aspect=aspect,
        sun_elevation_deg=sun_elevation_deg,
        sun_azimuth_deg=sun_azimuth_deg,
    )

    # Terrain is shadowed if direct solar illumination is below threshold
    # and terrain has non-zero slope (flat areas are illuminated unless occluded)
    hillshade_mask = (illumination < illumination_threshold).astype(np.uint8)

    return hillshade_mask, illumination, slope, aspect
"""Radiative-transfer approximations, illumination corrections, and sensor noise filtering."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import ndimage


def apply_solar_zenith_correction(
    reflectance: np.ndarray,
    sun_elevation_deg: float,
    reference_elevation_deg: float = 45.0,
) -> np.ndarray:
    """Normalize surface reflectance for solar zenith angle relative to a reference sun elevation.

    At the reference sun elevation (default 45°), the scaling factor is 1.0 so calibrated
    L2A reflectances retain their native dynamic range, while low-sun scenes are boosted
    and high-sun scenes are normalized:
        zenith = 90° - sun_elevation_deg
        ref_zenith = 90° - reference_elevation_deg
        rho_norm = rho * (cos(ref_zenith) / max(cos(zenith), 0.15))
    """
    zenith_deg = 90.0 - sun_elevation_deg
    ref_zenith_deg = 90.0 - reference_elevation_deg
    cos_zenith = max(float(np.cos(np.radians(zenith_deg))), 0.15)
    cos_ref = float(np.cos(np.radians(ref_zenith_deg)))
    scale = cos_ref / cos_zenith
    return np.clip(reflectance.astype(np.float32) * scale, 0.0, 1.0)


def apply_dark_object_subtraction(
    band: np.ndarray,
    percentile: float = 0.5,
    max_haze_removal: float = 0.03,
) -> np.ndarray:
    """Approximate atmospheric path radiance subtraction (Chavez DOS-1).

    Estimates the dark-object path radiance from the specified low percentile (capped at
    `max_haze_removal` so genuinely dark water/shadow signals are not zeroed out) and
    subtracts the excess atmospheric haze floor.
    """
    raw_dark = float(np.percentile(band, percentile))
    haze_offset = float(np.clip(raw_dark - 0.015, 0.0, max_haze_removal))
    if haze_offset <= 1e-6:
        return np.clip(band.astype(np.float32), 0.0, 1.0)
    corrected = np.maximum(band.astype(np.float32) - haze_offset, 0.0) / (1.0 - haze_offset + 1e-6)
    return np.clip(corrected, 0.0, 1.0)


def apply_c_correction_topography(
    band: np.ndarray,
    illumination: np.ndarray,
    sun_elevation_deg: float,
    slope_rad: Optional[np.ndarray] = None,
    hillshade_illumination_threshold: float = 0.10,
) -> np.ndarray:
    """Apply Teillet C-correction for topographic illumination normalization on illuminated slopes.

    Normalizes moderate terrain shading while preserving deep self-shadowed slopes
    (`illumination < hillshade_illumination_threshold`) so Rule 2 hillshade detection
    and Guardrail 5 remain physically intact.
    """
    zenith_rad = np.radians(90.0 - sun_elevation_deg)
    cos_zenith = float(np.cos(zenith_rad))

    # Empirical diffuse sky-irradiance C-parameter prevents over-correction on slopes
    c_param = 0.65
    denom = np.maximum(illumination.astype(np.float32) + c_param, 0.25)
    factor = (cos_zenith + c_param) / denom
    # Bound correction factor to prevent artificial blowout over cloud threshold
    factor = np.clip(factor, 0.85, 1.15)

    active_mask = illumination >= hillshade_illumination_threshold
    if slope_rad is not None:
        active_mask = active_mask & (slope_rad > 0.02)

    corrected = np.where(active_mask, band.astype(np.float32) * factor, band.astype(np.float32))
    return np.clip(corrected, 0.0, 1.0)


def detect_and_filter_sensor_noise(
    bands: Dict[str, np.ndarray],
    zscore_threshold: float = 3.5,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Detect and repair single-pixel impulse / salt-and-pepper sensor noise across bands.

    Computes a 3x3 spatial median filter and local Median Absolute Deviation (MAD) per band.
    Pixels deviating from their 3x3 neighborhood median by more than `zscore_threshold`
    robust standard deviations (and at least 0.08 absolute reflectance) are flagged as
    sensor speckle/dropouts and replaced with the 3x3 median.

    Returns:
        filtered_bands: Dictionary of despeckled reflectance arrays.
        noise_mask: 2D uint8 mask where 1 indicates a pixel repaired in at least one band.
    """
    first_band = next(iter(bands.values()))
    h, w = first_band.shape
    noise_mask = np.zeros((h, w), dtype=np.uint8)
    filtered_bands: Dict[str, np.ndarray] = {}

    for bname, arr in bands.items():
        arr_f = arr.astype(np.float32)
        if bname == "B10":
            med = ndimage.median_filter(arr_f, size=3, mode="nearest")
            diff = np.abs(arr_f - med)
            local_mad = ndimage.median_filter(diff, size=3, mode="nearest") + 1e-4
            robust_z = diff / (1.4826 * local_mad)
            outlier = (robust_z > (zscore_threshold + 1.0)) & (diff > 0.025)
        else:
            med = ndimage.median_filter(arr_f, size=3, mode="nearest")
            diff = np.abs(arr_f - med)
            local_mad = ndimage.median_filter(diff, size=3, mode="nearest") + 1e-3
            robust_z = diff / (1.4826 * local_mad)
            outlier = (robust_z > zscore_threshold) & (diff > 0.08)

        repaired = np.where(outlier, med, arr_f)
        filtered_bands[bname] = np.clip(repaired, 0.0, 1.0).astype(np.float32)
        noise_mask = (noise_mask | outlier.astype(np.uint8)).astype(np.uint8)

    return filtered_bands, noise_mask


def run_radiative_transfer_pipeline(
    bands: Dict[str, np.ndarray],
    sun_elevation_deg: float,
    illumination: np.ndarray,
    slope_rad: Optional[np.ndarray] = None,
    enable_noise_filtering: bool = True,
    enable_dos: bool = True,
    enable_solar_zenith: bool = True,
    enable_topographic: bool = True,
    dos_percentile: float = 0.5,
    noise_zscore_threshold: float = 3.5,
    hillshade_illumination_threshold: float = 0.10,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[str]]:
    """Run end-to-end sensor noise filtering and radiative transfer corrections on all bands."""
    applied_steps: List[str] = []
    first_band = next(iter(bands.values()))
    noise_mask = np.zeros(first_band.shape, dtype=np.uint8)

    working = {k: v.astype(np.float32).copy() for k, v in bands.items()}

    if enable_noise_filtering:
        working, noise_mask = detect_and_filter_sensor_noise(
            working, zscore_threshold=noise_zscore_threshold
        )
        applied_steps.append("3x3 MAD Impulse Noise Filtering")

    if enable_dos:
        for bname, arr in working.items():
            if bname != "B10":
                working[bname] = apply_dark_object_subtraction(arr, percentile=dos_percentile)
        applied_steps.append("Chavez DOS-1 Atmospheric Path Radiance Subtraction")

    if enable_solar_zenith:
        for bname, arr in working.items():
            if bname != "B10":
                working[bname] = apply_solar_zenith_correction(arr, sun_elevation_deg=sun_elevation_deg)
        applied_steps.append("Solar Zenith Cosine Normalization")

    if enable_topographic:
        for bname, arr in working.items():
            if bname != "B10":
                working[bname] = apply_c_correction_topography(
                    band=arr,
                    illumination=illumination,
                    sun_elevation_deg=sun_elevation_deg,
                    slope_rad=slope_rad,
                    hillshade_illumination_threshold=hillshade_illumination_threshold,
                )
        applied_steps.append("Teillet Topographic C-Correction")

    return working, noise_mask, applied_steps
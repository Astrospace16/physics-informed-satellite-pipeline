"""Spectral indices, seasonal phenology baselines, and physics-based disturbance checks."""

from typing import Dict, Optional, Tuple
import numpy as np


def compute_normalized_difference(
    band_a: np.ndarray,
    band_b: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute normalized difference index: (band_a - band_b) / (band_a + band_b).

    Clipped to valid physical range [-1.0, 1.0].
    """
    numerator = band_a.astype(np.float32) - band_b.astype(np.float32)
    denominator = band_a.astype(np.float32) + band_b.astype(np.float32) + eps
    index = numerator / denominator
    return np.clip(index, -1.0, 1.0)


def compute_ndsi(b3_green: np.ndarray, b11_swir1: np.ndarray) -> np.ndarray:
    """Normalized Difference Snow Index (NDSI).

    Formula:
        NDSI = (B3 - B11) / (B3 + B11)
    """
    return compute_normalized_difference(b3_green, b11_swir1)


def compute_ndwi(b3_green: np.ndarray, b8_nir: np.ndarray) -> np.ndarray:
    """Normalized Difference Water Index (NDWI) - McFeeters.

    Formula:
        NDWI = (B3 - B8) / (B3 + B8)
    """
    return compute_normalized_difference(b3_green, b8_nir)


def compute_ndvi(b8_nir: np.ndarray, b4_red: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index (NDVI).

    Formula:
        NDVI = (B8 - B4) / (B8 + B4)
    """
    return compute_normalized_difference(b8_nir, b4_red)


def compute_nbr(
    b8_nir: np.ndarray,
    b11_swir1: np.ndarray,
    b12_swir2: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Normalized Burn Ratio (NBR) for active fire / char scar detection.

    Uses B12 (SWIR-2) when present, or B11 (SWIR-1) in standard 6-band stacks:
        NBR = (B8 - B_SWIR) / (B8 + B_SWIR)
    """
    swir = b12_swir2 if b12_swir2 is not None else b11_swir1
    return compute_normalized_difference(b8_nir, swir)


def compute_ndmi(b8_nir: np.ndarray, b11_swir1: np.ndarray) -> np.ndarray:
    """Normalized Difference Moisture Index (NDMI) for canopy water content / drought stress.

    Formula:
        NDMI = (B8 - B11) / (B8 + B11)
    """
    return compute_normalized_difference(b8_nir, b11_swir1)


def compute_bsi(
    b2_blue: np.ndarray,
    b4_red: np.ndarray,
    b8_nir: np.ndarray,
    b11_swir1: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Bare Soil Index (BSI) for distinguishing bare soil / crop senescence from vegetation.

    Formula:
        BSI = ((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2))
    """
    num = (b11_swir1.astype(np.float32) + b4_red.astype(np.float32)) - (
        b8_nir.astype(np.float32) + b2_blue.astype(np.float32)
    )
    den = (
        b11_swir1.astype(np.float32)
        + b4_red.astype(np.float32)
        + b8_nir.astype(np.float32)
        + b2_blue.astype(np.float32)
        + eps
    )
    return np.clip(num / den, -1.0, 1.0)


def compute_brightness(
    b2_blue: np.ndarray,
    b3_green: np.ndarray,
    b4_red: np.ndarray,
) -> np.ndarray:
    """Calculate mean visible surface brightness in [0.0, 1.0]."""
    vis = (
        b2_blue.astype(np.float32)
        + b3_green.astype(np.float32)
        + b4_red.astype(np.float32)
    ) / 3.0
    return np.clip(vis, 0.0, 1.0)


def compute_seasonal_phenology(
    day_of_year: int,
    latitude_deg: float = 30.0,
    base_ndsi_snow_threshold: float = 0.40,
) -> Tuple[float, float]:
    """Compute seasonally modulated vegetation NDVI baseline and snow NDSI threshold.

    Models the annual harmonic phenology cycle:
        phase = 2 * pi * (doy - peak_doy) / 365.25
        expected_ndvi = 0.42 + 0.18 * cos(phase)
    where peak_doy = 172 (June 21) in the Northern Hemisphere and 355 in the Southern Hemisphere.
    """
    peak_doy = 172.0 if latitude_deg >= 0.0 else 355.0
    phase = 2.0 * np.pi * (float(day_of_year) - peak_doy) / 365.25
    expected_ndvi = float(np.clip(0.42 + 0.18 * np.cos(phase), 0.24, 0.60))
    return expected_ndvi, float(base_ndsi_snow_threshold)


def detect_spectral_snow(
    b3_green: np.ndarray,
    b11_swir1: np.ndarray,
    ndsi_threshold: float = 0.40,
    b2_blue: np.ndarray = None,
    b2_min_threshold: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect snow based on physics spectral rule.

    Rule:
        NDSI = (B3 - B11) / (B3 + B11) -> snow if NDSI > 0.4
    """
    ndsi = compute_ndsi(b3_green, b11_swir1)
    snow_mask = ndsi > ndsi_threshold

    if b2_blue is not None:
        snow_mask = snow_mask & (b2_blue > b2_min_threshold)

    return snow_mask.astype(np.uint8), ndsi


def detect_spectral_water(
    b3_green: np.ndarray,
    b8_nir: np.ndarray,
    ndwi_threshold: float = 0.30,
    b8_threshold: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect water bodies based on physics spectral rule.

    Rule:
        NDWI = (B3 - B8) / (B3 + B8) -> water if NDWI > 0.3 and B8 < 0.1
    """
    ndwi = compute_ndwi(b3_green, b8_nir)
    water_mask = (ndwi > ndwi_threshold) & (b8_nir < b8_threshold)
    return water_mask.astype(np.uint8), ndwi


def detect_spectral_clouds(
    b2_blue: np.ndarray,
    b3_green: np.ndarray,
    b4_red: np.ndarray,
    b8_nir: np.ndarray,
    b10_cirrus: np.ndarray,
    b11_swir1: np.ndarray,
    cirrus_b10_threshold: float = 0.03,
    visible_brightness_threshold: float = 0.20,
    ndsi_snow_threshold: float = 0.40,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect opaque and cirrus clouds using spectral criteria.

    Rules:
        - Cirrus: B10 > 0.03 combined with high visible brightness -> cloud
        - Opaque cloud: High visible and NIR reflectance, not snow (NDSI <= 0.4).
    """
    brightness = compute_brightness(b2_blue, b3_green, b4_red)
    ndsi = compute_ndsi(b3_green, b11_swir1)

    cirrus_detected = (b10_cirrus > cirrus_b10_threshold) & (
        brightness > visible_brightness_threshold
    )

    opaque_clouds = (
        (brightness > visible_brightness_threshold)
        & (b8_nir > visible_brightness_threshold)
        & (ndsi <= ndsi_snow_threshold)
    )

    combined_cloud_mask = (cirrus_detected | opaque_clouds).astype(np.uint8)
    return combined_cloud_mask, cirrus_detected.astype(np.uint8), brightness


def detect_land_change_disturbances(
    b2_blue: np.ndarray,
    b3_green: np.ndarray,
    b4_red: np.ndarray,
    b8_nir: np.ndarray,
    b11_swir1: np.ndarray,
    ndwi: np.ndarray,
    ndvi: np.ndarray,
    brightness: np.ndarray,
    cloud_mask: np.ndarray,
    predicted_shadow_mask: np.ndarray,
    hillshade_mask: np.ndarray,
    permanent_water_mask: np.ndarray,
    snow_mask: np.ndarray,
    seasonal_ndvi_baseline: float = 0.45,
    nbr_fire_threshold: float = -0.15,
    ndwi_flood_min: float = 0.10,
    crop_damage_ndvi_drop: float = 0.20,
    b12_swir2: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Actively detect physical land-change disturbances: Flood, Fire/Burn Scar, and Crop Damage.

    Operates on ground pixels not occluded by clouds, geometric cloud shadows, terrain
    hillshade, snow, or permanent clear water bodies:
      1. **Flood / Turbid Inundation (`flood_mask`)**:
         Standing water over land or turbid sediment-laden floodwater that is dark in NIR & SWIR
         (`B8 < 0.14` and `B11 < 0.09`), has positive water/moisture signature (`NDWI >= ndwi_flood_min`
         and `B3 > B4`), and is outside permanent clear water (`permanent_water_mask == 0`).
      2. **Fire / Burn Scar (`fire_mask`)**:
         Charcoal/ash deposition causes destroyed NIR cell structure (`B8 < 0.12`), elevated SWIR
         reflectance (`NBR < nbr_fire_threshold`, `B11 > B8`), and dark charcoal visible reflectance
         (`brightness < 0.09`).
      3. **Crop Damage / Severe Senescence (`crop_damage_mask`)**:
         Canopy collapse or severe stress relative to the seasonal phenology baseline:
         `NDVI < (seasonal_ndvi_baseline - crop_damage_ndvi_drop)` combined with high bare-soil /
         senescent SWIR response (`BSI > 0.08`, `B11 > 0.16`, `B4 > B3`), moderate visible brightness
         (`0.08 <= brightness <= 0.22`), and not flagged as fire or flood.
    """
    nbr = compute_nbr(b8_nir, b11_swir1, b12_swir2=b12_swir2)
    ndmi = compute_ndmi(b8_nir, b11_swir1)
    bsi = compute_bsi(b2_blue, b4_red, b8_nir, b11_swir1)

    unoccluded_land = (
        (cloud_mask == 0)
        & (predicted_shadow_mask == 0)
        & (hillshade_mask == 0)
        & (snow_mask == 0)
        & (permanent_water_mask == 0)
    )

    # 1. Flood / Turbid Surface Inundation
    flood_cond = (
        unoccluded_land
        & (ndwi >= ndwi_flood_min)
        & (b8_nir < 0.14)
        & (b11_swir1 < 0.09)
        & (b3_green > b4_red)
        & (brightness < 0.16)
    )
    flood_mask = flood_cond.astype(np.uint8)

    # 2. Fire / Burn Scar (Charcoal & Ash: dark visible < 0.09, destroyed NIR < 0.12, NBR < -0.15)
    fire_cond = (
        unoccluded_land
        & (~flood_cond)
        & (nbr < nbr_fire_threshold)
        & (b11_swir1 > b8_nir)
        & (b8_nir < 0.12)
        & (brightness < 0.09)
        & (ndwi < 0.05)
    )
    fire_mask = fire_cond.astype(np.uint8)

    # 3. Crop Damage / Severe Canopy Senescence (Seasonally Calibrated)
    ndvi_stress_ceiling = max(0.12, seasonal_ndvi_baseline - crop_damage_ndvi_drop)
    crop_cond = (
        unoccluded_land
        & (~flood_cond)
        & (~fire_cond)
        & (ndvi < ndvi_stress_ceiling)
        & (bsi > 0.08)
        & (b11_swir1 > 0.16)
        & (b4_red > b3_green * 1.05)
        & (brightness >= 0.07)
        & (brightness <= 0.22)
    )
    crop_damage_mask = crop_cond.astype(np.uint8)

    return flood_mask, fire_mask, crop_damage_mask, nbr, ndmi, bsi
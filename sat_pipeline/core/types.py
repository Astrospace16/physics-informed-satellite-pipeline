"""Data structures, dataclasses, and configurations for the pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
import json
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class TileMetadata:
    """Metadata describing a satellite tile and its acquisition conditions."""
    sun_elevation_deg: float
    sun_azimuth_deg: float
    pixel_size_m: float = 10.0
    tile_id: str = "SENTINEL2_TILE"
    acquisition_date: str = "2026-05-15"
    latitude_deg: float = 30.0
    crs: str = "EPSG:32632"
    bounds: List[float] = field(default_factory=lambda: [0.0, 0.0, 1000.0, 1000.0])
    band_names: List[str] = field(default_factory=lambda: ["B2", "B3", "B4", "B8", "B11", "B10"])

    def validate(self) -> None:
        if not (0.0 < self.sun_elevation_deg <= 90.0):
            raise ValueError(f"Sun elevation must be in (0, 90] degrees, got {self.sun_elevation_deg}")
        if not (0.0 <= self.sun_azimuth_deg < 360.0):
            self.sun_azimuth_deg = self.sun_azimuth_deg % 360.0
        if self.pixel_size_m <= 0:
            raise ValueError(f"pixel_size_m must be positive, got {self.pixel_size_m}")

    def day_of_year(self) -> int:
        """Parse acquisition_date (YYYY-MM-DD) into day of year [1..366]."""
        try:
            dt = datetime.strptime(self.acquisition_date.strip()[:10], "%Y-%m-%d")
            return int(dt.timetuple().tm_yday)
        except Exception:
            return 135  # Default mid-May (DOY 135)

    def season_name(self) -> str:
        """Determine meteorological season from day of year and hemisphere."""
        doy = self.day_of_year()
        northern = self.latitude_deg >= 0.0
        if 60 <= doy < 152:
            return "spring" if northern else "autumn"
        elif 152 <= doy < 244:
            return "summer" if northern else "winter"
        elif 244 <= doy < 335:
            return "autumn" if northern else "spring"
        else:
            return "winter" if northern else "summer"


@dataclass
class PhysicsFeatures:
    """Intermediate physics-derived features and masks."""
    predicted_shadow_mask: np.ndarray  # (H, W) binary/float
    hillshade_mask: np.ndarray         # (H, W) binary
    ndsi: np.ndarray                   # (H, W) float [-1, 1]
    ndwi: np.ndarray                   # (H, W) float [-1, 1]
    ndvi: np.ndarray                   # (H, W) float [-1, 1]
    brightness: np.ndarray             # (H, W) float [0, 1]
    cirrus_mask: np.ndarray            # (H, W) binary
    spectral_cloud_mask: np.ndarray    # (H, W) binary
    spectral_water_mask: np.ndarray    # (H, W) binary
    spectral_snow_mask: np.ndarray     # (H, W) binary
    estimated_cloud_height_m: float
    shadow_overlap_score: float
    illumination: np.ndarray           # (H, W) float [0, 1]
    slope_rad: np.ndarray              # (H, W) float
    aspect_rad: np.ndarray             # (H, W) float
    # Radiative transfer, noise, season & disturbance fields (with defaults for backward compat)
    nbr: Optional[np.ndarray] = None                # (H, W) float [-1, 1] Normalized Burn Ratio
    ndmi: Optional[np.ndarray] = None               # (H, W) float [-1, 1] Normalized Difference Moisture Index
    bsi: Optional[np.ndarray] = None                # (H, W) float [-1, 1] Bare Soil Index
    flood_mask: Optional[np.ndarray] = None         # (H, W) uint8 active flood / turbid inundation
    fire_mask: Optional[np.ndarray] = None          # (H, W) uint8 active burn scar / char
    crop_damage_mask: Optional[np.ndarray] = None   # (H, W) uint8 vegetation / crop senescence stress
    noise_mask: Optional[np.ndarray] = None         # (H, W) uint8 impulse / speckle pixels repaired
    seasonal_ndvi_baseline: float = 0.45
    season: str = "spring"
    radiative_corrected_bands: Optional[Dict[str, np.ndarray]] = None


@dataclass
class TileChangeLog:
    """Tile-level change log and diagnostic summary."""
    total_pixels: int
    class_counts: Dict[str, int]
    class_percentages: Dict[str, float]
    overrides: Dict[str, int]
    mean_trust: float
    min_trust: float
    max_trust: float
    estimated_cloud_height_m: float
    shadow_overlap_metric: float
    land_change_breakdown: Dict[str, int] = field(default_factory=dict)
    season: str = "spring"
    seasonal_ndvi_baseline: float = 0.45
    noise_repaired_pixels: int = 0
    radiative_corrections_applied: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_pixels": self.total_pixels,
            "season": self.season,
            "seasonal_ndvi_baseline": round(float(self.seasonal_ndvi_baseline), 4),
            "radiative_corrections_applied": self.radiative_corrections_applied,
            "noise_repaired_pixels": int(self.noise_repaired_pixels),
            "estimated_cloud_height_m": float(self.estimated_cloud_height_m),
            "shadow_overlap_metric": float(self.shadow_overlap_metric),
            "mean_trust": round(float(self.mean_trust), 4),
            "min_trust": round(float(self.min_trust), 4),
            "max_trust": round(float(self.max_trust), 4),
            "class_counts": self.class_counts,
            "class_percentages": {k: round(v, 2) for k, v in self.class_percentages.items()},
            "land_change_breakdown": self.land_change_breakdown,
            "guardrail_overrides": self.overrides,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Tile Processing Change Log",
            f"- **Total Pixels**: {self.total_pixels:,}",
            f"- **Season & Phenology Baseline**: `{self.season}` (Expected Healthy Crop NDVI: {self.seasonal_ndvi_baseline:.2f})",
            f"- **Radiative & Noise Corrections**: {', '.join(self.radiative_corrections_applied) if self.radiative_corrections_applied else 'None'} ({self.noise_repaired_pixels:,} impulse noise pixels repaired)",
            f"- **Estimated Cloud Height**: {self.estimated_cloud_height_m:.1f} m",
            f"- **Shadow-Darkness Overlap Metric**: {self.shadow_overlap_metric:.4f}",
            f"- **Trust Scores**: Mean: {self.mean_trust:.3f} | Min: {self.min_trust:.3f} | Max: {self.max_trust:.3f}",
            "",
            "## Class Breakdown",
            "| Class | Pixel Count | Percentage (%) |",
            "| :--- | :--- | :--- |",
        ]
        for cname, count in sorted(self.class_counts.items(), key=lambda x: -x[1]):
            pct = self.class_percentages.get(cname, 0.0)
            lines.append(f"| `{cname}` | {count:,} | {pct:.2f}% |")

        if self.land_change_breakdown:
            lines.extend([
                "",
                "## Land Change Attribution Breakdown",
                "| Disturbance Subtype | Pixel Count |",
                "| :--- | :--- |",
            ])
            for stype, scount in self.land_change_breakdown.items():
                lines.append(f"| `{stype}` | {scount:,} |")

        lines.extend([
            "",
            "## Physics Guardrail Overrides",
            "| Guardrail Rule | Overridden Pixels |",
            "| :--- | :--- |",
        ])
        for rule, count in self.overrides.items():
            lines.append(f"| {rule} | {count:,} |")

        return "\n".join(lines)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class PipelineConfig:
    """Hyperparameters and thresholds for the physics-informed pipeline."""
    # Model Selection
    model_type: str = "rule_based"  # "rule_based", "random_forest", "unet"
    model_path: Optional[str] = None

    # Radiative Transfer & Sensor Noise Pre-Processing
    enable_radiative_transfer: bool = True
    enable_dos_correction: bool = True
    enable_solar_zenith_correction: bool = True
    enable_topographic_correction: bool = True
    enable_noise_filtering: bool = True
    dos_percentile: float = 0.5
    noise_zscore_threshold: float = 3.5

    # Seasonal Modulation
    enable_seasonal_modulation: bool = True

    # Physics Shadow Search
    min_cloud_height_m: float = 200.0
    max_cloud_height_m: float = 2000.0
    cloud_height_step_m: float = 100.0
    brightness_dark_threshold: float = 0.15

    # Physics Spectral Thresholds
    ndsi_snow_threshold: float = 0.40
    ndwi_water_threshold: float = 0.30
    water_b8_threshold: float = 0.10
    cirrus_b10_threshold: float = 0.03
    visible_brightness_threshold: float = 0.20

    # Land Change Disturbance Thresholds (Flood, Fire, Crop Damage)
    nbr_fire_threshold: float = -0.15
    ndwi_flood_min: float = 0.10
    crop_damage_ndvi_drop: float = 0.20

    # Physics Hillshade Thresholds
    hillshade_illumination_threshold: float = 0.10

    # Trust Guardrail Values
    trust_shadow_consensus: float = 0.95
    trust_hillshade: float = 0.90
    trust_land_change_verified: float = 0.88
    trust_land_change: float = 0.55
    trust_water_shadow: float = 0.85
    trust_snow_cloud: float = 0.80
    trust_spectral_high: float = 0.92
    trust_ai_standard: float = 0.75

    # Pixel Size
    default_pixel_size_m: float = 10.0


@dataclass
class PipelineOutputs:
    """Final output bundle for a processed tile."""
    labels: np.ndarray                    # (H, W) uint8 or int32 label IDs
    label_names: np.ndarray              # (H, W) object array of label strings
    trust_scores: np.ndarray              # (H, W) float32 [0.0, 1.0]
    reason_codes: np.ndarray              # (H, W) object array of reason strings
    probabilities: np.ndarray            # (H, W, num_classes) float32
    physics_features: PhysicsFeatures
    change_log: TileChangeLog
    metadata: TileMetadata
    land_change_subtypes: Optional[np.ndarray] = None  # (H, W) int32 LandChangeSubtype IDs
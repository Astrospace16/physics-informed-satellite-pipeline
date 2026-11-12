"""Core constants, label enumerations, band definitions, and reason codes."""

from enum import IntEnum, unique
from typing import Dict, List


@unique
class Label(IntEnum):
    """Core 7-class label enumeration for physics-informed satellite pipeline."""
    CLEAR = 0
    CLOUD = 1
    CLOUD_SHADOW = 2
    WATER = 3
    SNOW = 4
    HILLSHADE = 5
    LAND_CHANGE = 6

    @classmethod
    def from_string(cls, name: str) -> "Label":
        name_clean = name.strip().lower()
        mapping = {
            "clear": cls.CLEAR,
            "cloud": cls.CLOUD,
            "cloud_shadow": cls.CLOUD_SHADOW,
            "shadow": cls.CLOUD_SHADOW,
            "water": cls.WATER,
            "snow": cls.SNOW,
            "hillshade": cls.HILLSHADE,
            "land_change": cls.LAND_CHANGE,
            "landchange": cls.LAND_CHANGE,
        }
        if name_clean in mapping:
            return mapping[name_clean]
        raise ValueError(f"Unknown label name: {name}")

    def to_string(self) -> str:
        return self.name.lower()


@unique
class LandChangeSubtype(IntEnum):
    """Specific physical disturbance subtypes within the LAND_CHANGE class."""
    NONE = 0
    FLOOD_INUNDATION = 1
    FIRE_BURN_SCAR = 2
    CROP_DAMAGE_SENESCENCE = 3
    UNEXPLAINED_DARK_ANOMALY = 4


LAND_CHANGE_SUBTYPE_NAMES: Dict[int, str] = {
    LandChangeSubtype.NONE: "none",
    LandChangeSubtype.FLOOD_INUNDATION: "flood_inundation",
    LandChangeSubtype.FIRE_BURN_SCAR: "fire_burn_scar",
    LandChangeSubtype.CROP_DAMAGE_SENESCENCE: "crop_damage",
    LandChangeSubtype.UNEXPLAINED_DARK_ANOMALY: "unexplained_dark_anomaly",
}


LABEL_NAMES: Dict[int, str] = {
    Label.CLEAR: "clear",
    Label.CLOUD: "cloud",
    Label.CLOUD_SHADOW: "cloud_shadow",
    Label.WATER: "water",
    Label.SNOW: "snow",
    Label.HILLSHADE: "hillshade",
    Label.LAND_CHANGE: "land_change",
}

NUM_CLASSES: int = len(LABEL_NAMES)

# Sentinel-2 L2A Band Mappings
# Standard 6-band stack + optional B12 (SWIR-2) for active NBR burn-scar physics
BAND_BLUE = "B2"
BAND_GREEN = "B3"
BAND_RED = "B4"
BAND_NIR = "B8"
BAND_SWIR1 = "B11"
BAND_CIRRUS = "B10"
BAND_SWIR2 = "B12"

DEFAULT_BAND_ORDER: List[str] = [
    BAND_BLUE,
    BAND_GREEN,
    BAND_RED,
    BAND_NIR,
    BAND_SWIR1,
    BAND_CIRRUS,
]

# Plain-English Reason Codes
REASON_PHYSICS_AI_SHADOW_CONSENSUS = (
    "PHYSICS_AI_SHADOW_CONSENSUS: Geometric shadow projection confirmed by AI residual model"
)
REASON_GUARDRAIL_SHADOW_TO_LAND_CHANGE = (
    "GUARDRAIL_LAND_CHANGE: AI shadow rejected by geometric projection (unsupported by sun angle)"
)
REASON_LAND_CHANGE_FLOOD = (
    "LAND_CHANGE_FLOOD: Unshadowed inundation anomaly (elevated NDWI, dark NIR/SWIR, outside permanent water)"
)
REASON_LAND_CHANGE_FIRE = (
    "LAND_CHANGE_FIRE: Active burn scar / char signature (negative NBR, SWIR-1 > NIR, dark visible reflectance)"
)
REASON_LAND_CHANGE_CROP_DAMAGE = (
    "LAND_CHANGE_CROP_DAMAGE: Severe vegetation/crop stress (NDVI drop below seasonal baseline, elevated SWIR-1)"
)
REASON_GUARDRAIL_WATER_TO_SHADOW = (
    "GUARDRAIL_WATER_TO_SHADOW: Geometric shadow projection overrides water false positive"
)
REASON_GUARDRAIL_SNOW_TO_CLOUD = (
    "GUARDRAIL_SNOW_TO_CLOUD: Low NDSI (<= 0.4) contradicts AI snow prediction, relabeled as cloud"
)
REASON_GUARDRAIL_HILLSHADE = (
    "GUARDRAIL_HILLSHADE: Low solar illumination on terrain slope explains darkening"
)
REASON_PHYSICS_CLOUD_CIRRUS = (
    "PHYSICS_CLOUD_CIRRUS: High cirrus reflectance (B10 > 0.03) and visible brightness confirm cloud"
)
REASON_PHYSICS_CLOUD_SPECTRAL = (
    "PHYSICS_CLOUD_SPECTRAL: High visible/NIR brightness and cold spectral signature confirm cloud"
)
REASON_SPECTRAL_SNOW = (
    "SPECTRAL_SNOW: High NDSI (> 0.4) and visible reflectance confirm snow/ice"
)
REASON_SPECTRAL_WATER = (
    "SPECTRAL_WATER: High NDWI (> 0.3) and low NIR (< 0.1) confirm water body"
)
REASON_NOISE_FILTERED = (
    "NOISE_DESPECKLED: Isolated sensor noise / speckle suppressed via spatial median filter"
)
REASON_AI_CLEAR = (
    "AI_CLEAR: AI residual model classified as clear surface with high confidence"
)
REASON_AI_PREDICTION = (
    "AI_PREDICTION: Standard model classification verified within physical bounds"
)
REASON_CLEAR_BACKGROUND = (
    "CLEAR_BACKGROUND: Clear ground surface reflectance"
)

# Default Physical Thresholds
DEFAULT_DARK_BRIGHTNESS_THRESHOLD: float = 0.15
DEFAULT_NDSI_SNOW_THRESHOLD: float = 0.40
DEFAULT_NDWI_WATER_THRESHOLD: float = 0.30
DEFAULT_WATER_B8_THRESHOLD: float = 0.10
DEFAULT_CIRRUS_B10_THRESHOLD: float = 0.03
DEFAULT_VISIBLE_BRIGHTNESS_THRESHOLD: float = 0.20
DEFAULT_HILLSHADE_ILLUMINATION_THRESHOLD: float = 0.10

# Default Land Change Disturbance Thresholds
DEFAULT_NBR_FIRE_THRESHOLD: float = -0.15
DEFAULT_NDWI_FLOOD_MIN: float = 0.12
DEFAULT_CROP_DAMAGE_NDVI_DROP: float = 0.22

# Default Trust Values
TRUST_SHADOW_CONSENSUS: float = 0.95
TRUST_HILLSHADE_EXPLAINED: float = 0.90
TRUST_LAND_CHANGE_VERIFIED: float = 0.88
TRUST_LAND_CHANGE_FALLBACK: float = 0.55
TRUST_WATER_SHADOW_OVERRIDE: float = 0.85
TRUST_SNOW_CLOUD_OVERRIDE: float = 0.80
TRUST_STANDARD_PHYSICS_HIGH: float = 0.92
TRUST_AI_BASELINE: float = 0.75
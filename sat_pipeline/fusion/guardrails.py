"""Physics-bounded fusion engine enforcing guardrails, active disturbance detection, and trust calibration."""

from typing import Dict, Optional, Tuple
import numpy as np

from sat_pipeline.core.constants import (
    LABEL_NAMES,
    LAND_CHANGE_SUBTYPE_NAMES,
    REASON_AI_CLEAR,
    REASON_AI_PREDICTION,
    REASON_CLEAR_BACKGROUND,
    REASON_GUARDRAIL_HILLSHADE,
    REASON_GUARDRAIL_SHADOW_TO_LAND_CHANGE,
    REASON_GUARDRAIL_SNOW_TO_CLOUD,
    REASON_GUARDRAIL_WATER_TO_SHADOW,
    REASON_LAND_CHANGE_CROP_DAMAGE,
    REASON_LAND_CHANGE_FIRE,
    REASON_LAND_CHANGE_FLOOD,
    REASON_PHYSICS_AI_SHADOW_CONSENSUS,
    REASON_PHYSICS_CLOUD_CIRRUS,
    REASON_PHYSICS_CLOUD_SPECTRAL,
    REASON_SPECTRAL_SNOW,
    REASON_SPECTRAL_WATER,
    Label,
    LandChangeSubtype,
)
from sat_pipeline.core.types import PhysicsFeatures, PipelineConfig, TileChangeLog


class PhysicsBoundedFusion:
    """Combines AI residual predictions with physical constraints and active disturbance detectors.

    Enforces the 5 core pitch-deck guardrails AND actively attributes land_change pixels
    to Flood Inundation, Fire / Burn Scars, Crop Damage / Senescence, or Unexplained Dark Anomalies:
      1. If AI says shadow but geometric projection disagrees -> relabel as land_change with trust 0.55
         (or 0.88 if actively confirmed as flood, fire, or crop damage).
      2. If AI says water but a geometric shadow is predicted -> shadow (trust 0.85).
      3. If AI says snow but NDSI is low -> cloud (trust 0.80).
      4. If physics confirms shadow and AI agrees -> shadow (trust 0.95).
      5. If hillshade explains the darkening -> hillshade (trust 0.90).
      6. Active Land-Change Attribution: Pixels with physical Flood, Fire (NBR), or Crop Damage
         (seasonal NDVI drop + BSI) signatures on unoccluded land are classified as `land_change`
         with subtype-specific reason codes (`LAND_CHANGE_FLOOD`, `LAND_CHANGE_FIRE`,
         `LAND_CHANGE_CROP_DAMAGE`) and trust 0.88.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def fuse(
        self,
        ai_probabilities: np.ndarray,
        physics: PhysicsFeatures,
        radiative_steps: Optional[list] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TileChangeLog]:
        """Apply physics guardrails, active disturbance rules, and calibrate per-pixel trust scores.

        Args:
            ai_probabilities: 3D array (H, W, num_classes) of model probability distributions.
            physics: Extracted PhysicsFeatures dataclass.
            radiative_steps: Optional list of radiative transfer correction step names applied.

        Returns:
            final_labels: (H, W) int32 class IDs.
            label_names: (H, W) object array of class name strings.
            trust_scores: (H, W) float32 trust values in [0.0, 1.0].
            reason_codes: (H, W) object array of plain-English reason strings.
            change_log: TileChangeLog summary.
        """
        h, w, num_classes = ai_probabilities.shape
        total_pixels = h * w

        ai_labels = np.argmax(ai_probabilities, axis=-1).astype(np.int32)
        ai_confidence = np.max(ai_probabilities, axis=-1).astype(np.float32)

        # Output buffers
        final_labels = ai_labels.copy()
        trust_scores = np.full((h, w), self.config.trust_ai_standard, dtype=np.float32)
        reason_codes = np.full((h, w), REASON_AI_PREDICTION, dtype=object)

        # Physical condition masks
        geom_shadow = physics.predicted_shadow_mask > 0
        is_dark = physics.brightness < self.config.brightness_dark_threshold
        is_hillshade = physics.hillshade_mask > 0
        ndsi_low = physics.ndsi <= self.config.ndsi_snow_threshold
        is_cirrus = physics.cirrus_mask > 0
        is_spectral_snow = physics.spectral_snow_mask > 0
        is_spectral_water = physics.spectral_water_mask > 0
        is_spectral_cloud = physics.spectral_cloud_mask > 0

        is_flood = (
            (physics.flood_mask > 0)
            if physics.flood_mask is not None
            else np.zeros((h, w), dtype=bool)
        )
        is_fire = (
            (physics.fire_mask > 0)
            if physics.fire_mask is not None
            else np.zeros((h, w), dtype=bool)
        )
        is_crop_damage = (
            (physics.crop_damage_mask > 0)
            if physics.crop_damage_mask is not None
            else np.zeros((h, w), dtype=bool)
        )
        noise_repaired = (
            int(np.sum(physics.noise_mask > 0))
            if physics.noise_mask is not None
            else 0
        )

        # Tracking override counts for change log
        count_shadow_to_lc = 0
        count_water_to_shadow = 0
        count_snow_to_cloud = 0
        count_shadow_consensus = 0
        count_hillshade_override = 0
        count_cirrus_boost = 0
        count_active_disturbance = 0

        # 1. Base AI assignments with confidence mapping
        clear_mask = ai_labels == Label.CLEAR
        trust_scores[clear_mask] = np.clip(ai_confidence[clear_mask], 0.70, 0.90)
        reason_codes[clear_mask] = REASON_CLEAR_BACKGROUND

        water_ai = (ai_labels == Label.WATER) & is_spectral_water
        trust_scores[water_ai] = self.config.trust_spectral_high
        reason_codes[water_ai] = REASON_SPECTRAL_WATER

        snow_ai = (ai_labels == Label.SNOW) & is_spectral_snow
        trust_scores[snow_ai] = self.config.trust_spectral_high
        reason_codes[snow_ai] = REASON_SPECTRAL_SNOW

        cloud_ai = (ai_labels == Label.CLOUD) & is_spectral_cloud
        trust_scores[cloud_ai] = self.config.trust_spectral_high
        reason_codes[cloud_ai] = REASON_PHYSICS_CLOUD_SPECTRAL

        # -------------------------------------------------------------
        # Guardrail 5: If hillshade explains the darkening -> hillshade with trust 0.90
        # -------------------------------------------------------------
        hillshade_explains = is_hillshade & is_dark
        if np.any(hillshade_explains):
            final_labels[hillshade_explains] = Label.HILLSHADE
            trust_scores[hillshade_explains] = self.config.trust_hillshade
            reason_codes[hillshade_explains] = REASON_GUARDRAIL_HILLSHADE
            count_hillshade_override = int(np.sum(hillshade_explains))

        # -------------------------------------------------------------
        # Guardrail 4: If physics confirms shadow and AI agrees -> trust 0.95
        # -------------------------------------------------------------
        shadow_consensus = (
            (ai_labels == Label.CLOUD_SHADOW) & geom_shadow & (~hillshade_explains)
        )
        if np.any(shadow_consensus):
            final_labels[shadow_consensus] = Label.CLOUD_SHADOW
            trust_scores[shadow_consensus] = self.config.trust_shadow_consensus
            reason_codes[shadow_consensus] = REASON_PHYSICS_AI_SHADOW_CONSENSUS
            count_shadow_consensus = int(np.sum(shadow_consensus))

        # -------------------------------------------------------------
        # Guardrail 1: If AI says shadow but geometric projection disagrees -> relabel as land_change with trust 0.55
        # -------------------------------------------------------------
        shadow_disagreement = (
            (ai_labels == Label.CLOUD_SHADOW) & (~geom_shadow) & (~hillshade_explains)
        )
        if np.any(shadow_disagreement):
            final_labels[shadow_disagreement] = Label.LAND_CHANGE
            trust_scores[shadow_disagreement] = self.config.trust_land_change
            reason_codes[shadow_disagreement] = REASON_GUARDRAIL_SHADOW_TO_LAND_CHANGE
            count_shadow_to_lc = int(np.sum(shadow_disagreement))

        # -------------------------------------------------------------
        # Active Physical Land-Change Attribution: Flood, Fire/Burn Scar, Crop Damage
        # -------------------------------------------------------------
        unoccluded = (~geom_shadow) & (~hillshade_explains) & (~is_spectral_cloud)

        # (a) Crop Damage / Severe Canopy Senescence (seasonally calibrated)
        crop_active = is_crop_damage & unoccluded
        if np.any(crop_active):
            final_labels[crop_active] = Label.LAND_CHANGE
            # Preserve 0.55 if triggered strictly by Guardrail 1 shadow rejection, else 0.88 verified
            verified_crop = crop_active & (~shadow_disagreement)
            trust_scores[verified_crop] = self.config.trust_land_change_verified
            reason_codes[crop_active] = REASON_LAND_CHANGE_CROP_DAMAGE

        # (b) Fire / Burn Scar (NBR < -0.15, B11 > B8, dark visible)
        fire_active = is_fire & unoccluded
        if np.any(fire_active):
            final_labels[fire_active] = Label.LAND_CHANGE
            verified_fire = fire_active & (~shadow_disagreement)
            trust_scores[verified_fire] = self.config.trust_land_change_verified
            reason_codes[fire_active] = REASON_LAND_CHANGE_FIRE

        # (c) Flood / Turbid Surface Inundation (NDWI >= 0.10, low NIR/SWIR, outside permanent water)
        flood_active = is_flood & unoccluded
        if np.any(flood_active):
            final_labels[flood_active] = Label.LAND_CHANGE
            verified_flood = flood_active & (~shadow_disagreement)
            trust_scores[verified_flood] = self.config.trust_land_change_verified
            reason_codes[flood_active] = REASON_LAND_CHANGE_FLOOD

        count_active_disturbance = int(
            np.sum((crop_active | fire_active | flood_active) & (~shadow_disagreement))
        )

        # Ensure any remaining AI-predicted LAND_CHANGE pixels have a clear reason code
        ai_lc_remaining = (
            (final_labels == Label.LAND_CHANGE)
            & (~shadow_disagreement)
            & (~crop_active)
            & (~fire_active)
            & (~flood_active)
        )
        if np.any(ai_lc_remaining):
            trust_scores[ai_lc_remaining] = self.config.trust_land_change
            reason_codes[ai_lc_remaining] = REASON_GUARDRAIL_SHADOW_TO_LAND_CHANGE

        # -------------------------------------------------------------
        # Guardrail 2: If AI says water but a geometric shadow is predicted -> shadow
        # -------------------------------------------------------------
        water_shadow_conflict = (
            (ai_labels == Label.WATER) & geom_shadow & (~hillshade_explains)
        )
        if np.any(water_shadow_conflict):
            final_labels[water_shadow_conflict] = Label.CLOUD_SHADOW
            trust_scores[water_shadow_conflict] = self.config.trust_water_shadow
            reason_codes[water_shadow_conflict] = REASON_GUARDRAIL_WATER_TO_SHADOW
            count_water_to_shadow = int(np.sum(water_shadow_conflict))

        # -------------------------------------------------------------
        # Guardrail 3: If AI says snow but NDSI is low -> cloud
        # -------------------------------------------------------------
        snow_cloud_conflict = (ai_labels == Label.SNOW) & ndsi_low
        if np.any(snow_cloud_conflict):
            final_labels[snow_cloud_conflict] = Label.CLOUD
            trust_scores[snow_cloud_conflict] = self.config.trust_snow_cloud
            reason_codes[snow_cloud_conflict] = REASON_GUARDRAIL_SNOW_TO_CLOUD
            count_snow_to_cloud = int(np.sum(snow_cloud_conflict))

        # -------------------------------------------------------------
        # Cirrus boost: B10 > 0.03 and high visible brightness -> cloud
        # -------------------------------------------------------------
        cirrus_only = is_cirrus & (final_labels != Label.CLOUD)
        if np.any(cirrus_only):
            final_labels[cirrus_only] = Label.CLOUD
            trust_scores[cirrus_only] = self.config.trust_spectral_high
            reason_codes[cirrus_only] = REASON_PHYSICS_CLOUD_CIRRUS
            count_cirrus_boost = int(np.sum(cirrus_only))

        # Build string name array
        label_names = np.empty((h, w), dtype=object)
        for label_val, name in LABEL_NAMES.items():
            label_names[final_labels == label_val] = name

        # Compute Tile Change Log statistics
        class_counts: Dict[str, int] = {}
        class_percentages: Dict[str, float] = {}
        for label_val, name in LABEL_NAMES.items():
            cnt = int(np.sum(final_labels == label_val))
            class_counts[name] = cnt
            class_percentages[name] = (cnt / float(total_pixels)) * 100.0

        lc_mask = final_labels == Label.LAND_CHANGE
        lc_flood_cnt = int(np.sum(lc_mask & flood_active))
        lc_fire_cnt = int(np.sum(lc_mask & fire_active & (~flood_active)))
        lc_crop_cnt = int(
            np.sum(lc_mask & crop_active & (~fire_active) & (~flood_active))
        )
        lc_unexplained_cnt = int(
            np.sum(lc_mask & (~flood_active) & (~fire_active) & (~crop_active))
        )

        land_change_breakdown = {
            "flood_inundation": lc_flood_cnt,
            "fire_burn_scar": lc_fire_cnt,
            "crop_damage_senescence": lc_crop_cnt,
            "unexplained_dark_anomaly": lc_unexplained_cnt,
        }

        overrides = {
            "AI shadow rejected (relabelled as land_change)": count_shadow_to_lc,
            "Active physical land_change (flood, fire, crop damage)": count_active_disturbance,
            "Water false-positive overridden by geometric shadow": count_water_to_shadow,
            "Low-NDSI snow overridden to cloud": count_snow_to_cloud,
            "Physics and AI confirmed shadow consensus": count_shadow_consensus,
            "Darkness explained by terrain hillshade": count_hillshade_override,
            "Cirrus spectral detection override to cloud": count_cirrus_boost,
        }

        change_log = TileChangeLog(
            total_pixels=total_pixels,
            class_counts=class_counts,
            class_percentages=class_percentages,
            overrides=overrides,
            mean_trust=float(np.mean(trust_scores)),
            min_trust=float(np.min(trust_scores)),
            max_trust=float(np.max(trust_scores)),
            estimated_cloud_height_m=physics.estimated_cloud_height_m,
            shadow_overlap_metric=physics.shadow_overlap_score,
            land_change_breakdown=land_change_breakdown,
            season=physics.season,
            seasonal_ndvi_baseline=physics.seasonal_ndvi_baseline,
            noise_repaired_pixels=noise_repaired,
            radiative_corrections_applied=radiative_steps or [],
        )

        return final_labels, label_names, trust_scores, reason_codes, change_log
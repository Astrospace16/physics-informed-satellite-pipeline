"""Main production-ready SatellitePipeline coordinating ingestion, radiative transfer, physics, ML, and fusion."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from sat_pipeline.core.constants import (
    DEFAULT_BAND_ORDER,
    LABEL_NAMES,
    Label,
    LandChangeSubtype,
)
from sat_pipeline.core.types import (
    PhysicsFeatures,
    PipelineConfig,
    PipelineOutputs,
    TileChangeLog,
    TileMetadata,
)
from sat_pipeline.fusion.guardrails import PhysicsBoundedFusion
from sat_pipeline.ingest.reader import TileReader
from sat_pipeline.models.base import BaseResidualModel
from sat_pipeline.models.factory import ModelFactory
from sat_pipeline.physics.hillshade import compute_hillshade_mask
from sat_pipeline.physics.radiative_transfer import run_radiative_transfer_pipeline
from sat_pipeline.physics.spectral import (
    compute_brightness,
    compute_ndvi,
    compute_ndsi,
    compute_ndwi,
    compute_seasonal_phenology,
    detect_land_change_disturbances,
    detect_spectral_clouds,
    detect_spectral_snow,
    detect_spectral_water,
)
from sat_pipeline.physics.sun_geometry import estimate_cloud_height


class SatellitePipeline:
    """Physics-Informed Satellite Pipeline.

    Integrates sensor noise filtering, radiative-transfer corrections (DOS-1, solar zenith,
    and topographic C-correction), sun geometry shadow projection, DEM hillshade, seasonal
    phenology baselines, active disturbance physics (flood, fire, crop damage), and an AI
    residual model bounded by strict physics guardrails.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        model: Optional[BaseResidualModel] = None,
    ):
        self.config = config or PipelineConfig()
        if model is not None:
            self.model = model
        else:
            self.model = ModelFactory.create(self.config.model_type)
            if self.config.model_path and Path(self.config.model_path).is_file():
                self.model = self.model.load(self.config.model_path)

        self.fusion = PhysicsBoundedFusion(self.config)
        self.last_radiative_steps: List[str] = []

    def extract_physics_features(
        self,
        bands: Dict[str, np.ndarray],
        dem: np.ndarray,
        metadata: TileMetadata,
    ) -> PhysicsFeatures:
        """Apply radiative-transfer & noise corrections and extract all physical features."""
        metadata.validate()

        # 1. DEM Hillshade & Topographic Geometry (Rule 2)
        # Computed first so illumination and slope feed into topographic C-correction
        hillshade_mask, illumination, slope, aspect = compute_hillshade_mask(
            dem=dem,
            sun_elevation_deg=metadata.sun_elevation_deg,
            sun_azimuth_deg=metadata.sun_azimuth_deg,
            pixel_size_m=metadata.pixel_size_m,
            illumination_threshold=self.config.hillshade_illumination_threshold,
        )

        # 2. Sensor Noise Despeckling & Radiative-Transfer Corrections
        # Calls detect_and_filter_sensor_noise, apply_dark_object_subtraction,
        # apply_solar_zenith_correction, and apply_c_correction_topography
        if self.config.enable_radiative_transfer:
            corrected_bands, noise_mask, applied_steps = run_radiative_transfer_pipeline(
                bands=bands,
                sun_elevation_deg=metadata.sun_elevation_deg,
                illumination=illumination,
                slope_rad=slope,
                enable_noise_filtering=self.config.enable_noise_filtering,
                enable_dos=self.config.enable_dos_correction,
                enable_solar_zenith=self.config.enable_solar_zenith_correction,
                enable_topographic=self.config.enable_topographic_correction,
                dos_percentile=self.config.dos_percentile,
                noise_zscore_threshold=self.config.noise_zscore_threshold,
                hillshade_illumination_threshold=self.config.hillshade_illumination_threshold,
            )
            self.last_radiative_steps = applied_steps
        else:
            corrected_bands = {k: v.astype(np.float32).copy() for k, v in bands.items()}
            first_b = next(iter(corrected_bands.values()))
            noise_mask = np.zeros(first_b.shape, dtype=np.uint8)
            self.last_radiative_steps = []

        b2 = corrected_bands["B2"]
        b3 = corrected_bands["B3"]
        b4 = corrected_bands["B4"]
        b8 = corrected_bands["B8"]
        b11 = corrected_bands["B11"]
        b10 = corrected_bands.get("B10", np.zeros_like(b2))
        b12 = corrected_bands.get("B12", None)

        # 3. Seasonal Phenology Calibration
        doy = metadata.day_of_year()
        season = metadata.season_name()
        if self.config.enable_seasonal_modulation:
            seasonal_ndvi_base, effective_ndsi_thresh = compute_seasonal_phenology(
                day_of_year=doy,
                latitude_deg=metadata.latitude_deg,
                base_ndsi_snow_threshold=self.config.ndsi_snow_threshold,
            )
        else:
            seasonal_ndvi_base = 0.45
            effective_ndsi_thresh = self.config.ndsi_snow_threshold

        # 4. Core Spectral Indices (on radiatively corrected bands)
        ndsi = compute_ndsi(b3, b11)
        ndwi = compute_ndwi(b3, b8)
        ndvi = compute_ndvi(b8, b4)
        brightness = compute_brightness(b2, b3, b4)

        # 5. Physics-based Spectral Detections (Rule 3)
        spectral_snow, _ = detect_spectral_snow(
            b3_green=b3,
            b11_swir1=b11,
            ndsi_threshold=effective_ndsi_thresh,
            b2_blue=b2,
        )
        spectral_water, _ = detect_spectral_water(
            b3_green=b3,
            b8_nir=b8,
            ndwi_threshold=self.config.ndwi_water_threshold,
            b8_threshold=self.config.water_b8_threshold,
        )
        spectral_clouds, cirrus_mask, _ = detect_spectral_clouds(
            b2_blue=b2,
            b3_green=b3,
            b4_red=b4,
            b8_nir=b8,
            b10_cirrus=b10,
            b11_swir1=b11,
            cirrus_b10_threshold=self.config.cirrus_b10_threshold,
            visible_brightness_threshold=self.config.visible_brightness_threshold,
            ndsi_snow_threshold=effective_ndsi_thresh,
        )

        # 6. Sun Geometry Shadow Projection and Height Search (Rule 1)
        best_height_m, predicted_shadow_mask, overlap_scores = estimate_cloud_height(
            cloud_mask=spectral_clouds,
            brightness=brightness,
            sun_elevation_deg=metadata.sun_elevation_deg,
            sun_azimuth_deg=metadata.sun_azimuth_deg,
            pixel_size_m=metadata.pixel_size_m,
            min_height_m=self.config.min_cloud_height_m,
            max_height_m=self.config.max_cloud_height_m,
            step_m=self.config.cloud_height_step_m,
            dark_threshold=self.config.brightness_dark_threshold,
        )

        max_overlap_count = overlap_scores.get(best_height_m, 0)
        total_shadow_pixels = np.sum(predicted_shadow_mask > 0)
        overlap_score = float(max_overlap_count / (total_shadow_pixels + 1e-6))

        # 7. Active Physical Land-Change Disturbances: Flood, Fire/Burn Scar, Crop Damage
        flood_mask, fire_mask, crop_damage_mask, nbr, ndmi, bsi = detect_land_change_disturbances(
            b2_blue=b2,
            b3_green=b3,
            b4_red=b4,
            b8_nir=b8,
            b11_swir1=b11,
            ndwi=ndwi,
            ndvi=ndvi,
            brightness=brightness,
            cloud_mask=spectral_clouds,
            predicted_shadow_mask=predicted_shadow_mask,
            hillshade_mask=hillshade_mask,
            permanent_water_mask=spectral_water,
            snow_mask=spectral_snow,
            seasonal_ndvi_baseline=seasonal_ndvi_base,
            nbr_fire_threshold=self.config.nbr_fire_threshold,
            ndwi_flood_min=self.config.ndwi_flood_min,
            crop_damage_ndvi_drop=self.config.crop_damage_ndvi_drop,
            b12_swir2=b12,
        )

        return PhysicsFeatures(
            predicted_shadow_mask=predicted_shadow_mask,
            hillshade_mask=hillshade_mask,
            ndsi=ndsi,
            ndwi=ndwi,
            ndvi=ndvi,
            brightness=brightness,
            cirrus_mask=cirrus_mask,
            spectral_cloud_mask=spectral_clouds,
            spectral_water_mask=spectral_water,
            spectral_snow_mask=spectral_snow,
            estimated_cloud_height_m=best_height_m,
            shadow_overlap_score=overlap_score,
            illumination=illumination,
            slope_rad=slope,
            aspect_rad=aspect,
            nbr=nbr,
            ndmi=ndmi,
            bsi=bsi,
            flood_mask=flood_mask,
            fire_mask=fire_mask,
            crop_damage_mask=crop_damage_mask,
            noise_mask=noise_mask,
            seasonal_ndvi_baseline=seasonal_ndvi_base,
            season=season,
            radiative_corrected_bands=corrected_bands,
        )

    def assemble_feature_stack(
        self,
        bands: Dict[str, np.ndarray],
        physics: PhysicsFeatures,
    ) -> np.ndarray:
        """Stack radiatively corrected bands and physics features into 12-channel model input (H, W, 12)."""
        active_bands = (
            physics.radiative_corrected_bands
            if physics.radiative_corrected_bands is not None
            else bands
        )
        b2 = active_bands["B2"]
        h, w = b2.shape
        stack = np.empty((h, w, 12), dtype=np.float32)

        stack[:, :, 0] = active_bands["B2"]
        stack[:, :, 1] = active_bands["B3"]
        stack[:, :, 2] = active_bands["B4"]
        stack[:, :, 3] = active_bands["B8"]
        stack[:, :, 4] = active_bands["B11"]
        stack[:, :, 5] = active_bands.get("B10", np.zeros((h, w), dtype=np.float32))

        stack[:, :, 6] = physics.predicted_shadow_mask.astype(np.float32)
        stack[:, :, 7] = physics.hillshade_mask.astype(np.float32)
        stack[:, :, 8] = physics.ndsi
        stack[:, :, 9] = physics.ndwi
        stack[:, :, 10] = physics.ndvi
        stack[:, :, 11] = physics.brightness

        return stack

    def run(
        self,
        bands_or_path: Union[Dict[str, np.ndarray], str, Path],
        dem_or_path: Optional[Union[np.ndarray, str, Path]] = None,
        metadata_or_path: Optional[Union[TileMetadata, str, Path]] = None,
        sun_elevation_deg: float = 45.0,
        sun_azimuth_deg: float = 135.0,
        pixel_size_m: float = 10.0,
    ) -> PipelineOutputs:
        """Run the full physics-informed satellite pipeline on a single tile."""
        # 1. Ingest Bands
        if isinstance(bands_or_path, (str, Path)):
            bands = TileReader.read_multiband_tiff(bands_or_path)
        else:
            bands = {k: v.astype(np.float32) for k, v in bands_or_path.items()}

        reference_shape = next(iter(bands.values())).shape

        # 2. Ingest DEM
        if isinstance(dem_or_path, np.ndarray):
            dem = dem_or_path.astype(np.float32)
        else:
            dem = TileReader.read_dem(dem_or_path, reference_shape=reference_shape)

        # 3. Ingest Metadata
        if isinstance(metadata_or_path, TileMetadata):
            metadata = metadata_or_path
        elif isinstance(metadata_or_path, (str, Path)):
            metadata = TileReader.read_metadata(metadata_or_path)
        else:
            metadata = TileMetadata(
                sun_elevation_deg=sun_elevation_deg,
                sun_azimuth_deg=sun_azimuth_deg,
                pixel_size_m=pixel_size_m,
            )

        # 4. Extract Physics Features (Radiative Transfer, Noise, Season, Sun Geometry, Hillshade, Spectral, Disturbances)
        physics_features = self.extract_physics_features(bands, dem, metadata)

        # 5. Assemble 12-channel feature tensor (using radiatively corrected bands)
        features_12ch = self.assemble_feature_stack(bands, physics_features)

        # 6. Run AI Residual Model
        ai_probabilities = self.model.predict_proba(features_12ch)

        # 7. Apply Physics Guardrails & Active Disturbance Fusion
        labels, label_names, trust_scores, reason_codes, change_log = self.fusion.fuse(
            ai_probabilities=ai_probabilities,
            physics=physics_features,
            radiative_steps=self.last_radiative_steps,
        )

        # 8. Build Land-Change Subtype Raster (H, W)
        h, w = labels.shape
        lc_subtypes = np.full((h, w), LandChangeSubtype.NONE, dtype=np.int32)
        lc_mask = labels == Label.LAND_CHANGE
        if np.any(lc_mask):
            lc_subtypes[lc_mask] = LandChangeSubtype.UNEXPLAINED_DARK_ANOMALY
            if physics_features.crop_damage_mask is not None:
                lc_subtypes[lc_mask & (physics_features.crop_damage_mask > 0)] = (
                    LandChangeSubtype.CROP_DAMAGE_SENESCENCE
                )
            if physics_features.fire_mask is not None:
                lc_subtypes[lc_mask & (physics_features.fire_mask > 0)] = (
                    LandChangeSubtype.FIRE_BURN_SCAR
                )
            if physics_features.flood_mask is not None:
                lc_subtypes[lc_mask & (physics_features.flood_mask > 0)] = (
                    LandChangeSubtype.FLOOD_INUNDATION
                )

        return PipelineOutputs(
            labels=labels,
            label_names=label_names,
            trust_scores=trust_scores,
            reason_codes=reason_codes,
            probabilities=ai_probabilities,
            physics_features=physics_features,
            change_log=change_log,
            metadata=metadata,
            land_change_subtypes=lc_subtypes,
        )

    def export_results(
        self,
        outputs: PipelineOutputs,
        output_dir: Union[str, Path],
    ) -> Dict[str, Path]:
        """Export output rasters, changelog, and diagnostics to disk."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        exported_files: Dict[str, Path] = {}

        # 1. Labels TIFF
        lbl_im = Image.fromarray(outputs.labels.astype(np.uint8))
        lbl_file = out_path / "classification_labels.tif"
        lbl_im.save(str(lbl_file))
        exported_files["labels"] = lbl_file

        # 2. Colorized label map for quick visual QA
        palette = [
            34, 139, 34,     # 0: clear (forest green)
            255, 255, 255,   # 1: cloud (white)
            40, 40, 40,      # 2: cloud_shadow (dark charcoal)
            0, 119, 190,     # 3: water (ocean blue)
            0, 255, 255,     # 4: snow (cyan)
            128, 0, 128,     # 5: hillshade (purple)
            255, 140, 0,     # 6: land_change (dark orange)
        ] + [0] * (256 * 3 - 21)
        color_im = Image.fromarray(outputs.labels.astype(np.uint8), mode="P")
        color_im.putpalette(palette)
        color_file = out_path / "classification_colorized.png"
        color_im.save(str(color_file))
        exported_files["colorized"] = color_file

        # 3. Trust Score Map (normalized to 0-255 grayscale)
        trust_u8 = np.clip(outputs.trust_scores * 255.0, 0, 255).astype(np.uint8)
        trust_im = Image.fromarray(trust_u8)
        trust_file = out_path / "trust_score_map.png"
        trust_im.save(str(trust_file))
        exported_files["trust"] = trust_file

        # 4. Change Log JSON & Markdown
        cl_json_file = out_path / "tile_changelog.json"
        cl_json_file.write_text(outputs.change_log.to_json(), encoding="utf-8")
        exported_files["changelog_json"] = cl_json_file

        cl_md_file = out_path / "tile_changelog.md"
        cl_md_file.write_text(outputs.change_log.to_markdown(), encoding="utf-8")
        exported_files["changelog_md"] = cl_md_file

        # 5. Reason Codes Sample / Diagnostic Map
        reason_summary_file = out_path / "reason_codes_summary.txt"
        unique_reasons, counts = np.unique(outputs.reason_codes, return_counts=True)
        summary_lines = ["# Pixel Reason Code Breakdown\n"]
        for r, c in sorted(zip(unique_reasons, counts), key=lambda x: -x[1]):
            pct = (c / float(outputs.labels.size)) * 100.0
            summary_lines.append(f"- [{c:8,d} px | {pct:5.2f}%] {r}")
        reason_summary_file.write_text("\n".join(summary_lines), encoding="utf-8")
        exported_files["reasons_summary"] = reason_summary_file

        return exported_files
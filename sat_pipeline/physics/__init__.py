"""Physics modules package initialization."""

from sat_pipeline.physics.hillshade import (
    compute_dem_gradients,
    compute_hillshade_mask,
    compute_slope_and_aspect,
    compute_solar_illumination,
)
from sat_pipeline.physics.radiative_transfer import (
    apply_c_correction_topography,
    apply_dark_object_subtraction,
    apply_solar_zenith_correction,
    detect_and_filter_sensor_noise,
    run_radiative_transfer_pipeline,
)
from sat_pipeline.physics.spectral import (
    compute_brightness,
    compute_bsi,
    compute_nbr,
    compute_ndmi,
    compute_ndvi,
    compute_ndsi,
    compute_ndwi,
    compute_seasonal_phenology,
    detect_land_change_disturbances,
    detect_spectral_clouds,
    detect_spectral_snow,
    detect_spectral_water,
)
from sat_pipeline.physics.sun_geometry import (
    calculate_shadow_offset_m,
    calculate_shadow_shift_pixels,
    estimate_cloud_height,
    project_cloud_mask,
    project_cloud_shadow,
)

__all__ = [
    "calculate_shadow_offset_m",
    "calculate_shadow_shift_pixels",
    "project_cloud_mask",
    "project_cloud_shadow",
    "estimate_cloud_height",
    "compute_dem_gradients",
    "compute_slope_and_aspect",
    "compute_solar_illumination",
    "compute_hillshade_mask",
    "compute_ndsi",
    "compute_ndwi",
    "compute_ndvi",
    "compute_nbr",
    "compute_ndmi",
    "compute_bsi",
    "compute_brightness",
    "compute_seasonal_phenology",
    "detect_spectral_snow",
    "detect_spectral_water",
    "detect_spectral_clouds",
    "detect_land_change_disturbances",
    "apply_solar_zenith_correction",
    "apply_dark_object_subtraction",
    "apply_c_correction_topography",
    "detect_and_filter_sensor_noise",
    "run_radiative_transfer_pipeline",
]
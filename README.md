# Physics-Informed Satellite Pipeline (Sentinel-2 L2A)

Production-ready Python repository implementing a **Physics-Informed Satellite Pipeline** that separates clouds from their shadows using solar geometry, DEM topographic hillshade, radiative-transfer & sensor noise pre-processing, seasonal phenology baselines, active disturbance physics (`flood_inundation`, `fire_burn_scar`, `crop_damage_senescence`), and a residual ML model bounded by physical guardrails.

Runnable end-to-end on a single Sentinel-2 L2A tile with **zero external model weights required** (demo fallback mode) or with trained residual models (RandomForest and U-Net).

---

## 1. System Architecture

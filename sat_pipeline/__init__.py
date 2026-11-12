"""Physics-Informed Satellite Pipeline package."""

from sat_pipeline.core.constants import LABEL_NAMES, NUM_CLASSES, Label
from sat_pipeline.core.types import (
    PhysicsFeatures,
    PipelineConfig,
    PipelineOutputs,
    TileChangeLog,
    TileMetadata,
)
from sat_pipeline.models.factory import ModelFactory
from sat_pipeline.pipeline import SatellitePipeline

__version__ = "1.0.0"

__all__ = [
    "SatellitePipeline",
    "PipelineConfig",
    "PipelineOutputs",
    "TileMetadata",
    "TileChangeLog",
    "PhysicsFeatures",
    "Label",
    "LABEL_NAMES",
    "NUM_CLASSES",
    "ModelFactory",
]
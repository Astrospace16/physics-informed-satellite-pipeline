"""Model factory for residual models in the satellite pipeline."""

from typing import Dict, Type
from sat_pipeline.models.base import BaseResidualModel
from sat_pipeline.models.random_forest import RandomForestResidualModel
from sat_pipeline.models.rule_based import RuleBasedPseudoProbabilityModel
from sat_pipeline.models.unet import UNetResidualModel


class ModelFactory:
    """Factory to instantiate AI residual models and fallback demo engines."""

    _REGISTRY: Dict[str, Type[BaseResidualModel]] = {
        "rule_based": RuleBasedPseudoProbabilityModel,
        "demo": RuleBasedPseudoProbabilityModel,
        "pseudo_probability": RuleBasedPseudoProbabilityModel,
        "random_forest": RandomForestResidualModel,
        "rf": RandomForestResidualModel,
        "unet": UNetResidualModel,
        "u-net": UNetResidualModel,
    }

    @classmethod
    def create(cls, model_type: str = "rule_based", **kwargs) -> BaseResidualModel:
        """Create a residual model instance by name.

        Args:
            model_type: One of 'rule_based' (default demo), 'random_forest', 'unet'.
            **kwargs: Extra hyperparameters passed to model constructor.

        Returns:
            An instance conforming to BaseResidualModel.
        """
        key = model_type.strip().lower()
        if key not in cls._REGISTRY:
            available = list(cls._REGISTRY.keys())
            raise ValueError(f"Unknown model_type '{model_type}'. Choose from {available}")

        model_cls = cls._REGISTRY[key]
        return model_cls(**kwargs)

    @classmethod
    def register(cls, name: str, model_cls: Type[BaseResidualModel]) -> None:
        """Register a new residual model architecture."""
        cls._REGISTRY[name.strip().lower()] = model_cls
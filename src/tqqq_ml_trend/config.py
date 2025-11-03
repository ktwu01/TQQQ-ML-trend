"""Configuration objects for the TQQQ ML trend project."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Sequence


@dataclass(slots=True)
class DataConfig:
    """Configuration for data acquisition."""

    symbol: str = "TQQQ"
    start: str | None = "2010-02-11"
    end: str | None = None
    cache_path: Path | None = None


@dataclass(slots=True)
class FeatureConfig:
    """Configuration for feature engineering."""

    windows: Sequence[int] = (5, 20, 60)
    include_volume: bool = True


@dataclass(slots=True)
class TrainingConfig:
    """Configuration for model training."""

    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    enable_torch: bool = True
    enable_tensorflow: bool = True


@dataclass(slots=True)
class ForecastConfig:
    """Configuration for forward predictions."""

    forecast_end: date = date(2026, 12, 31)
    confidence_interval: float = 0.8
    simulations: int = 100


@dataclass(slots=True)
class ProjectConfig:
    """Container for all project configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)


DEFAULT_CONFIG = ProjectConfig()
"""Default configuration used by the command line interface."""

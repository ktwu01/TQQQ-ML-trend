"""Training pipeline orchestrating data download, feature engineering, and modelling."""

from __future__ import annotations

from typing import Any

import numpy as np
import warnings

from ..config import ProjectConfig
from ..data.downloader import download_price_history
from ..features.technical import engineer_features
from ..models.regressors import train_regressors

try:
    from ..models.torch_regressor import predict_torch, train_torch_regressor
except ImportError:  # pragma: no cover - optional dependency
    predict_torch = train_torch_regressor = None  # type: ignore

try:
    from ..models.tensorflow_regressor import (
        predict_tensorflow,
        train_tensorflow_regressor,
    )
except ImportError:  # pragma: no cover - optional dependency
    predict_tensorflow = train_tensorflow_regressor = None  # type: ignore


class TrainingArtifacts(dict):
    """Dictionary-like container storing training results for each model family."""


def run_training_pipeline(config: ProjectConfig) -> TrainingArtifacts:
    """Execute the end-to-end training pipeline."""
    price_history = download_price_history(config.data)
    feature_table, feature_columns, target_column = engineer_features(
        price_history, config.features
    )

    artifacts: TrainingArtifacts = TrainingArtifacts()

    sklearn_results = train_regressors(
        feature_table, feature_columns, target_column, config.training
    )
    artifacts["sklearn"] = sklearn_results

    if config.training.enable_torch:
        if train_torch_regressor is None:
            warnings.warn(
                "PyTorch is not installed; skipping torch-based training.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            torch_result = train_torch_regressor(
                feature_table, feature_columns, target_column, config.training
            )
            artifacts["torch"] = {
                "neural_network": {
                    "rmse": torch_result.rmse,
                    "model": torch_result.model,
                    "predict_fn": predict_torch,
                    "feature_columns": feature_columns,
                }
            }

    if config.training.enable_tensorflow:
        if train_tensorflow_regressor is None:
            warnings.warn(
                "TensorFlow is not installed; skipping tensorflow-based training.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            tf_result = train_tensorflow_regressor(
                feature_table, feature_columns, target_column, config.training
            )
            artifacts["tensorflow"] = {
                "dense_network": {
                    "rmse": tf_result.rmse,
                    "model": tf_result.model,
                    "predict_fn": predict_tensorflow,
                    "feature_columns": feature_columns,
                }
            }

    artifacts["feature_table"] = feature_table
    artifacts["feature_columns"] = feature_columns
    artifacts["target_column"] = target_column
    artifacts["price_history"] = price_history
    artifacts["feature_config"] = config.features
    artifacts["training_config"] = config.training
    artifacts["data_config"] = config.data

    return artifacts


def select_best_model(artifacts: TrainingArtifacts) -> tuple[str, Any]:
    """Select the best-performing model across all trained families."""
    best_name = ""
    best_score = np.inf
    best_payload: Any = None

    skip_keys = {
        "feature_table",
        "feature_columns",
        "target_column",
        "price_history",
        "feature_config",
        "training_config",
        "data_config",
    }

    for family, details in artifacts.items():
        if family in skip_keys:
            continue
        if isinstance(details, dict):
            for name, result in details.items():
                if isinstance(result, dict):
                    rmse = float(result.get("rmse", np.inf))
                    if rmse < best_score:
                        best_name = f"{family}:{name}"
                        best_score = rmse
                        best_payload = result
                else:
                    rmse = float(details.get("rmse", np.inf))  # type: ignore[arg-type]
                    if rmse < best_score:
                        best_name = family
                        best_score = rmse
                        best_payload = details
        else:
            rmse = float(details.get("rmse", np.inf))  # type: ignore[arg-type]
            if rmse < best_score:
                best_name = family
                best_score = rmse
                best_payload = details

    if not best_name:
        raise RuntimeError("No models trained; unable to select best model.")

    return best_name, best_payload

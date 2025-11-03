"""TensorFlow-based neural network regressor."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from ..config import TrainingConfig


@dataclass
class TensorFlowTrainingResult:
    """Container for TensorFlow training artefacts."""

    model: object
    rmse: float


def _load_tensorflow():
    tf_spec = importlib.util.find_spec("tensorflow")
    if tf_spec is None:
        raise ImportError(
            "TensorFlow is not installed. Install tensorflow to enable neural training."
        )
    tf = importlib.import_module("tensorflow")
    return tf


def train_tensorflow_regressor(
    data: pd.DataFrame,
    feature_columns: Iterable[str],
    target_column: str,
    config: TrainingConfig,
) -> TensorFlowTrainingResult:
    """Train a dense neural network using TensorFlow Keras."""
    tf = _load_tensorflow()

    X = data.loc[:, feature_columns].to_numpy().astype("float32")
    y = data[target_column].to_numpy().astype("float32")

    split_index = int(len(X) * (1 - config.test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(len(feature_columns),)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )

    model.fit(
        X_train,
        y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=0,
        validation_split=config.validation_size,
        shuffle=True,
    )

    evaluation = model.evaluate(X_test, y_test, verbose=0)
    rmse = float(evaluation[1])

    return TensorFlowTrainingResult(model=model, rmse=rmse)


def predict_tensorflow(model: object, features: np.ndarray) -> np.ndarray:
    """Run inference with the trained TensorFlow model."""
    predictions = model.predict(features, verbose=0)
    return predictions.flatten()

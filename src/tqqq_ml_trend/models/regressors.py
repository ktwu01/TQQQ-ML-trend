"""Traditional machine learning regressors for return forecasting."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import TrainingConfig


def create_regressor_pipelines(feature_names: Iterable[str]) -> Dict[str, Pipeline]:
    """Create a dictionary of sklearn pipelines keyed by descriptive names."""
    numeric_features = list(feature_names)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
        ]
    )

    pipelines: Dict[str, Pipeline] = {
        "linear_regression": Pipeline(
            steps=[("preprocess", preprocessor), ("model", LinearRegression())]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        random_state=0,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    GradientBoostingRegressor(random_state=0),
                ),
            ]
        ),
    }
    return pipelines


def train_regressors(
    data: pd.DataFrame,
    feature_columns: Iterable[str],
    target_column: str,
    config: TrainingConfig,
) -> dict[str, dict[str, float | Pipeline]]:
    """Train a suite of scikit-learn regressors."""
    X = data.loc[:, feature_columns]
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        shuffle=False,
    )

    results: dict[str, dict[str, float | Pipeline]] = {}
    for name, pipeline in create_regressor_pipelines(feature_columns).items():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        results[name] = {"rmse": rmse, "model": pipeline}

    return results

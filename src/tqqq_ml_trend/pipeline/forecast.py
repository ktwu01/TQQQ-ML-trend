"""Forecasting helpers for projecting TQQQ trends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..config import ForecastConfig
from ..features.technical import compute_indicator_frame
from ..utils.dates import generate_trading_days


@dataclass
class ForecastResult:
    """Output of the forecasting routine."""

    forecast: pd.DataFrame
    model_name: str


def iterative_forecast(
    artifacts: dict[str, Any],
    model_payload: Any,
    model_name: str,
    feature_columns: Iterable[str],
    forecast_config: ForecastConfig,
) -> ForecastResult:
    """Generate a forward-looking forecast until ``forecast_config.forecast_end``."""
    history: pd.DataFrame = artifacts["price_history"].copy()
    history = history.rename(columns={"date": "date", "close": "close"})
    history = history.set_index("date")
    last_known_date = history.index[-1]

    future_dates = generate_trading_days(
        last_known_date.date(), forecast_config.forecast_end
    )[1:]

    forecast_records: list[dict[str, Any]] = []
    working_history = history.copy()

    feature_config = artifacts["feature_config"]

    for future_day in future_dates:
        engineered = compute_indicator_frame(
            working_history.reset_index(),
            feature_config,
            dropna=False,
        ).dropna().reset_index(drop=True)

        if engineered.empty:
            raise RuntimeError("Not enough historical data to compute indicators.")

        latest_features = engineered.loc[:, feature_columns].iloc[-1].to_numpy()
        latest_features_2d = latest_features.reshape(1, -1)

        if isinstance(model_payload, dict) and "model" in model_payload:
            model = model_payload["model"]
            predict_fn = model_payload.get("predict_fn")
            if callable(predict_fn):
                predicted_return = float(predict_fn(model, latest_features_2d)[0])
            else:
                predicted_return = float(model.predict(latest_features_2d)[0])
        else:
            model = model_payload
            predicted_return = float(model.predict(latest_features_2d)[0])

        last_price = float(working_history["close"].iloc[-1])
        next_price = last_price * np.exp(predicted_return)

        future_row = {
            "date": pd.Timestamp(future_day),
            "close": next_price,
        }
        if "volume" in working_history.columns:
            future_row["volume"] = float(working_history["volume"].iloc[-1])
        working_history = pd.concat(
            [
                working_history,
                pd.DataFrame([future_row]).set_index("date"),
            ]
        )
        forecast_records.append(future_row)

    forecast_df = pd.DataFrame(forecast_records)
    return ForecastResult(forecast=forecast_df, model_name=model_name)


def forecast_to_target_date(
    artifacts: dict[str, Any],
    best_model: tuple[str, Any],
    forecast_config: ForecastConfig,
) -> ForecastResult:
    """High-level helper that wraps :func:`iterative_forecast`."""
    model_name, payload = best_model
    feature_columns = artifacts["feature_columns"]
    return iterative_forecast(
        artifacts=artifacts,
        model_payload=payload,
        model_name=model_name,
        feature_columns=feature_columns,
        forecast_config=forecast_config,
    )

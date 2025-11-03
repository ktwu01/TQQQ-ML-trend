"""Feature engineering helpers for technical indicators."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..config import FeatureConfig


def engineer_features(
    price_history: pd.DataFrame,
    config: FeatureConfig,
) -> tuple[pd.DataFrame, Sequence[str], str]:
    """Generate model-ready features from raw price history.

    Parameters
    ----------
    price_history:
        DataFrame returned by :func:`tqqq_ml_trend.data.downloader.download_price_history`.
    config:
        Feature engineering configuration.

    Returns
    -------
    tuple[pandas.DataFrame, Sequence[str], str]
        The feature matrix with aligned target column, list of feature names, and the
        name of the prediction target column.
    """
    df = compute_indicator_frame(price_history, config, dropna=False)

    df["target_log_return"] = df["log_return"].shift(-1)
    df = df.dropna().reset_index(drop=True)

    feature_columns: list[str] = _collect_feature_columns(df.columns, config)
    target_column = "target_log_return"

    return df, feature_columns, target_column


def compute_indicator_frame(
    price_history: pd.DataFrame,
    config: FeatureConfig,
    dropna: bool = True,
) -> pd.DataFrame:
    """Compute technical indicators without creating a prediction target."""
    df = price_history.copy()
    df = df.rename(columns={"close": "close", "volume": "volume"})

    df["close"] = df["close"].astype(float)
    if "volume" in df:
        df["volume"] = df["volume"].astype(float)

    df["log_return"] = np.log(df["close"]).diff()
    df["daily_return"] = df["close"].pct_change()

    for window in config.windows:
        df[f"sma_{window}"] = df["close"].rolling(window=window).mean()
        df[f"ema_{window}"] = df["close"].ewm(span=window, adjust=False).mean()
        df[f"volatility_{window}"] = df["log_return"].rolling(window).std()

    if config.include_volume and "volume" in df:
        for window in config.windows:
            df[f"volume_mean_{window}"] = df["volume"].rolling(window=window).mean()
            df[f"volume_z_{window}"] = (
                (df["volume"] - df[f"volume_mean_{window}"])
                / df["volume"].rolling(window=window).std()
            )

    if dropna:
        df = df.dropna().reset_index(drop=True)

    return df


def _collect_feature_columns(columns: Iterable[str], config: FeatureConfig) -> list[str]:
    base_features = {"close", "log_return", "daily_return"}
    if config.include_volume:
        base_features.add("volume")

    allowed_prefixes = (
        "sma_",
        "ema_",
        "volatility_",
        "volume_mean_",
        "volume_z_",
    )

    feature_columns: list[str] = []
    for name in columns:
        if name in {"date", "target_log_return"}:
            continue
        if name in base_features:
            feature_columns.append(name)
            continue
        if any(name.startswith(prefix) for prefix in allowed_prefixes):
            if config.include_volume or not name.startswith("volume"):
                feature_columns.append(name)
    return feature_columns

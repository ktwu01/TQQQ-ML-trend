"""Date utility helpers."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


def generate_trading_days(start: date, end: date) -> list[date]:
    """Generate trading days between start and end using NYSE calendar approximation."""
    current = start
    trading_days: list[date] = []
    while current <= end:
        if current.weekday() < 5:  # Monday-Friday
            trading_days.append(current)
        current += timedelta(days=1)
    return trading_days


def next_trading_day(previous: pd.Timestamp) -> pd.Timestamp:
    """Return the next weekday date after ``previous``."""
    next_day = previous + pd.tseries.offsets.BDay(1)
    return pd.Timestamp(next_day.date())

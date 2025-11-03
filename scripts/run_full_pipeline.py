"""Convenience script to execute the full training + forecasting workflow."""

from __future__ import annotations

from pathlib import Path

from tqqq_ml_trend.config import ProjectConfig
from tqqq_ml_trend.pipeline.forecast import forecast_to_target_date
from tqqq_ml_trend.pipeline.train import run_training_pipeline, select_best_model


def main() -> None:
    config = ProjectConfig()
    artifacts = run_training_pipeline(config)
    best_model = select_best_model(artifacts)
    forecast = forecast_to_target_date(artifacts, best_model, config.forecast)

    output = Path("artifacts/forecast_2026.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    forecast.forecast.to_csv(output, index=False)
    print(f"Saved forecast using {forecast.model_name} to {output}")


if __name__ == "__main__":  # pragma: no cover
    main()

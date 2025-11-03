"""Command line interface for the TQQQ ML trend framework."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import ProjectConfig
from .data.downloader import download_price_history
from .pipeline.forecast import forecast_to_target_date
from .pipeline.train import run_training_pipeline, select_best_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train models and forecast TQQQ trends using machine learning.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser(
        "download-data", help="Download and cache historical data only."
    )
    download_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/tqqq_history.parquet"),
        help="Where to store the downloaded dataset.",
    )

    train_parser = subparsers.add_parser(
        "train", help="Run the full training pipeline and report metrics."
    )
    train_parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts"),
        help="Directory for saving model summaries.",
    )

    forecast_parser = subparsers.add_parser(
        "forecast", help="Train models and forecast prices out to 2026."
    )
    forecast_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/forecast_2026.csv"),
        help="Destination for the generated forecast CSV.",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = ProjectConfig()

    if args.command == "download-data":
        config.data.cache_path = args.output
        df = download_price_history(config.data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.output)
        print(f"Downloaded {len(df)} rows to {args.output}")
        return

    if args.command == "train":
        artifacts = run_training_pipeline(config)
        artifacts_dir: Path = args.artifacts
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        metrics_rows: list[dict[str, float | str]] = []
        skip_keys = {
            "feature_table",
            "feature_columns",
            "target_column",
            "price_history",
            "feature_config",
            "training_config",
            "data_config",
        }

        for family, results in artifacts.items():
            if family in skip_keys:
                continue
            for name, payload in results.items():  # type: ignore[assignment]
                rmse = payload["rmse"] if isinstance(payload, dict) else payload.rmse
                metrics_rows.append({"model": f"{family}:{name}", "rmse": float(rmse)})

        metrics_df = pd.DataFrame(metrics_rows).sort_values("rmse")
        metrics_path = artifacts_dir / "model_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(metrics_df)
        print(f"Saved metrics to {metrics_path}")
        return

    if args.command == "forecast":
        artifacts = run_training_pipeline(config)
        best_model = select_best_model(artifacts)
        forecast_result = forecast_to_target_date(
            artifacts, best_model, config.forecast
        )
        output_path: Path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        forecast_result.forecast.to_csv(output_path, index=False)
        print(
            f"Forecast using {forecast_result.model_name} saved to {output_path}."
        )
        return

    raise ValueError(f"Unknown command {args.command!r}")


if __name__ == "__main__":  # pragma: no cover
    main()

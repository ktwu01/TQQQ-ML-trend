# TQQQ Machine Learning Trend Framework

This repository provides an end-to-end, English-language framework for downloading, preparing, and modelling the historical performance of the ProShares UltraPro QQQ (TQQQ) exchange-traded fund. The toolkit downloads data from Yahoo! Finance, engineers technical indicators, trains a diverse ensemble of machine learning models (including scikit-learn pipelines, PyTorch, and TensorFlow networks), and projects potential trajectories through the end of 2026.

## Project Goals

1. **Acquire data** – fetch the complete daily TQQQ history directly from Yahoo! Finance.
2. **Engineer signals** – create rolling technical indicators and lagged returns that feed the models.
3. **Train multiple learners** – compare linear models, tree ensembles, and neural networks built with PyTorch and TensorFlow.
4. **Forecast 2026** – iterate the best-performing model forward to estimate future prices up to 31 December 2026.

## Repository Structure

```
├── .gitignore                 # Ignore datasets and generated artefacts
├── README.md                  # Project overview and usage instructions
├── requirements.txt           # Core Python dependencies
├── scripts/                   # Convenience scripts and helpers
├── src/
│   └── tqqq_ml_trend/
│       ├── __init__.py
│       ├── cli.py             # Command line interface
│       ├── config.py          # Dataclass-driven configuration
│       ├── data/
│       │   └── downloader.py  # Yahoo! Finance data access
│       ├── features/
│       │   └── technical.py   # Feature engineering utilities
│       ├── models/
│       │   ├── regressors.py          # Scikit-learn models
│       │   ├── tensorflow_regressor.py# TensorFlow dense regressor
│       │   └── torch_regressor.py     # PyTorch feed-forward regressor
│       ├── pipeline/
│       │   ├── forecast.py     # Iterative forecasting helpers
│       │   └── train.py        # End-to-end training orchestration
│       └── utils/
│           └── dates.py        # Trading-day utilities
```

## Installation

Create a virtual environment and install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\\Scripts\\activate`
pip install -r requirements.txt
```

### Optional deep learning backends

Install PyTorch and/or TensorFlow to unlock the neural regressors:

```bash
pip install torch torchvision torchaudio   # PyTorch (choose the right build for your platform)
pip install tensorflow                     # TensorFlow
```

## Usage

The project exposes a single command-line entry point:

```bash
python -m tqqq_ml_trend.cli <command> [options]
```

### 1. Download data

```bash
python -m tqqq_ml_trend.cli download-data --output data/tqqq_history.parquet
```

The command pulls the full TQQQ history from Yahoo! Finance and caches it locally in Parquet format.

### 2. Train models

```bash
python -m tqqq_ml_trend.cli train --artifacts artifacts/
```

This runs the entire training pipeline:

1. Downloads or loads cached data.
2. Builds rolling technical indicators.
3. Trains the scikit-learn pipelines and any available neural networks.
4. Writes a ranked `model_metrics.csv` file in the chosen artifacts directory.

### 3. Forecast prices through 2026

```bash
python -m tqqq_ml_trend.cli forecast --output artifacts/forecast_2026.csv
```

The forecast procedure selects the model with the lowest validation RMSE, then iteratively projects adjusted log returns for each future trading day through 31 December 2026. The resulting CSV contains the projected close price for every predicted date.

## Extending the Framework

- Tune the look-back windows and feature set by editing `FeatureConfig` in `config.py`.
- Adjust training hyperparameters (epochs, batch size, learning rate) via `TrainingConfig`.
- Replace or add models in `models/` to experiment with alternative learners.
- Swap out the forecasting strategy in `pipeline/forecast.py` for Monte-Carlo simulations or scenario analysis.

## Caveats

- The framework uses a simplified trading calendar (weekdays only) and does not model market holidays explicitly.
- Forecasts rely on iteratively feeding model predictions back into the feature generator. Small errors can compound over long horizons; treat forecasts as exploratory scenarios, not investment advice.
- Neural components require optional dependencies (PyTorch/TensorFlow) that are not installed automatically by the base requirements.

## License

This project is released under the MIT License. See `LICENSE` for details.

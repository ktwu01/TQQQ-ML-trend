"""PyTorch-based neural network regressor."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from ..config import TrainingConfig


@dataclass
class TorchTrainingResult:
    """Container for PyTorch training artefacts."""

    model: object
    rmse: float


def _load_torch():
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None:
        raise ImportError(
            "PyTorch is not installed. Install torch to enable neural training."
        )
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    utils = importlib.import_module("torch.utils.data")
    return torch, nn, utils


def train_torch_regressor(
    data: pd.DataFrame,
    feature_columns: Iterable[str],
    target_column: str,
    config: TrainingConfig,
) -> TorchTrainingResult:
    """Train a simple feed-forward neural network using PyTorch."""
    torch, nn, utils = _load_torch()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.tensor(data.loc[:, feature_columns].to_numpy(), dtype=torch.float32)
    y = torch.tensor(data[target_column].to_numpy(), dtype=torch.float32).unsqueeze(1)

    dataset = utils.data.TensorDataset(X, y)
    train_size = int(len(dataset) * (1 - config.test_size))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = utils.data.random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(config.random_state),
    )

    train_loader = utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    test_loader = utils.data.DataLoader(test_dataset, batch_size=config.batch_size)

    model = nn.Sequential(
        nn.Linear(len(feature_columns), 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    model.train()
    for _ in range(config.epochs):
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    predictions: list[float] = []
    actuals: list[float] = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            preds = model(batch_X).cpu().numpy().flatten()
            predictions.extend(preds.tolist())
            actuals.extend(batch_y.numpy().flatten().tolist())

    rmse = float(np.sqrt(np.mean((np.array(predictions) - np.array(actuals)) ** 2)))

    return TorchTrainingResult(model=model.cpu(), rmse=rmse)


def predict_torch(model: object, features: np.ndarray) -> np.ndarray:
    """Run inference with the trained PyTorch model."""
    torch = importlib.import_module("torch")
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(features, dtype=torch.float32)
        outputs = model(tensor).cpu().numpy().flatten()
    return outputs

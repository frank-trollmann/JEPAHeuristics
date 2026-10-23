"""
GPU training utilities for the JEPA heuristic dataset.

No RAPIDS/cuDF dependency: parquet I/O uses polars, all training runs on the
GPU through PyTorch (linear regression, MLP) or XGBoost (`device="cuda"`).
Everything used here is already covered by requirements.txt / pyproject.toml.
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import torch
import xgboost as xgb


def convert_pickle_to_parquet(pickle_path: str) -> str:
    """
    Converts a JEPA heuristic dataset from pickle (.pkl) to parquet (.parquet).
    Optimizes memory by casting float64 to float32.

    Args:
        pickle_path: Path to the source .pkl file (e.g., 'data/2023-10-27_14-30-05_10000_batch_heuristic_dataset.pkl')

    Returns:
        The path to the created .parquet file.
    """
    if not pickle_path.endswith(".pkl"):
        raise ValueError("The provided file must be a .pkl file.")

    # Generate parquet path by replacing extension
    parquet_path = pickle_path.replace(".pkl", ".parquet")

    # 1. Load pickle using pandas (most reliable for .pkl)
    df_pandas = pd.read_pickle(pickle_path)

    # 2. Convert to Polars for efficient casting and writing
    df = pl.DataFrame(df_pandas)
    del df_pandas  # avoid holding both copies in host RAM

    # 3. Optimize: Cast all float64 to float32 to save 50% space and speed up GPU loading
    df = df.with_columns(
        pl.col(pl.Float64).cast(pl.Float32)
    )

    # 4. Save as Parquet
    df.write_parquet(parquet_path, compression="snappy")

    return parquet_path


def _column_matrix(series: pl.Series) -> np.ndarray:
    """
    Turns one parquet column into a 2-D float32 array.

    A scalar column becomes (N, 1); a column of latent vectors (List dtype,
    e.g. a 192-dim JEPA encoding) becomes (N, 192).
    """
    values = series.to_numpy()
    if values.dtype == object:  # List(Float32) comes back as an object array
        values = np.stack(values.tolist())
    return np.asarray(values, dtype=np.float32).reshape(len(series), -1)


def load_features(parquet_path: str, target_col: str = "heuristic", device: str = "cuda"):
    """
    Reads the parquet dataset and returns (X, y) as float32 tensors on `device`.

    X has one column per feature value, so latent-vector columns are flattened
    (two 192-dim encodings -> 384 features).
    """
    df = pl.read_parquet(parquet_path)

    feature_cols = [c for c in df.columns if c != target_col]
    X = np.concatenate([_column_matrix(df[c]) for c in feature_cols], axis=1)
    # np.array() forces a writable copy; polars hands out read-only buffers
    y = np.array(df[target_col].to_numpy(), dtype=np.float32)
    del df

    X = torch.from_numpy(X).to(device, non_blocking=True)
    y = torch.from_numpy(y).to(device, non_blocking=True)
    return X, y


def split_data(X, y, train_size: float = 0.8, seed: int | None = None):
    """
    Shuffles and splits on the device where the data already lives.

    Returns:
        x_train, x_test, y_train, y_test as torch tensors.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device=X.device).manual_seed(seed)

    permutation = torch.randperm(X.shape[0], device=X.device, generator=generator)
    split_idx = int(X.shape[0] * train_size)

    train_idx = permutation[:split_idx]
    test_idx = permutation[split_idx:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class LinearRegressor:
    """
    Ordinary least squares solved on the GPU.

    The normal equations are accumulated chunk-wise in float64 instead of
    building one huge design matrix: no 2x copy of X and no QR/SVD workspace
    that scales with the number of rows.
    """

    def __init__(self, chunk_size: int = 131072):
        self.chunk_size = chunk_size

    def fit(self, X, y):
        num_rows, num_features = X.shape
        device = X.device
        targets = y.reshape(-1, 1)

        gram = torch.zeros(num_features + 1, num_features + 1,
                           dtype=torch.float64, device=device)
        rhs = torch.zeros(num_features + 1, 1, dtype=torch.float64, device=device)

        for start in range(0, num_rows, self.chunk_size):
            chunk = X[start:start + self.chunk_size].to(torch.float64)
            ones = torch.ones(chunk.shape[0], 1, dtype=torch.float64, device=device)
            chunk = torch.cat([chunk, ones], dim=1)

            gram += chunk.T @ chunk
            rhs += chunk.T @ targets[start:start + chunk.shape[0]].to(torch.float64)

        # Tiny relative jitter keeps the solve stable for collinear features
        jitter = gram.diagonal().mean().clamp_min(1e-12) * 1e-10
        gram += torch.eye(num_features + 1, dtype=torch.float64, device=device) * jitter

        try:
            solution = torch.linalg.solve(gram, rhs)
        except RuntimeError:
            solution = torch.linalg.pinv(gram) @ rhs

        self.coef_ = solution[:-1].T.to(X.dtype).contiguous()  # (1, num_features)
        self.intercept_ = solution[-1].to(X.dtype)             # (1,)
        return self

    def predict(self, X):
        return (X @ self.coef_.T + self.intercept_).squeeze(1)


def train_xgboost(X, y, n_estimators: int = 500, max_depth: int = 6,
                  learning_rate: float = 0.1, device: str = "cuda",
                  max_rows: int = 0):
    """
    Gradient boosted trees on the GPU (XGBoost `tree_method="hist"`).

    XGBoost takes host/NumPy input here, so the training matrix is copied back
    once; prediction happens on the GPU.

    `max_rows > 0` trains on a random subset. Tree models plateau early, and
    for multi-million row datasets the host copy of X is by far the largest
    allocation of the whole pipeline.
    """
    # On macOS, torch and xgboost ship two libomp runtimes; spawning more than
    # one OpenMP thread then crashes with "OMP: Error #179". Linux is unaffected.
    n_jobs = 1 if sys.platform == "darwin" else -1

    num_rows = X.shape[0]
    if 0 < max_rows < num_rows:
        print(f"XGBoost: training on a random subset of {max_rows:,} "
              f"of {num_rows:,} rows")
        subset = torch.randperm(num_rows, device=X.device)[:max_rows]
        X, y = X[subset], y[subset]

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device=device,
        n_jobs=n_jobs,
    )
    model.fit(_to_numpy(X), _to_numpy(y))
    return model


class _Standardize(torch.nn.Module):
    """Feature scaling with statistics taken from the training split."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        return (x - self.mean) / self.std


def train_mlp(X, y, hidden_layers=(64, 32), epochs: int = 100,
              batch_size: int = 4096, lr: float = 1e-3):
    """
    Trains a Multi-Layer Perceptron with PyTorch on the device of `X`.

    The latent features are standardized on the training split; without this
    the network barely beats predicting the mean.
    """
    device = X.device
    y = y.reshape(-1, 1)

    mean = X.mean(dim=0)
    std = X.std(dim=0).clamp_min(1e-6)

    layers = [_Standardize(mean, std)]
    prev_dim = X.shape[1]
    for h_dim in hidden_layers:
        layers.append(torch.nn.Linear(prev_dim, h_dim))
        layers.append(torch.nn.ReLU())
        prev_dim = h_dim
    layers.append(torch.nn.Linear(prev_dim, 1))

    model = torch.nn.Sequential(*layers).to(device)
    # Start from the mean target instead of ~0 (heuristics are ~127 on average)
    model[-1].bias.data.fill_(float(y.mean()))

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(X.shape[0], device=device)
        for start in range(0, X.shape[0], batch_size):
            indices = permutation[start:start + batch_size]

            optimizer.zero_grad()
            loss = criterion(model(X[indices]), y[indices])
            loss.backward()
            optimizer.step()

    return model


def _to_numpy(data) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Returns 1-D predictions on the same device as X."""
    if isinstance(model, torch.nn.Module):
        model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, X.shape[0], batch_size):
                outputs.append(model(X[start:start + batch_size]).squeeze(1))
        return torch.cat(outputs)

    if isinstance(model, LinearRegressor):
        with torch.no_grad():
            return model.predict(X)

    # XGBoost
    pred = model.predict(_to_numpy(X))
    return torch.as_tensor(pred, device=X.device, dtype=torch.float32).reshape(-1)


def _rank(values: torch.Tensor) -> torch.Tensor:
    """0-based ranks (ties keep their arbitrary order, values are continuous)."""
    order = torch.argsort(values)
    ranks = torch.empty(values.numel(), dtype=torch.float64, device=values.device)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float64,
                                device=values.device)
    return ranks


def _correlations(y_pred: torch.Tensor, y_true: torch.Tensor) -> dict:
    """Pearson and Spearman correlation, computed in float64 on the device."""
    pred = y_pred.to(torch.float64)
    true = y_true.to(torch.float64)

    def pearson(a, b):
        a = a - a.mean()
        b = b - b.mean()
        denom = (a.norm() * b.norm()).clamp_min(1e-12)
        return (a * b).sum() / denom

    return {
        "Pearson": float(pearson(pred, true)),
        "Spearman": float(pearson(_rank(y_pred), _rank(y_true))),
    }


def evaluate_model(model, X_test, y_test, batch_size: int = 8192):
    """
    Evaluates a model and returns MAE/MSE/R2 plus Pearson/Spearman.
    """
    y_pred = predict(model, X_test, batch_size=batch_size)
    y_true = y_test.reshape(-1).to(y_pred.dtype)

    error = y_pred - y_true
    ss_res = (error ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()

    results = {
        "MAE": float(error.abs().mean()),
        "MSE": float((error ** 2).mean()),
        "R2": float(1.0 - ss_res / ss_tot),
    }
    results.update(_correlations(y_pred, y_true))
    return results


def save_results(results: dict, results_dir: str = "data/MLresults") -> str:
    """
    Writes the metrics of all trained models into one timestamped JSON file.

    Args:
        results: mapping of model name -> metric dict

    Returns:
        Path of the written file.
    """
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(results_dir, f"{timestamp}_results.json")

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    return save_path

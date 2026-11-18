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

def split_three_way(
    X, y, train_size: float = 0.7, val_size: float = 0.15, seed: int | None = None
):
    """
    Shuffles and splits on the device into train / val / test.

    ``train_size + val_size`` must be < 1; the remainder becomes the test set.

    Returns:
        x_train, x_val, x_test, y_train, y_val, y_test as torch tensors.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device=X.device).manual_seed(seed)

    n = X.shape[0]
    permutation = torch.randperm(n, device=X.device, generator=generator)
    train_idx = permutation[: int(n * train_size)]
    val_idx = permutation[int(n * train_size) : int(n * (train_size + val_size))]
    test_idx = permutation[int(n * (train_size + val_size)) :]

    return (
        X[train_idx], X[val_idx], X[test_idx],
        y[train_idx], y[val_idx], y[test_idx],
    )


def load_split(directory: str, device: str = "cuda") -> tuple:
    """
    Load a train/val/test split from a directory produced by ``scripts.split_dataset``.

    Expects ``{directory}/{train,val,test}.parquet`` and ``{directory}/split_config.json``.

    Returns:
        x_train, x_val, x_test, y_train, y_val, y_test as torch tensors.
    """
    def _load(path: str):
        df = pl.read_parquet(path)
        feature_cols = [c for c in df.columns if c != "heuristic"]
        X = np.concatenate([_column_matrix(df[c]) for c in feature_cols], axis=1)
        y = np.array(df["heuristic"].to_numpy(), dtype=np.float32)
        del df
        return (
            torch.from_numpy(X).to(device, non_blocking=True),
            torch.from_numpy(y).to(device, non_blocking=True),
        )

    x_train, y_train = _load(os.path.join(directory, "train.parquet"))
    x_val,   y_val   = _load(os.path.join(directory, "val.parquet"))
    x_test,  y_test  = _load(os.path.join(directory, "test.parquet"))
    return x_train, x_val, x_test, y_train, y_val, y_test


def resolve_input_path(args_input: str) -> str:
    """
    Resolve ``--input`` to a parquet file path.

    If ``args_input`` points to a directory containing ``split_config.json``
    (produced by ``scripts.split_dataset``), returns the path to ``train.parquet``
    inside that directory.  Otherwise falls back to the existing .pkl → .parquet
    resolution.
    """
    if os.path.isdir(args_input):
        config_path = os.path.join(args_input, "split_config.json")
        if os.path.exists(config_path):
            train_path = os.path.join(args_input, "train.parquet")
            if os.path.exists(train_path):
                return train_path
    # Legacy: .pkl → .parquet
    parquet_path = args_input.replace(".pkl", ".parquet")
    if os.path.isfile(parquet_path):
        return parquet_path
    converted = convert_pickle_to_parquet(args_input)
    return converted


class LinearRegressor:
    """
    Ordinary least squares solved on the GPU.

    The normal equations are accumulated chunk-wise in float64 instead of
    building one huge design matrix: no 2x copy of X and no QR/SVD workspace
    that scales with the number of rows.
    """

    def __init__(self, chunk_size: int = 131072, standardize: bool = True):
        self.chunk_size = chunk_size
        self.standardize = standardize
        self.standardizer = Standardizer() if standardize else None

    def fit(self, X, y):
        num_rows, num_features = X.shape
        device = X.device
        targets = y.reshape(-1, 1)

        if self.standardizer is not None:
            X = self.standardizer.fit_transform(X)

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
        if self.standardizer is not None:
            X = self.standardizer.transform(X)
        return (X @ self.coef_.T + self.intercept_).squeeze(1)


def train_xgboost(X, y, n_estimators: int = 500, max_depth: int = 6,
                  learning_rate: float = 0.1, device: str = "cuda",
                  max_rows: int = 0, standardize: bool = True,
                  subsample: float = 0.8, colsample_bytree: float = 0.8,
                  n_jobs: int = 1):
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
    if n_jobs <= 0:
        n_jobs = 1 if sys.platform == "darwin" else -1

    num_rows = X.shape[0]
    if 0 < max_rows < num_rows:
        print(f"XGBoost: training on a random subset of {max_rows:,} "
              f"of {num_rows:,} rows")
        subset = torch.randperm(num_rows, device=X.device)[:max_rows]
        X, y = X[subset], y[subset]

    # XGBoost benefits from standardization (tree models are invariant to monotonic
    # transformations, but hist-based splitting is numerically more stable with
    # zero-mean, unit-variance features).
    standardizer = Standardizer() if standardize else None
    X_np = _to_numpy(standardizer.fit_transform(X)) if standardizer else _to_numpy(X)
    y_np = _to_numpy(y)

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        tree_method="hist",
        device=device,
        n_jobs=n_jobs,
    )
    model.fit(X_np, y_np)
    model.standardizer = standardizer  # attach for later transform in predict()
    return model


class _Standardize(torch.nn.Module):
    """Feature scaling with statistics taken from the training split."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        return (x - self.mean) / self.std


class Standardizer:
    """
    Computes mean/std from X_train and applies z-score normalization.

    Works on both CPU (numpy) and GPU (torch) tensors. The fitted statistics
    are stored as numpy arrays so the same object can be used to transform
    both CPU-based models (XGBoost) and GPU-based models (Linear, MLP).
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        """Compute mean and std from X. X can be a torch.Tensor or numpy array."""
        if isinstance(X, torch.Tensor):
            self.mean = X.mean(dim=0).cpu().numpy()
            self.std = X.std(dim=0).cpu().numpy()
        else:
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0)
        self.std = np.where(self.std < 1e-6, 1.0, self.std)
        return self

    def transform(self, X):
        """Apply z-score normalization. Returns same type as input."""
        if isinstance(X, torch.Tensor):
            return (X - torch.from_numpy(self.mean).to(X.device)) / torch.from_numpy(self.std).to(X.device)
        else:
            return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def train_mlp(X, y, hidden_layers=(64, 32), epochs: int = 100,
              batch_size: int = 4096, lr: float = 1e-3,
              x_val=None, y_val=None):
    """
    Trains a Multi-Layer Perceptron with PyTorch on the device of `X`.

    The latent features are standardized on the training split; without this
    the network barely beats predicting the mean.

    When ``x_val`` and ``y_val`` are provided, each epoch is evaluated on the
    validation split and the model state with the lowest validation MAE is
    returned (together with a history dict and the best MAE).  Otherwise the
    final model state is returned as before.
    """
    device = X.device
    y = y.reshape(-1, 1)

    mean = X.mean(dim=0)
    std = X.std(dim=0).clamp_min(1e-6)
    standardizer_mean = mean.cpu().numpy()
    standardizer_std = std.cpu().numpy()

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

    X_std = model[0](X)  # pre-standardize train input

    history = None
    best_state = None
    best_mae = float("inf")
    last_state = None

    if x_val is not None and y_val is not None:
        y_val = y_val.reshape(-1, 1)
        X_val_std = model[0](x_val)
        history = []
        n = X.shape[0]

        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                optimizer.zero_grad()
                loss = criterion(model(X_std[idx]), y[idx])
                loss.backward()
                optimizer.step()

            # Validation eval (no grads, eval mode)
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_std).squeeze(1)
            val_error = val_pred - y_val.squeeze(1)
            val_mse = (val_error ** 2).mean().item()
            val_mae = val_error.abs().mean().item()

            # Train eval (no grads, eval mode) for history
            with torch.no_grad():
                train_pred = model(X_std).squeeze(1)
            train_error = train_pred - y.squeeze(1)
            train_mse = (train_error ** 2).mean().item()
            train_mae = train_error.abs().mean().item()

            history.append({
                "epoch": epoch + 1,
                "train_mse": train_mse,
                "train_mae": train_mae,
                "val_mse": val_mse,
                "val_mae": val_mae,
            })

            last_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if val_mae < best_mae:
                best_mae = val_mae
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Return best checkpoint state, not final epoch state
        if best_state is not None:
            model.load_state_dict(best_state)

    else:
        # Original behaviour: train for all epochs, return final state
        model.train()
        for _ in range(epochs):
            perm = torch.randperm(X.shape[0], device=device)
            for start in range(0, X.shape[0], batch_size):
                idx = perm[start:start + batch_size]
                optimizer.zero_grad()
                loss = criterion(model(X_std[idx]), y[idx])
                loss.backward()
                optimizer.step()

    # Attach standardizer stats for later use
    model.standardizer_mean = standardizer_mean
    model.standardizer_std = standardizer_std

    if history is not None:
        return model, history, best_mae
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
    if hasattr(model, "standardizer") and model.standardizer is not None:
        X_np = _to_numpy(model.standardizer.transform(X))
    else:
        X_np = _to_numpy(X)
    pred = model.predict(X_np)
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

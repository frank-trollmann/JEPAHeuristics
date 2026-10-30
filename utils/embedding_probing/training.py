import json
import os
from datetime import datetime

# macOS-only fix for a torch/xgboost conflict
# Not needed on the Spark
# Uncomment both lines when developing locally on a Mac:

# os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from utils.embedding_probing.data.splitter import split_dataset
from utils.embedding_probing.evaluation import evaluate_model
from utils.embedding_probing.models.linear_regression import (
  save_linear_regression,
  train_linear_regression,
)
from utils.embedding_probing.models.mlp.persistence import save_torch_mlp
from utils.embedding_probing.models.mlp.train import predict_torch_mlp, train_torch_mlp
from utils.embedding_probing.models.mlp.tune import mlp_params_to_kwargs, tune_mlp
from utils.embedding_probing.models.random_forest import (
  save_random_forest,
  train_random_forest,
  tune_random_forest,
)
from utils.embedding_probing.models.xgboost_model import (
  save_xgboost,
  train_xgboost,
  tune_xgboost,
)


def _subsample(X, y, sample_size, seed):
  """
  Randomly subsample (X, y) down to `sample_size` rows, without replacement.
  Returns the full (X, y) unchanged if sample_size is None or already >= len(X).
  """
  if sample_size is None or sample_size >= len(X):
    return X, y
  rng = np.random.default_rng(seed)
  idx = rng.choice(len(X), size=sample_size, replace=False)
  return X[idx], y[idx]


def log_experiment(rows, experiment, path=None):
  """
  Append experiment result rows to a persistent CSV log.
  """
  if path is None:
    path = f"experiment_log_{experiment}.csv"
  for row in rows:
    row["experiment"] = experiment
  new_df = pd.DataFrame(rows)
  if os.path.exists(path):
    existing_df = pd.read_csv(path)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
  else:
    combined_df = new_df
  combined_df.to_csv(path, index=False)


def train_and_compare_models(
  X,
  y,
  dataset_size=None,
  n_trials=50,
  device="cpu",
  models_dir="models",
  seed=42,
  n_jobs=None,
  tuning_sample_size=None,
):
  """
  Full pipeline: split -> tune each model on Train/Val -> final fit on
  Train -> evaluate once on held-out Test -> save each model -> return a
  results DataFrame.

  tuning_sample_size: if set, each Optuna trial trains on a random subset of
  this many Train rows
  """
  os.makedirs(models_dir, exist_ok=True)
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

  X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
    X, y, random_state=seed
  )
  X_tune, y_tune = _subsample(X_train, y_train, tuning_sample_size, seed)

  results = []

  # --- Linear Regression ---
  model = train_linear_regression(X_train, y_train)
  metrics = evaluate_model(model, X_test, y_test)
  path = os.path.join(models_dir, f"{timestamp}_linear_regression.joblib")
  save_linear_regression(model, path)
  results.append({"model": "Linear Regression", "params": {}, "model_path": path, **metrics})

  # --- Random Forest ---
  best_params, _ = tune_random_forest(
    X_tune, y_tune, X_val, y_val, n_trials=n_trials, random_state=seed, seed=seed, n_jobs=n_jobs
  )
  model = train_random_forest(X_train, y_train, params=best_params, random_state=seed, n_jobs=n_jobs)
  metrics = evaluate_model(model, X_test, y_test)
  path = os.path.join(models_dir, f"{timestamp}_random_forest.joblib")
  save_random_forest(model, path)
  results.append({"model": "Random Forest", "params": best_params, "model_path": path, **metrics})

  # --- XGBoost ---
  best_params, _ = tune_xgboost(
    X_tune, y_tune, X_val, y_val,
    n_trials=n_trials, device=device, random_state=seed, seed=seed,
  )
  model = train_xgboost(X_train, y_train, params=best_params, device=device, random_state=seed)
  metrics = evaluate_model(model, X_test, y_test)
  path = os.path.join(models_dir, f"{timestamp}_xgboost.json")
  save_xgboost(model, path)
  results.append({"model": "XGBoost", "params": best_params, "model_path": path, **metrics})

  # --- MLP ---
  best_params, _ = tune_mlp(
    X_tune, y_tune, X_val, y_val, n_trials=n_trials, device=device, seed=seed
  )
  mlp_kwargs = mlp_params_to_kwargs(best_params)
  model, _ = train_torch_mlp(
    X_train, y_train, X_val, y_val, device=device, seed=seed, **mlp_kwargs
  )
  metrics = evaluate_model(
    model, X_test, y_test,
    predict_fn=lambda m, X: predict_torch_mlp(m, X, device=device),
  )
  path = os.path.join(models_dir, f"{timestamp}_mlp")
  save_torch_mlp(
    model, path,
    input_dim=X.shape[1],
    hidden_dims=mlp_kwargs["hidden_dims"],
    output_dim=y.shape[1] if y.ndim > 1 else 1,
    dropout=mlp_kwargs["dropout"],
  )
  results.append({"model": "MLP", "params": best_params, "model_path": path, **metrics})

  for row in results:
    row["timestamp"] = timestamp
    row["dataset_size"] = dataset_size if dataset_size is not None else len(X)
    row["n_trials"] = n_trials
    row["device"] = device
    row["n_jobs"] = n_jobs
    row["seed"] = seed
    row["tuning_sample_size"] = tuning_sample_size if tuning_sample_size is not None else len(X_train)
    row["params"] = json.dumps(row["params"])

  return pd.DataFrame(results)

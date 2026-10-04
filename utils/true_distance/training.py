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

from utils.true_distance.data.splitter import split_dataset
from utils.true_distance.evaluation import evaluate_model
from utils.true_distance.models.linear_regression import (
  save_linear_regression,
  train_linear_regression,
)
from utils.true_distance.models.mlp.persistence import save_torch_mlp
from utils.true_distance.models.mlp.train import predict_torch_mlp, train_torch_mlp
from utils.true_distance.models.mlp.tune import mlp_params_to_kwargs, tune_mlp
from utils.true_distance.models.random_forest import (
  save_random_forest,
  train_random_forest,
  tune_random_forest,
)
from utils.true_distance.models.xgboost_model import (
  save_xgboost,
  train_xgboost,
  tune_xgboost,
)


def log_experiment(rows, path="experiment_log.csv"):
  """
  Create Log csv file
  """
  new_df = pd.DataFrame(rows)
  file_exists = os.path.exists(path)
  new_df.to_csv(path, mode="a", header=not file_exists, index=False)


def train_and_compare_models(
  X,
  y,
  dataset_size=None,
  n_trials=50,
  device="cpu",
  models_dir="models",
  seed=42,
):
  """
  Full pipeline: split -> tune each model on Train/Val -> final fit on
  Train -> evaluate once on held-out Test -> save each model -> return a
  results DataFrame (one row per model, also writable via log_experiment).
  """
  os.makedirs(models_dir, exist_ok=True)
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

  X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
    X, y, random_state=seed
  )

  results = []

  # --- Linear Regression ---
  model = train_linear_regression(X_train, y_train)
  metrics = evaluate_model(model, X_test, y_test)
  path = os.path.join(models_dir, f"{timestamp}_linear_regression.joblib")
  save_linear_regression(model, path)
  results.append({"model": "Linear Regression", "params": {}, "model_path": path, **metrics})

  # --- Random Forest ---
  best_params, _ = tune_random_forest(
    X_train, y_train, X_val, y_val, n_trials=n_trials, random_state=seed, seed=seed
  )
  model = train_random_forest(X_train, y_train, params=best_params, random_state=seed)
  metrics = evaluate_model(model, X_test, y_test)
  path = os.path.join(models_dir, f"{timestamp}_random_forest.joblib")
  save_random_forest(model, path)
  results.append({"model": "Random Forest", "params": best_params, "model_path": path, **metrics})

  # --- XGBoost ---
  best_params, _ = tune_xgboost(
    X_train, y_train, X_val, y_val,
    n_trials=n_trials, device=device, random_state=seed, seed=seed,
  )
  model = train_xgboost(X_train, y_train, params=best_params, device=device, random_state=seed)
  metrics = evaluate_model(model, X_test, y_test)
  path = os.path.join(models_dir, f"{timestamp}_xgboost.json")
  save_xgboost(model, path)
  results.append({"model": "XGBoost", "params": best_params, "model_path": path, **metrics})

  # --- MLP ---
  best_params, _ = tune_mlp(
    X_train, y_train, X_val, y_val, n_trials=n_trials, device=device, seed=seed
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
    dropout=mlp_kwargs["dropout"],
  )
  results.append({"model": "MLP", "params": best_params, "model_path": path, **metrics})

  for row in results:
    row["timestamp"] = timestamp
    row["dataset_size"] = dataset_size if dataset_size is not None else len(X)
    row["params"] = json.dumps(row["params"])

  return pd.DataFrame(results)

import os

import numpy as np
import optuna
import pandas as pd

from utils.true_distance.training import log_experiment, train_and_compare_models

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _synthetic_data(n=150, dim=6, seed=0):
  rng = np.random.default_rng(seed)
  X = rng.normal(size=(n, dim)).astype(np.float32)
  y = (X.sum(axis=1) * 10 + rng.normal(scale=0.1, size=n)).astype(np.float32)
  return X, y


def test_train_and_compare_models_returns_one_row_per_model(tmp_path):
  X, y = _synthetic_data()
  models_dir = tmp_path / "models"

  results = train_and_compare_models(
    X, y, n_trials=2, device="cpu", models_dir=str(models_dir), seed=0
  )

  assert set(results["model"]) == {"Linear Regression", "Random Forest", "XGBoost", "MLP"}
  assert len(results) == 4


def test_train_and_compare_models_has_expected_columns(tmp_path):
  X, y = _synthetic_data()
  models_dir = tmp_path / "models"

  results = train_and_compare_models(
    X, y, n_trials=2, device="cpu", models_dir=str(models_dir), seed=0
  )

  expected_cols = {
    "model", "params", "model_path", "MAE", "MSE", "R2", "Pearson",
    "Spearman", "timestamp", "dataset_size",
  }
  assert expected_cols.issubset(set(results.columns))


def test_train_and_compare_models_saves_model_files(tmp_path):
  X, y = _synthetic_data()
  models_dir = tmp_path / "models"

  results = train_and_compare_models(
    X, y, n_trials=2, device="cpu", models_dir=str(models_dir), seed=0
  )

  for _, row in results.iterrows():
    if row["model"] == "MLP":
      assert os.path.exists(row["model_path"] + ".pt")
      assert os.path.exists(row["model_path"] + ".json")
    else:
      assert os.path.exists(row["model_path"])


def test_train_and_compare_models_records_dataset_size(tmp_path):
  X, y = _synthetic_data(n=150)
  models_dir = tmp_path / "models"

  results = train_and_compare_models(
    X, y, dataset_size=150, n_trials=2, device="cpu", models_dir=str(models_dir), seed=0
  )

  assert (results["dataset_size"] == 150).all()


def test_log_experiment_creates_file_with_header(tmp_path):
  log_path = tmp_path / "experiment_log.csv"
  rows = [{"model": "Linear Regression", "MAE": 1.0}]

  log_experiment(rows, path=str(log_path))

  df = pd.read_csv(log_path)
  assert list(df.columns) == ["model", "MAE"]
  assert len(df) == 1


def test_log_experiment_appends_without_duplicating_header(tmp_path):
  log_path = tmp_path / "experiment_log.csv"

  log_experiment([{"model": "Linear Regression", "MAE": 1.0}], path=str(log_path))
  log_experiment([{"model": "Random Forest", "MAE": 0.5}], path=str(log_path))

  df = pd.read_csv(log_path)
  assert len(df) == 2
  assert list(df["model"]) == ["Linear Regression", "Random Forest"]

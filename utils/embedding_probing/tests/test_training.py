import os

import numpy as np
import optuna
import pandas as pd

from utils.embedding_probing.training import _subsample, log_experiment, train_and_compare_models

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _synthetic_data(n=150, dim=6, seed=0):
  rng = np.random.default_rng(seed)
  X = rng.normal(size=(n, dim)).astype(np.float32)
  y = (X.sum(axis=1) * 10 + rng.normal(scale=0.1, size=n)).astype(np.float32)
  return X, y


def _synthetic_multioutput_data(n=150, dim=6, seed=0):
  rng = np.random.default_rng(seed)
  X = rng.normal(size=(n, dim)).astype(np.float32)
  y = np.stack([
    X.sum(axis=1) * 10 + rng.normal(scale=0.1, size=n),
    X[:, 0] * 5 + rng.normal(scale=0.1, size=n),
  ], axis=1).astype(np.float32)
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
    "Spearman", "timestamp", "dataset_size", "n_trials", "device", "n_jobs", "seed",
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


def test_train_and_compare_models_works_with_multioutput_y(tmp_path):
  X, y = _synthetic_multioutput_data()
  models_dir = tmp_path / "models"

  results = train_and_compare_models(
    X, y, n_trials=2, device="cpu", models_dir=str(models_dir), seed=0
  )

  assert len(results) == 4
  assert set(results["model"]) == {"Linear Regression", "Random Forest", "XGBoost", "MLP"}
  # metrics must be plain scalars (averaged across the 2 output columns), not NaN/arrays
  assert results["Pearson"].notna().all()
  assert results["MAE"].apply(np.isscalar).all()

  for _, row in results.iterrows():
    if row["model"] == "MLP":
      assert os.path.exists(row["model_path"] + ".pt")
      assert os.path.exists(row["model_path"] + ".json")
    else:
      assert os.path.exists(row["model_path"]) or os.path.exists(row["model_path"] + ".manifest.json")


def test_subsample_returns_full_data_when_none():
  X, y = _synthetic_data(n=50)
  X_sub, y_sub = _subsample(X, y, None, seed=0)
  np.testing.assert_array_equal(X_sub, X)
  np.testing.assert_array_equal(y_sub, y)


def test_subsample_returns_full_data_when_larger_than_data():
  X, y = _synthetic_data(n=50)
  X_sub, y_sub = _subsample(X, y, 1000, seed=0)
  assert len(X_sub) == 50
  assert len(y_sub) == 50


def test_subsample_returns_requested_size():
  X, y = _synthetic_data(n=1000)
  X_sub, y_sub = _subsample(X, y, 100, seed=0)
  assert len(X_sub) == 100
  assert len(y_sub) == 100


def test_subsample_rows_come_from_original_data():
  X, y = _synthetic_data(n=1000)
  X_sub, y_sub = _subsample(X, y, 100, seed=0)
  # every sampled y value must exist in the original y (no fabricated rows)
  assert set(y_sub.tolist()).issubset(set(y.tolist()))


def test_subsample_reproducible_with_same_seed():
  X, y = _synthetic_data(n=1000)
  X_sub_a, y_sub_a = _subsample(X, y, 100, seed=7)
  X_sub_b, y_sub_b = _subsample(X, y, 100, seed=7)
  np.testing.assert_array_equal(X_sub_a, X_sub_b)
  np.testing.assert_array_equal(y_sub_a, y_sub_b)


def test_train_and_compare_models_respects_tuning_sample_size(tmp_path):
  X, y = _synthetic_data(n=150)
  models_dir = tmp_path / "models"

  results = train_and_compare_models(
    X, y, n_trials=2, device="cpu", models_dir=str(models_dir), seed=0,
    tuning_sample_size=30,
  )

  assert (results["tuning_sample_size"] == 30).all()
  assert len(results) == 4  # still produces a full, usable results table


def test_train_and_compare_models_defaults_tuning_sample_size_to_full_train(tmp_path):
  X, y = _synthetic_data(n=150)
  models_dir = tmp_path / "models"

  results = train_and_compare_models(
    X, y, n_trials=2, device="cpu", models_dir=str(models_dir), seed=0,
  )

  # 150 samples, 75% train split -> 112 train rows
  assert (results["tuning_sample_size"] == 112).all()


def test_log_experiment_creates_file_with_header(tmp_path):
  log_path = tmp_path / "experiment_log.csv"
  rows = [{"model": "Linear Regression", "MAE": 1.0}]

  log_experiment(rows, experiment="true_distance", path=str(log_path))

  df = pd.read_csv(log_path)
  assert set(df.columns) == {"model", "MAE", "experiment"}
  assert len(df) == 1


def test_log_experiment_appends_without_duplicating_header(tmp_path):
  log_path = tmp_path / "experiment_log.csv"

  log_experiment([{"model": "Linear Regression", "MAE": 1.0}], experiment="true_distance", path=str(log_path))
  log_experiment([{"model": "Random Forest", "MAE": 0.5}], experiment="true_distance", path=str(log_path))

  df = pd.read_csv(log_path)
  assert len(df) == 2
  assert list(df["model"]) == ["Linear Regression", "Random Forest"]


def test_log_experiment_stamps_each_row_with_experiment_name(tmp_path):
  log_path = tmp_path / "experiment_log.csv"
  rows = [{"model": "Linear Regression", "MAE": 1.0}, {"model": "Random Forest", "MAE": 0.5}]

  log_experiment(rows, experiment="position_probing", path=str(log_path))

  df = pd.read_csv(log_path)
  assert (df["experiment"] == "position_probing").all()


def test_log_experiment_derives_default_path_from_experiment_name(tmp_path, monkeypatch):
  monkeypatch.chdir(tmp_path)
  rows = [{"model": "Linear Regression", "MAE": 1.0}]

  log_experiment(rows, experiment="obstacle_probing")

  assert (tmp_path / "experiment_log_obstacle_probing.csv").exists()

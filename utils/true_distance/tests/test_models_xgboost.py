import numpy as np
import optuna
from xgboost import XGBRegressor

from utils.true_distance.models.xgboost_model import (
  load_xgboost,
  save_xgboost,
  train_xgboost,
  tune_xgboost,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _synthetic_data(n=100, dim=5, seed=0):
  rng = np.random.default_rng(seed)
  X = rng.normal(size=(n, dim))
  y = X.sum(axis=1) * 10 + rng.normal(scale=0.1, size=n)
  return X, y


def test_train_xgboost_with_defaults_on_cpu():
  X, y = _synthetic_data()
  model = train_xgboost(X, y, device="cpu")
  assert isinstance(model, XGBRegressor)
  assert model.predict(X).shape == (len(y),)


def test_train_xgboost_applies_given_params():
  X, y = _synthetic_data()
  model = train_xgboost(X, y, params={"n_estimators": 7, "max_depth": 2}, device="cpu")
  assert model.n_estimators == 7
  assert model.max_depth == 2


def test_train_xgboost_defaults_to_cpu_device():
  X, y = _synthetic_data()
  model = train_xgboost(X, y)
  assert model.device == "cpu"


def test_tune_xgboost_returns_valid_params_and_study():
  X_train, y_train = _synthetic_data(n=80, seed=1)
  X_val, y_val = _synthetic_data(n=20, seed=2)

  best_params, study = tune_xgboost(
    X_train, y_train, X_val, y_val, n_trials=3, device="cpu", seed=0
  )

  expected_keys = {
    "n_estimators", "max_depth", "learning_rate", "subsample",
    "colsample_bytree", "min_child_weight", "reg_alpha", "reg_lambda",
  }
  assert set(best_params.keys()) == expected_keys
  assert len(study.trials) == 3
  assert study.best_value >= 0  # MAE can't be negative


def test_tune_xgboost_best_params_are_usable():
  X_train, y_train = _synthetic_data(n=80, seed=1)
  X_val, y_val = _synthetic_data(n=20, seed=2)

  best_params, _ = tune_xgboost(X_train, y_train, X_val, y_val, n_trials=3, device="cpu", seed=0)

  model = train_xgboost(X_train, y_train, params=best_params, device="cpu")
  assert model.predict(X_val).shape == (20,)


def test_save_and_load_xgboost_roundtrip(tmp_path):
  X, y = _synthetic_data()
  model = train_xgboost(X, y, params={"n_estimators": 5}, device="cpu")

  path = str(tmp_path / "xgboost_model.json")
  save_xgboost(model, path)
  loaded = load_xgboost(path, device="cpu")

  assert isinstance(loaded, XGBRegressor)
  np.testing.assert_allclose(loaded.predict(X), model.predict(X))

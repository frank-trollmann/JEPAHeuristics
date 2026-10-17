import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor

from utils.true_distance.models.random_forest import (
  load_random_forest,
  save_random_forest,
  train_random_forest,
  tune_random_forest,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _synthetic_data(n=100, dim=5, seed=0):
  rng = np.random.default_rng(seed)
  X = rng.normal(size=(n, dim))
  y = X.sum(axis=1) * 10 + rng.normal(scale=0.1, size=n)
  return X, y


def test_train_random_forest_with_defaults():
  X, y = _synthetic_data()
  model = train_random_forest(X, y)
  assert isinstance(model, RandomForestRegressor)
  assert hasattr(model, "estimators_")  # only set after fit()


def test_train_random_forest_applies_given_params():
  X, y = _synthetic_data()
  model = train_random_forest(X, y, params={"n_estimators": 7, "max_depth": 2})
  assert model.n_estimators == 7
  assert model.max_depth == 2


def test_tune_random_forest_returns_valid_params_and_study():
  X_train, y_train = _synthetic_data(n=80, seed=1)
  X_val, y_val = _synthetic_data(n=20, seed=2)

  best_params, study = tune_random_forest(
    X_train, y_train, X_val, y_val, n_trials=3, seed=0
  )

  expected_keys = {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features"}
  assert set(best_params.keys()) == expected_keys
  assert len(study.trials) == 3
  assert study.best_value >= 0  # MAE can't be negative


def test_tune_random_forest_best_params_are_usable():
  X_train, y_train = _synthetic_data(n=80, seed=1)
  X_val, y_val = _synthetic_data(n=20, seed=2)

  best_params, _ = tune_random_forest(X_train, y_train, X_val, y_val, n_trials=3, seed=0)

  # the returned params should fit without error
  model = train_random_forest(X_train, y_train, params=best_params)
  assert model.predict(X_val).shape == (20,)


def test_save_and_load_random_forest_roundtrip(tmp_path):
  X, y = _synthetic_data()
  model = train_random_forest(X, y, params={"n_estimators": 5})

  path = tmp_path / "random_forest.joblib"
  save_random_forest(model, path)
  loaded = load_random_forest(path)

  assert isinstance(loaded, RandomForestRegressor)
  np.testing.assert_allclose(loaded.predict(X), model.predict(X))

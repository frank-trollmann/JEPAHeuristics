import json

import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor


def train_xgboost(X_train, y_train, params=None, device="cpu", random_state=42):
  """
  Fit an XGBRegressor with the given hyperparameters.
  """
  params = params or {}
  model = XGBRegressor(
    tree_method="hist",
    device=device,
    random_state=random_state,
    **params,
  )
  y_train = np.asarray(y_train)
  # for multi output, wrap model in MultiOutputRegressor (independant Regressor per output column)
  if y_train.ndim > 1: 
    model = MultiOutputRegressor(model)
  model.fit(X_train, y_train)
  return model


def _objective(trial, X_train, y_train, X_val, y_val, device, random_state):
  params = {
    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
    "max_depth": trial.suggest_int("max_depth", 2, 12),
    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
  }
  model = train_xgboost(X_train, y_train, params, device=device, random_state=random_state)
  y_pred = model.predict(X_val)
  return mean_absolute_error(y_val, y_pred)


def tune_xgboost(X_train, y_train, X_val, y_val, n_trials=50, device="cpu", random_state=42, seed=None):
  """
  Optuna hyperparameter search for XGBRegressor
  """
  sampler = optuna.samplers.TPESampler(seed=seed)
  study = optuna.create_study(direction="minimize", sampler=sampler)
  study.optimize(
    lambda trial: _objective(trial, X_train, y_train, X_val, y_val, device, random_state),
    n_trials=n_trials,
  )
  return study.best_params, study


def save_xgboost(model, path):
  """
  Save a fitted model to disk.
  MultiOutputRegressor has no native .save_model()
  """
  if isinstance(model, MultiOutputRegressor):
    manifest = {"multi_output": True, "n_outputs": len(model.estimators_)}
    for i, estimator in enumerate(model.estimators_):
      estimator.save_model(f"{path}.output{i}.json")
  else:
    manifest = {"multi_output": False}
    model.save_model(path)

  with open(f"{path}.manifest.json", "w") as f:
    json.dump(manifest, f)


def load_xgboost(path, device="cpu"):
  with open(f"{path}.manifest.json") as f:
    manifest = json.load(f)

  if manifest["multi_output"]:
    model = MultiOutputRegressor(XGBRegressor(device=device))
    estimators = []
    for i in range(manifest["n_outputs"]):
      estimator = XGBRegressor(device=device)
      estimator.load_model(f"{path}.output{i}.json")
      estimators.append(estimator)
    model.estimators_ = estimators
    return model

  model = XGBRegressor(device=device)
  model.load_model(path)
  return model

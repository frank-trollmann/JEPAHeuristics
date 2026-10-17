import optuna
from sklearn.metrics import mean_absolute_error
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
  model.save_model(path)


def load_xgboost(path, device="cpu"):
  model = XGBRegressor(device=device)
  model.load_model(path)
  return model

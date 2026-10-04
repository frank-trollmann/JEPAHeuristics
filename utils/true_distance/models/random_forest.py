import joblib
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def train_random_forest(X_train, y_train, params=None, random_state=42):
  """
  Fit a RandomForestRegressor with the given hyperparameters
  """
  params = params or {}
  model = RandomForestRegressor(random_state=random_state, **params)
  model.fit(X_train, y_train)
  return model


def _objective(trial, X_train, y_train, X_val, y_val, random_state):
  params = {
    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
    "max_depth": trial.suggest_int("max_depth", 3, 30),
    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
  }
  model = train_random_forest(X_train, y_train, params, random_state=random_state)
  y_pred = model.predict(X_val)
  return mean_absolute_error(y_val, y_pred)


def tune_random_forest(X_train, y_train, X_val, y_val, n_trials=50, random_state=42, seed=None):
  """
  Optuna hyperparameter search for RandomForestRegressor
  """
  sampler = optuna.samplers.TPESampler(seed=seed)
  study = optuna.create_study(direction="minimize", sampler=sampler)
  study.optimize(
    lambda trial: _objective(trial, X_train, y_train, X_val, y_val, random_state),
    n_trials=n_trials,
  )
  return study.best_params, study


def save_random_forest(model, path):
  joblib.dump(model, path)


def load_random_forest(path):
  return joblib.load(path)

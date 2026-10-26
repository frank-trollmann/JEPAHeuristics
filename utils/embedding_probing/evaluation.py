import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr


def _mean_correlation(y_true, y_pred, corr_fn):
  """
  pearsonr/spearmanr only accept 1D arrays. For multi-output y (x, y),
  compute the correlation per output column and average
  """
  if y_true.ndim == 1:
    return corr_fn(y_true, y_pred)[0]
  per_column = [corr_fn(y_true[:, i], y_pred[:, i])[0] for i in range(y_true.shape[1])]
  return float(np.mean(per_column))


def evaluate_predictions(y_true, y_pred):
  """
  Compute MAE, MSE, R2, Pearson, Spearman between true and predicted values.
  Returns a dict - the same metrics used throughout this project.
  """
  y_true = np.asarray(y_true, dtype=float)
  y_pred = np.asarray(y_pred, dtype=float)

  return {
    "MAE": mean_absolute_error(y_true, y_pred),
    "MSE": mean_squared_error(y_true, y_pred),
    "R2": r2_score(y_true, y_pred),
    "Pearson": _mean_correlation(y_true, y_pred, pearsonr),
    "Spearman": _mean_correlation(y_true, y_pred, spearmanr),
  }


def evaluate_model(model, X_test, y_test, predict_fn=None):
  if predict_fn is None:
    y_pred = model.predict(X_test)
  else:
    y_pred = predict_fn(model, X_test)

  return evaluate_predictions(y_test, y_pred)

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr


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
    "Pearson": pearsonr(y_true, y_pred)[0],
    "Spearman": spearmanr(y_true, y_pred)[0],
  }


def evaluate_model(model, X_test, y_test, predict_fn=None):
  if predict_fn is None:
    y_pred = model.predict(X_test)
  else:
    y_pred = predict_fn(model, X_test)

  return evaluate_predictions(y_test, y_pred)

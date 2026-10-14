"""
Probe: how well can the position (x, y) be read from a single encoding?

If it can be read almost exactly, high scores for pair-based distance models are
partly explained by "the position is easy to read and the map never changes".
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_position_probe(encodings, positions, hidden=None, seed=0):
  """
  Fits a model from encodings (N, D) to positions (N, 2).
  hidden=None gives a linear model, otherwise an MLP with these layer sizes, e.g. (256, 256).
  """
  if hidden is None:
    model = make_pipeline(StandardScaler(), LinearRegression())
  else:
    model = make_pipeline(
      StandardScaler(),
      MLPRegressor(hidden_layer_sizes=hidden, early_stopping=True, max_iter=300, random_state=seed),
    )
  return model.fit(encodings, positions)


def position_errors(model, encodings, positions):
  """Mean absolute error in pixels along x, along y, and the mean distance between predicted and true position."""
  error = model.predict(encodings) - positions
  return {
    "mae_x": np.abs(error[:, 0]).mean(),
    "mae_y": np.abs(error[:, 1]).mean(),
    "mean_position_error": np.linalg.norm(error, axis=1).mean(),
  }

import numpy as np
from sklearn.linear_model import LinearRegression

from utils.true_distance.models.linear_regression import (
  load_linear_regression,
  save_linear_regression,
  train_linear_regression,
)


def test_train_linear_regression_returns_fitted_model():
  model = train_linear_regression(
    X_train=np.array([[0.0], [1.0], [2.0], [3.0]]),
    y_train=np.array([0.0, 2.0, 4.0, 6.0]),
  )
  assert isinstance(model, LinearRegression)
  assert hasattr(model, "coef_")  # only set after fit()


def test_train_linear_regression_recovers_known_linear_relationship():
  # y = 2*x1 + 3*x2 + 1, noiseless -> should fit (almost) exactly
  rng = np.random.default_rng(0)
  X = rng.uniform(-10, 10, size=(200, 2))
  y = 2 * X[:, 0] + 3 * X[:, 1] + 1

  model = train_linear_regression(X, y)
  y_pred = model.predict(X)

  np.testing.assert_allclose(y_pred, y, atol=1e-8)


def test_save_and_load_linear_regression_roundtrip(tmp_path):
  X = np.array([[0.0], [1.0], [2.0], [3.0]])
  y = np.array([0.0, 2.0, 4.0, 6.0])
  model = train_linear_regression(X, y)

  path = tmp_path / "linear_regression.joblib"
  save_linear_regression(model, path)
  loaded = load_linear_regression(path)

  assert isinstance(loaded, LinearRegression)
  np.testing.assert_allclose(loaded.predict(X), model.predict(X))

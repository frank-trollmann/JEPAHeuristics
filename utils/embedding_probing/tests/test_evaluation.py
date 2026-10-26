import numpy as np
import pytest

from utils.embedding_probing.evaluation import evaluate_predictions, evaluate_model


def test_evaluate_predictions_perfect_match():
  y = np.array([10.0, 20.0, 30.0, 40.0])
  metrics = evaluate_predictions(y, y)

  assert metrics["MAE"] == pytest.approx(0.0)
  assert metrics["MSE"] == pytest.approx(0.0)
  assert metrics["R2"] == pytest.approx(1.0)
  assert metrics["Pearson"] == pytest.approx(1.0)
  assert metrics["Spearman"] == pytest.approx(1.0)


def test_evaluate_predictions_known_mae_mse():
  y_true = np.array([0.0, 0.0, 0.0, 0.0])
  y_pred = np.array([1.0, -1.0, 2.0, -2.0])

  metrics = evaluate_predictions(y_true, y_pred)

  assert metrics["MAE"] == pytest.approx(1.5)  # mean(|1|,|-1|,|2|,|-2|)
  assert metrics["MSE"] == pytest.approx(2.5)  # mean(1,1,4,4)


def test_evaluate_predictions_returns_all_expected_keys():
  y = np.array([1.0, 2.0, 3.0])
  metrics = evaluate_predictions(y, y)
  assert set(metrics.keys()) == {"MAE", "MSE", "R2", "Pearson", "Spearman", "L2Error"}


def test_evaluate_predictions_worse_than_mean_gives_negative_r2():
  y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
  # constant prediction far from the data -> worse than just predicting the mean
  y_pred = np.full_like(y_true, 1000.0)

  metrics = evaluate_predictions(y_true, y_pred)
  assert metrics["R2"] < 0


class _StubModel:
  """Minimal stand-in for a fitted sklearn-style model."""

  def predict(self, X):
    return np.asarray(X).sum(axis=1)


def test_evaluate_model_uses_predict_by_default():
  model = _StubModel()
  X_test = np.array([[1.0, 1.0], [2.0, 2.0]])
  y_test = np.array([2.0, 4.0])  # matches sum(axis=1) exactly

  metrics = evaluate_model(model, X_test, y_test)
  assert metrics["MAE"] == pytest.approx(0.0)


def test_evaluate_predictions_multioutput_perfect_match():
  y = np.array([[10.0, 1.0], [20.0, 2.0], [30.0, 3.0], [40.0, 4.0]])
  metrics = evaluate_predictions(y, y)

  assert metrics["MAE"] == pytest.approx(0.0)
  assert metrics["R2"] == pytest.approx(1.0)
  assert metrics["Pearson"] == pytest.approx(1.0)
  assert metrics["Spearman"] == pytest.approx(1.0)
  assert metrics["L2Error"] == pytest.approx(0.0)


def test_evaluate_predictions_l2_error_equals_mae_for_single_output():
  y_true = np.array([0.0, 0.0, 0.0, 0.0])
  y_pred = np.array([1.0, -1.0, 2.0, -2.0])

  metrics = evaluate_predictions(y_true, y_pred)

  # L2 norm of a scalar is its absolute value - L2Error must exactly
  # reduce to MAE for a single-output target, not just be close to it
  assert metrics["L2Error"] == pytest.approx(metrics["MAE"])


def test_evaluate_predictions_l2_error_is_euclidean_distance_for_multioutput():
  # true points at the origin, predictions offset by a 3-4-5 triangle
  y_true = np.array([[0.0, 0.0], [0.0, 0.0]])
  y_pred = np.array([[3.0, 4.0], [3.0, 4.0]])

  metrics = evaluate_predictions(y_true, y_pred)

  assert metrics["L2Error"] == pytest.approx(5.0)


def test_evaluate_predictions_l2_error_catches_combined_error_mae_hides():
  # two small per-axis errors (MAE looks tiny) combine into one bigger
  # Euclidean miss that a position-prediction use case actually cares about
  # (pearsonr/spearmanr need >=2 samples, hence two identical rows here)
  y_true = np.array([[0.0, 0.0], [0.0, 0.0]])
  y_pred = np.array([[3.0, 4.0], [3.0, 4.0]])

  metrics = evaluate_predictions(y_true, y_pred)

  assert metrics["MAE"] == pytest.approx(3.5)  # mean(3, 4)
  assert metrics["L2Error"] == pytest.approx(5.0)  # sqrt(3^2 + 4^2)


def test_evaluate_predictions_multioutput_averages_per_column_correlation():
  # column 0: perfectly correlated, column 1: perfectly anti-correlated
  y_true = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
  y_pred = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])

  metrics = evaluate_predictions(y_true, y_pred)

  # mean(1.0, -1.0) == 0.0
  assert metrics["Pearson"] == pytest.approx(0.0)
  assert metrics["Spearman"] == pytest.approx(0.0)


def test_evaluate_model_uses_custom_predict_fn():
  model = object()  # doesn't even need .predict, since we override it
  X_test = np.array([[1.0], [2.0], [3.0]])
  y_test = np.array([10.0, 20.0, 30.0])

  def predict_fn(model, X):
    return np.asarray(X).flatten() * 10.0

  metrics = evaluate_model(model, X_test, y_test, predict_fn=predict_fn)
  assert metrics["MAE"] == pytest.approx(0.0)

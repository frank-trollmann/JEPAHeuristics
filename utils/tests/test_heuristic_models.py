import numpy as np

from utils.heuristic_models import fit_linear, fit_mean_baseline, latent_distance, metrics, train_mlp


def test_metrics_perfect_and_mean_predictions():
  y = np.array([1.0, 2.0, 3.0, 4.0])

  perfect = metrics(y, y)
  mean_only = metrics(y, np.full(4, y.mean()))

  assert perfect["mae"] == 0 and perfect["r2"] == 1
  assert abs(mean_only["r2"]) < 1e-9


def test_mean_baseline_predicts_the_training_mean():
  model = fit_mean_baseline(np.array([2.0, 4.0, 9.0]))

  assert np.allclose(model.predict(np.zeros((5, 1))), 5.0)


def test_latent_distance_squared_and_plain():
  encodings = np.array([[0.0, 0.0], [3.0, 4.0]])
  agent, goal = np.array([0]), np.array([1])

  assert latent_distance(encodings, agent, goal, squared=False).tolist() == [[5.0]]
  assert latent_distance(encodings, agent, goal, squared=True).tolist() == [[25.0]]


def test_mlp_learns_a_distance_that_a_linear_model_cannot():
  rng = np.random.default_rng(0)
  x = rng.uniform(-1, 1, size=(4000, 4))  # agent (x, y) and goal (x, y)
  y = np.linalg.norm(x[:, :2] - x[:, 2:], axis=1)
  train, val = slice(0, 3000), slice(3000, 4000)

  predict = train_mlp(x[train], y[train], x[val], y[val], device="cpu", seed=0, epochs=40)
  linear = fit_linear(x[train], y[train])

  assert metrics(y[val], predict(x[val]))["r2"] > 0.9
  assert metrics(y[val], linear.predict(x[val]))["r2"] < 0.5

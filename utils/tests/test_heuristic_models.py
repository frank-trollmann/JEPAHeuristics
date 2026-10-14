import numpy as np

from utils.heuristic_models import fit_linear, fit_mean_baseline, latent_distance, metrics, train_mlp
from utils.pair_dataset import concat_features


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
  encodings = rng.uniform(-1, 1, size=(300, 2))  # stand-ins for encodings: 2-D points
  agent, goal = rng.integers(0, 300, 4000), rng.integers(0, 300, 4000)
  label = np.linalg.norm(encodings[agent] - encodings[goal], axis=1)
  train, val = slice(0, 3000), slice(3000, 4000)

  predict = train_mlp(encodings, (agent[train], goal[train], label[train]),
                      (agent[val], goal[val], label[val]), device="cpu", seed=0, epochs=40)
  linear = fit_linear(concat_features(encodings, agent[train], goal[train]), label[train])

  assert metrics(label[val], predict(agent[val], goal[val]))["r2"] > 0.9
  linear_prediction = linear.predict(concat_features(encodings, agent[val], goal[val]))
  assert metrics(label[val], linear_prediction)["r2"] < 0.5

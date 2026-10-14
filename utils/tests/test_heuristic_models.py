import numpy as np
import pytest

from utils.heuristic_models import (
  fit_linear,
  fit_linear_in_chunks,
  fit_mean_baseline,
  latent_distance,
  MODEL_KINDS,
  metrics,
  train_pair_model,
)
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


def test_chunked_least_squares_matches_linear_regression():
  rng = np.random.default_rng(0)
  encodings = rng.normal(size=(200, 6))
  agent, goal = rng.integers(0, 200, 1000), rng.integers(0, 200, 1000)
  label = rng.normal(size=1000) + concat_features(encodings, agent, goal) @ rng.normal(size=12)
  new_agent, new_goal = rng.integers(0, 200, 50), rng.integers(0, 200, 50)

  expected = fit_linear(concat_features(encodings, agent, goal), label).predict(
    concat_features(encodings, new_agent, new_goal))
  chunked = fit_linear_in_chunks(lambda a, g: concat_features(encodings, a, g), agent, goal, label, chunk=300)

  assert np.allclose(chunked(concat_features(encodings, new_agent, new_goal)), expected)


def make_distance_task():
  rng = np.random.default_rng(0)
  encodings = rng.uniform(-1, 1, size=(300, 2))  # stand-ins for encodings: 2-D points
  agent, goal = rng.integers(0, 300, 4000), rng.integers(0, 300, 4000)
  label = np.linalg.norm(encodings[agent] - encodings[goal], axis=1)
  train, val = slice(0, 3000), slice(3000, 4000)
  return encodings, (agent[train], goal[train], label[train]), (agent[val], goal[val], label[val])


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_every_model_learns_a_distance_that_a_linear_model_cannot(kind):
  encodings, train, val = make_distance_task()

  predict = train_pair_model(encodings, train, val, device="cpu", seed=0, kind=kind, epochs=40)
  linear = fit_linear(concat_features(encodings, train[0], train[1]), train[2])

  assert metrics(val[2], predict(val[0], val[1]))["r2"] > 0.9
  assert metrics(val[2], linear.predict(concat_features(encodings, val[0], val[1])))["r2"] < 0.5


def test_metric_model_is_symmetric_and_zero_on_identical_encodings():
  encodings, train, val = make_distance_task()

  predict = train_pair_model(encodings, train, val, device="cpu", seed=0, kind="metric", epochs=5)

  agent, goal = np.arange(50), np.arange(50, 100)
  assert np.allclose(predict(agent, goal), predict(goal, agent), atol=1e-5)
  assert np.allclose(predict(agent, agent), 0, atol=1e-3)


def test_on_latents_gives_the_same_predictions_as_the_indexed_call():
  import torch

  encodings, train, val = make_distance_task()
  predictor = train_pair_model(encodings, train, val, device="cpu", seed=0, kind="mlp", epochs=3)

  agent, goal = np.arange(40), np.arange(40, 80)
  from_indices = predictor(agent, goal)
  from_latents = predictor.on_latents(
    torch.tensor(encodings[agent], dtype=torch.float32), torch.tensor(encodings[goal], dtype=torch.float32)).numpy()

  assert np.allclose(from_indices, from_latents, atol=1e-4)

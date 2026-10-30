import numpy as np
import pandas as pd

from utils.embedding_probing.experiments.true_distance.extraction import heuristic_dataset_to_xy


def _fake_dataset(n=100, dim=192, seed=0):
  rng = np.random.default_rng(seed)
  return pd.DataFrame({
    "agent_pos_enc": [rng.normal(size=dim) for _ in range(n)],
    "goal_pos_enc": [rng.normal(size=dim) for _ in range(n)],
    "heuristic": rng.uniform(0, 300, size=n),
  })


def test_heuristic_dataset_to_xy_shapes():
  df = _fake_dataset(n=50, dim=192)
  X, y = heuristic_dataset_to_xy(df)
  assert X.shape == (50, 384)
  assert y.shape == (50,)


def test_heuristic_dataset_to_xy_concatenation_order():
  df = _fake_dataset(n=3, dim=4)
  X, y = heuristic_dataset_to_xy(df)
  np.testing.assert_allclose(X[:, :4], np.array(df["agent_pos_enc"].tolist()))
  np.testing.assert_allclose(X[:, 4:], np.array(df["goal_pos_enc"].tolist()))

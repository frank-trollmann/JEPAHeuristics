import numpy as np
import pandas as pd

from utils.embedding_probing.experiments.position_probing.extraction import position_dataset_to_xy


def _fake_position_dataset(n=100, dim=192, seed=0):
  rng = np.random.default_rng(seed)
  return pd.DataFrame({
    "agent_pos_enc": [rng.normal(size=dim) for _ in range(n)],
    "goal_pos_enc": [rng.normal(size=dim) for _ in range(n)],
    "agent_pos": [rng.uniform(0, 224, size=2) for _ in range(n)],
    "goal_pos": [rng.uniform(0, 224, size=2) for _ in range(n)],
  })


def test_position_dataset_to_xy_shapes():
  df = _fake_position_dataset(n=50, dim=192)
  X, y = position_dataset_to_xy(df)
  assert X.shape == (100, 192)  # agent + goal embeddings, stacked
  assert y.shape == (100, 2)


def test_position_dataset_to_xy_matches_embedding_to_position():
  df = _fake_position_dataset(n=20, dim=8)
  X, y = position_dataset_to_xy(df)

  agent_encs = np.array(df["agent_pos_enc"].tolist())
  agent_pos = np.array(df["agent_pos"].tolist())

  for i, enc in enumerate(agent_encs):
    match = np.where((X == enc).all(axis=1))[0]
    assert len(match) == 1
    np.testing.assert_allclose(y[match[0]], agent_pos[i])


def test_position_dataset_to_xy_deduplicates_identical_embeddings():
  df = _fake_position_dataset(n=10, dim=8)
  # force a duplicate: goal embedding of row 0 == agent embedding of row 0
  df.at[0, "goal_pos_enc"] = df.at[0, "agent_pos_enc"]
  df.at[0, "goal_pos"] = df.at[0, "agent_pos"]

  X, y = position_dataset_to_xy(df)

  # 10 agent + 10 goal embeddings, minus 1 duplicate = 19 unique rows
  assert X.shape[0] == 19
  assert y.shape[0] == 19

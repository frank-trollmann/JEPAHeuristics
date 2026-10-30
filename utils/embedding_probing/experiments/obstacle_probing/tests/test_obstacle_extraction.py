import numpy as np
import pandas as pd

from utils.embedding_probing.experiments.obstacle_probing.extraction import obstacle_dataset_to_xy


def _fake_obstacle_dataset(n=100, dim=192, seed=0):
  rng = np.random.default_rng(seed)
  return pd.DataFrame({
    "agent_pos_enc": [rng.normal(size=dim) for _ in range(n)],
    "goal_pos_enc": [rng.normal(size=dim) for _ in range(n)],
    "door_y": rng.uniform(0, 224, size=n),
    "door_size": rng.uniform(1, 21, size=n),
  })


def test_obstacle_dataset_to_xy_shapes():
  df = _fake_obstacle_dataset(n=50, dim=192)
  X, y = obstacle_dataset_to_xy(df)
  assert X.shape == (100, 192)  # agent + goal embeddings, stacked
  assert y.shape == (100, 2)


def test_obstacle_dataset_to_xy_matches_embedding_to_door_geometry():
  df = _fake_obstacle_dataset(n=20, dim=8)
  X, y = obstacle_dataset_to_xy(df)

  agent_encs = np.array(df["agent_pos_enc"].tolist())
  door_geom = df[["door_y", "door_size"]].values

  for i, enc in enumerate(agent_encs):
    match = np.where((X == enc).all(axis=1))[0]
    assert len(match) == 1
    np.testing.assert_allclose(y[match[0]], door_geom[i])


def test_obstacle_dataset_to_xy_same_geometry_for_agent_and_goal_embedding():
  df = _fake_obstacle_dataset(n=5, dim=8)
  X, y = obstacle_dataset_to_xy(df)

  agent_encs = np.array(df["agent_pos_enc"].tolist())
  goal_encs = np.array(df["goal_pos_enc"].tolist())

  for i in range(len(df)):
    agent_idx = np.where((X == agent_encs[i]).all(axis=1))[0][0]
    goal_idx = np.where((X == goal_encs[i]).all(axis=1))[0][0]
    np.testing.assert_allclose(y[agent_idx], y[goal_idx])


def test_obstacle_dataset_to_xy_deduplicates_identical_embeddings():
  df = _fake_obstacle_dataset(n=10, dim=8)
  # force a duplicate: goal embedding of row 0 == agent embedding of row 0
  df.at[0, "goal_pos_enc"] = df.at[0, "agent_pos_enc"]

  X, y = obstacle_dataset_to_xy(df)

  # 10 agent + 10 goal embeddings, minus 1 duplicate = 19 unique rows
  assert X.shape[0] == 19
  assert y.shape[0] == 19

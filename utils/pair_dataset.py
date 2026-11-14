"""
(agent, goal) pairs built from a position dataset (see position_dataset.py).

A pair is a pair of indices into the stored positions and encodings. Both indices
of a pair come from the same split, so no position is shared between splits.
"""
import numpy as np


def load_position_dataset(path):
  """Returns positions (N, 2), encodings (N, D) and splits (N,) from a saved .npz file."""
  data = np.load(path)
  return data["positions"], data["encodings"], data["splits"]


def sample_pairs(splits, split, num_pairs, seed):
  """
  Draws num_pairs random (agent, goal) index pairs, both from the given split.
  Returns two index arrays: agent indices and goal indices.
  """
  candidates = np.flatnonzero(splits == split)
  rng = np.random.default_rng(seed)
  return rng.choice(candidates, num_pairs), rng.choice(candidates, num_pairs)


def concat_features(encodings, agent, goal):
  """Model input: the agent and goal encodings side by side, shape (num_pairs, 2 * D)."""
  return np.concatenate([encodings[agent], encodings[goal]], axis=1)


def euclidean_labels(positions, agent, goal):
  """Straight-line distance in pixels between the agent and goal coordinates."""
  return np.linalg.norm(positions[agent] - positions[goal], axis=1)


def crosses_wall(positions, agent, goal, wall_center):
  """
  True where agent and goal are in different rooms.
  Assumes the default vertical wall at x = wall_center (TwoRoomEnv.WALL_CENTER).
  """
  return (positions[agent, 0] < wall_center) != (positions[goal, 0] < wall_center)

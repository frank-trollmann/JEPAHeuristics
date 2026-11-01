import numpy as np


def position_dataset_to_xy(df):
  """
  Create DF from dataset

  X: (N, 192) - single embeddings (agent and goal, deduplicated)
  y: (N, 2) - corresponding true (x, y) position
  """
  agent_encs = np.array(df["agent_pos_enc"].tolist(), dtype=float)
  goal_encs = np.array(df["goal_pos_enc"].tolist(), dtype=float)
  agent_pos = np.array(df["agent_pos"].tolist(), dtype=float)
  goal_pos = np.array(df["goal_pos"].tolist(), dtype=float)

  X = np.vstack([agent_encs, goal_encs])
  y = np.vstack([agent_pos, goal_pos])

  X_unique, idx = np.unique(X, axis=0, return_index=True)
  return X_unique, y[idx]

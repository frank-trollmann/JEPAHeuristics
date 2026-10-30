import numpy as np


def heuristic_dataset_to_xy(df):
  """
  Create DF from dataset

  X: (N, 384) - agent_pos_enc and goal_pos_enc concatenated
  y: (N,) - distance label
  """
  agent_encs = np.array(df["agent_pos_enc"].tolist(), dtype=float)
  goal_encs = np.array(df["goal_pos_enc"].tolist(), dtype=float)
  X = np.hstack([agent_encs, goal_encs])
  y = df["heuristic"].values.astype(float)
  return X, y

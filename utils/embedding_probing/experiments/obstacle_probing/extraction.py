import numpy as np


def obstacle_dataset_to_xy(df):
  """
  Create DF from dataset

  X: (N, 192) - single embeddings (agent and goal, deduplicated)
  y: (N, 2) - (door_y, door_size) for that sample - the same for both the
  agent and goal embedding of a given row
  """
  agent_encs = np.array(df["agent_pos_enc"].tolist(), dtype=float)
  goal_encs = np.array(df["goal_pos_enc"].tolist(), dtype=float)
  door_geom = df[["door_y", "door_size"]].values.astype(float)

  X = np.vstack([agent_encs, goal_encs])
  y = np.vstack([door_geom, door_geom])

  X_unique, idx = np.unique(X, axis=0, return_index=True)
  return X_unique, y[idx]

import numpy as np
from sklearn.model_selection import train_test_split


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


def split_dataset(X, y, train_size=0.75, val_size=0.10, test_size=0.15, random_state=42):
  """
  train, val, test  
  """

  # should be 1 
  assert abs(train_size + val_size + test_size - 1.0) < 1e-9, (
    "train_size + val_size + test_size must sum to 1.0"
  )

  X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, train_size=train_size, random_state=random_state
  )

  # split the remaining (val+test) chunk proportionally
  relative_val_size = val_size / (val_size + test_size)
  X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, train_size=relative_val_size, random_state=random_state
  )

  return X_train, X_val, X_test, y_train, y_val, y_test

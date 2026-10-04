import numpy as np
import pandas as pd
import pytest

from utils.true_distance.data.splitter import heuristic_dataset_to_xy, split_dataset


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


def test_split_dataset_sizes():
  X = np.arange(1000).reshape(-1, 1).astype(float)
  y = np.arange(1000).astype(float)

  X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
    X, y, train_size=0.75, val_size=0.10, test_size=0.15
  )

  assert len(X_train) == 750
  assert len(X_val) == 100
  assert len(X_test) == 150
  assert len(y_train) == 750
  assert len(y_val) == 100
  assert len(y_test) == 150


def test_split_dataset_no_overlap():
  X = np.arange(500).reshape(-1, 1).astype(float)
  y = np.arange(500).astype(float)

  X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

  train_ids = set(X_train[:, 0].tolist())
  val_ids = set(X_val[:, 0].tolist())
  test_ids = set(X_test[:, 0].tolist())

  assert train_ids.isdisjoint(val_ids)
  assert train_ids.isdisjoint(test_ids)
  assert val_ids.isdisjoint(test_ids)
  assert train_ids | val_ids | test_ids == set(range(500))


def test_split_dataset_rejects_proportions_not_summing_to_one():
  X = np.arange(10).reshape(-1, 1).astype(float)
  y = np.arange(10).astype(float)
  with pytest.raises(AssertionError):
    split_dataset(X, y, train_size=0.5, val_size=0.2, test_size=0.2)


def test_split_dataset_reproducible_with_same_seed():
  X = np.arange(200).reshape(-1, 1).astype(float)
  y = np.arange(200).astype(float)

  result_a = split_dataset(X, y, random_state=7)
  result_b = split_dataset(X, y, random_state=7)

  for a, b in zip(result_a, result_b):
    np.testing.assert_array_equal(a, b)

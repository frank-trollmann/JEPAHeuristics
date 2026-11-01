import numpy as np
import pytest

from utils.embedding_probing.data.splitter import split_dataset


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

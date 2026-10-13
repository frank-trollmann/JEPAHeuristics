import numpy as np

from utils.pair_dataset import concat_features, crosses_wall, euclidean_labels, sample_pairs


def test_pairs_stay_inside_their_split():
  splits = np.array(["train"] * 60 + ["val"] * 20 + ["test"] * 20)

  for split in ("train", "val", "test"):
    agent, goal = sample_pairs(splits, split, num_pairs=500, seed=0)
    assert (splits[agent] == split).all()
    assert (splits[goal] == split).all()


def test_pairs_are_reproducible_per_seed():
  splits = np.array(["train"] * 50)

  first = sample_pairs(splits, "train", 100, seed=5)
  again = sample_pairs(splits, "train", 100, seed=5)
  other = sample_pairs(splits, "train", 100, seed=6)

  assert all((a == b).all() for a, b in zip(first, again))
  assert (first[0] != other[0]).any()


def test_euclidean_label_is_the_norm_of_the_coordinate_difference():
  positions = np.array([[0, 0], [3, 4], [10, 0]])

  labels = euclidean_labels(positions, agent=np.array([0, 0, 1]), goal=np.array([1, 2, 2]))

  assert np.allclose(labels, [5.0, 10.0, np.hypot(7, 4)])


def test_crosses_wall_only_between_rooms():
  positions = np.array([[50, 10], [100, 200], [150, 10], [112, 10]])

  result = crosses_wall(positions, agent=np.array([0, 0, 2]), goal=np.array([1, 2, 2]), wall_center=112)

  assert result.tolist() == [False, True, False]


def test_concat_features_places_agent_before_goal():
  encodings = np.array([[1.0, 2.0], [3.0, 4.0]])

  features = concat_features(encodings, agent=np.array([0]), goal=np.array([1]))

  assert features.tolist() == [[1.0, 2.0, 3.0, 4.0]]

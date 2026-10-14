import numpy as np
import pytest
from stable_worldmodel.envs.two_room import TwoRoomEnv

from utils.pair_dataset import euclidean_labels
from utils.position_dataset import valid_positions
from utils.reference_distance import build_grid_graph, grid_labels


def distances_between(positions, pairs):
  graph = build_grid_graph(positions)
  agent = np.array([a for a, _ in pairs])
  goal = np.array([g for _, g in pairs])
  return grid_labels(graph, agent, goal)


def test_open_grid_uses_sideways_and_diagonal_steps():
  positions = np.array([(x, y) for x in range(5) for y in range(5)])
  index = {tuple(p): i for i, p in enumerate(positions)}

  result = distances_between(positions, [
    (index[(0, 0)], index[(4, 0)]),
    (index[(0, 0)], index[(4, 4)]),
    (index[(0, 0)], index[(4, 2)]),
    (index[(2, 2)], index[(2, 2)]),
  ])

  assert np.allclose(result, [4.0, 4 * 2 ** 0.5, 2 * 2 ** 0.5 + 2, 0.0])


def test_path_must_go_around_a_wall():
  # A U-shaped corridor: (0,0) and (2,0) are 2 apart in a straight line, but 6 steps by the path.
  positions = np.array([(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0)])

  result = distances_between(positions, [(0, 6)])

  assert np.allclose(result, [6.0])


def test_diagonal_does_not_cut_a_wall_corner():
  # (0,0) and (1,1) touch diagonally, but (1,0) and (0,1) are missing, so they are not connected.
  positions = np.array([(0, 0), (1, 1)])

  result = distances_between(positions, [(0, 1)])

  assert np.isinf(result[0])


def test_distances_are_symmetric():
  positions = np.array([(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0)])

  forward = distances_between(positions, [(0, 6), (1, 5)])
  backward = distances_between(positions, [(6, 0), (5, 1)])

  assert np.allclose(forward, backward)


@pytest.fixture(scope="module")
def two_room():
  positions = valid_positions(TwoRoomEnv())
  return positions, build_grid_graph(positions)


def test_two_room_map_is_fully_connected_and_never_shorter_than_straight_line(two_room):
  positions, graph = two_room
  rng = np.random.default_rng(0)
  agent = rng.integers(0, len(positions), 2000)
  goal = rng.integers(0, len(positions), 2000)

  grid = grid_labels(graph, agent, goal)

  assert np.isfinite(grid).all()
  assert (grid >= euclidean_labels(positions, agent, goal) - 1e-6).all()


def test_crossing_far_from_the_door_needs_a_long_detour(two_room):
  positions, graph = two_room
  index = {tuple(p): i for i, p in enumerate(positions.tolist())}
  agent, goal = index[(40, 200)], index[(190, 200)]  # door is near y = 49, so this pair is far from it

  grid = grid_labels(graph, np.array([agent]), np.array([goal]))[0]
  straight = euclidean_labels(positions, np.array([agent]), np.array([goal]))[0]

  assert grid > 1.3 * straight

import numpy as np
import pytest
from stable_worldmodel.envs.two_room import TwoRoomEnv

from utils.pair_dataset import euclidean_labels
from utils.position_dataset import valid_positions
from utils.reference_distance import build_grid_graph, door_geometry, formula_labels, grid_labels


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


# --- continuous formula -----------------------------------------------------

@pytest.fixture(scope="module")
def geometries():
  env = TwoRoomEnv()
  env.reset()
  return door_geometry(env, inflated=True), door_geometry(env, inflated=False)


def formula(a, g, geometry):
  return formula_labels(np.array([a], dtype=float), np.array([g], dtype=float), geometry)[0]


def test_formula_is_the_straight_line_when_nothing_blocks_it(geometries):
  grown, thin = geometries

  for geometry in geometries:
    assert formula((50, 150), (60, 120), geometry) == pytest.approx(np.hypot(10, 30))  # same room
    assert formula((50, 49), (170, 49), geometry) == pytest.approx(120.0)  # straight through the door


def test_thin_formula_bends_around_the_door_edge_exactly(geometries):
  _, thin = geometries

  # the door edge is at (112, 63): the path is two straight pieces
  expected = np.hypot(112 - 50, 150 - 63) + np.hypot(170 - 112, 150 - 63)
  assert formula((50, 150), (170, 150), thin) == pytest.approx(expected, abs=0.01)


def test_grown_wall_formula_lies_between_two_independent_bounds(geometries):
  grown, _ = geometries
  # lower bound: a point agent going around the wall pixel centres (107, 64) and (117, 64)
  lower = np.hypot(107 - 50, 150 - 64) + 10 + np.hypot(170 - 117, 150 - 64)
  # upper bound: the same path around the corners of the square grown wall (100, 56) and (124, 56)
  upper = np.hypot(100 - 50, 150 - 56) + 24 + np.hypot(170 - 124, 150 - 56)

  assert lower < formula((50, 150), (170, 150), grown) < upper


def test_formula_is_symmetric_and_the_vertex_paths_are_finite(geometries):
  grown, thin = geometries

  for geometry in geometries:
    assert np.isfinite(geometry["paths"]).all()
    assert np.allclose(geometry["paths"], geometry["paths"].T)
  assert formula((50, 150), (170, 90), grown) == pytest.approx(formula((170, 90), (50, 150), grown))


def test_formula_agrees_with_the_grid_label_on_the_real_map(two_room, geometries):
  positions, graph = two_room
  grown, _ = geometries
  rng = np.random.default_rng(0)
  agent = rng.integers(0, len(positions), 2000)
  goal = rng.integers(0, len(positions), 2000)

  grid = grid_labels(graph, agent, goal)
  continuous = formula_labels(positions[agent].astype(float), positions[goal].astype(float), grown)
  relative = (grid - continuous) / np.maximum(continuous, 1)

  assert np.isfinite(continuous).all()
  assert (continuous >= euclidean_labels(positions, agent, goal) - 1e-6).all()
  # 8-direction steps can only make paths longer, by at most about 10%
  assert relative.min() > -0.01 and relative.max() < 0.12

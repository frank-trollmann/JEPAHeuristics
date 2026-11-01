import numpy as np
import pytest

from utils.reference_distance import build_grid_graph
from utils.search import PLANNERS, make_adjacency, search


def open_grid(size):
  positions = np.array([(x, y) for x in range(size) for y in range(size)])
  index = {tuple(p): i for i, p in enumerate(positions)}
  return positions, index, make_adjacency(build_grid_graph(positions))


def octile(positions, goal_position):
  """Exact distance on an open grid with 8-direction steps: a lower bound that never overestimates."""
  dx, dy = np.abs(positions - goal_position).T
  return np.maximum(dx, dy) + (2 ** 0.5 - 1) * np.minimum(dx, dy)


def test_every_planner_finds_a_path_on_an_open_grid():
  positions, index, adjacency = open_grid(6)
  start, goal = index[(0, 0)], index[(5, 3)]
  h = octile(positions, positions[goal])

  for planner in PLANNERS:
    result = search(adjacency, start, goal, h, planner)
    assert result["found"] and result["cost"] >= octile(positions, positions[goal])[start] - 1e-9


def test_astar_with_an_admissible_heuristic_is_optimal_and_expands_fewer_nodes_than_without():
  positions, index, adjacency = open_grid(15)
  start, goal = index[(0, 0)], index[(14, 9)]
  optimal = octile(positions, positions[goal])[start]

  informed = search(adjacency, start, goal, octile(positions, positions[goal]), "astar")
  uninformed = search(adjacency, start, goal, np.zeros(len(positions)), "astar")

  assert informed["cost"] == pytest.approx(optimal)
  assert uninformed["cost"] == pytest.approx(optimal)
  assert informed["expansions"] < uninformed["expansions"]


def test_greedy_search_with_a_misleading_heuristic_still_finds_the_goal_but_expands_more():
  # A U-shaped corridor: (0, 0) and (2, 0) are close in a straight line, but 6 steps apart on the path.
  positions = np.array([(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0)])
  adjacency = make_adjacency(build_grid_graph(positions))
  straight_line = np.linalg.norm(positions - positions[6], axis=1)

  result = search(adjacency, 0, 6, straight_line, "greedy")

  assert result["found"] and result["cost"] == pytest.approx(6.0)
  assert result["expansions"] == 7  # it walks into the dead end of the U before it can leave it


def test_unreachable_goal_is_reported():
  positions = np.array([(0, 0), (1, 0), (5, 5)])
  adjacency = make_adjacency(build_grid_graph(positions))

  for planner in PLANNERS:
    assert not search(adjacency, 0, 2, np.zeros(3), planner)["found"]


def test_astar_reopens_a_node_when_a_cheaper_path_is_found():
  # S=0, A=1, B=2, X=3, G=4. X is reached directly at cost 4 and expanded early, because its heuristic is 0.
  # The cheaper way S-A-B-X costs 3 and is only found later. The heuristic never overestimates but is
  # inconsistent (h(B) = 6 > cost(B, X) + h(X) = 1), which is the case where re-opening matters.
  from scipy.sparse import csr_matrix

  rows, cols, costs = [0, 1, 2, 0, 3], [1, 2, 3, 3, 4], [1.0, 1.0, 1.0, 4.0, 5.0]
  adjacency = make_adjacency(csr_matrix((costs, (rows, cols)), shape=(5, 5)))
  heuristic = np.array([0.0, 7.0, 6.0, 0.0, 0.0])

  result = search(adjacency, 0, 4, heuristic, "astar")

  assert result["cost"] == pytest.approx(8.0)  # without re-opening X, the path S-X-G (cost 9) would be returned


def test_recording_returns_the_expansion_order_and_a_path_that_adds_up_to_the_cost():
  positions, index, adjacency = open_grid(8)
  start, goal = index[(0, 0)], index[(7, 4)]
  h = octile(positions, positions[goal])

  for planner in ("greedy", "astar"):
    result = search(adjacency, start, goal, h, planner, record=True)

    assert len(result["expanded"]) == result["expansions"]
    assert result["expanded"][0] == start and result["expanded"][-1] == goal
    path = result["path"]
    assert path[0] == start and path[-1] == goal
    steps = np.linalg.norm(np.diff(positions[path], axis=0), axis=1)
    assert steps.sum() == pytest.approx(result["cost"])


def test_recording_does_not_change_the_search():
  positions, index, adjacency = open_grid(10)
  start, goal = index[(0, 0)], index[(9, 6)]
  h = octile(positions, positions[goal])

  for planner in PLANNERS:
    plain = search(adjacency, start, goal, h, planner)
    recorded = search(adjacency, start, goal, h, planner, record=True)

    assert (plain["found"], plain["cost"], plain["expansions"]) == (recorded["found"], recorded["cost"], recorded["expansions"])

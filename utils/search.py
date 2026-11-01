"""
Graph search for the planning experiment: depth-first, greedy best-first and A*.

The graph is the grid of valid positions (see reference_distance.py). A heuristic is given as one
number per node for the current goal, so the planners never see how the numbers were produced.
"""
import heapq

import numpy as np

PLANNERS = ("dfs", "greedy", "astar")


def make_adjacency(graph):
  """Both directions of every edge, as a CSR matrix. build_grid_graph lists each edge once."""
  return (graph + graph.T).tocsr()


def search(adjacency, start, goal, heuristic, planner, record=False):
  """
  Finds a path from start to goal (node indices).

  dfs: ignores the heuristic and goes deep first (neighbours in a fixed order).
  greedy: expands the node with the smallest heuristic value.
  astar: expands the node with the smallest cost so far plus heuristic value.

  Returns a dict: found, cost (length of the path that was found), expansions (nodes taken off the frontier).
  With record=True it also returns expanded (the nodes in the order they were expanded) and path (the nodes
  from start to goal, or None for dfs).
  """
  if planner not in PLANNERS:
    raise ValueError(f"unknown planner: {planner}")
  indptr, indices, weights = adjacency.indptr, adjacency.indices, adjacency.data

  expansions = 0
  expanded = []

  def result(found, cost, path=None):
    outcome = {"found": found, "cost": cost, "expansions": expansions}
    if record:
      outcome.update(expanded=expanded, path=path)
    return outcome

  if planner == "dfs":
    stack, visited = [(start, 0.0)], set()
    while stack:
      node, cost = stack.pop()
      if node in visited:
        continue
      visited.add(node)
      expansions += 1
      expanded.append(node)
      if node == goal:
        return result(True, cost)
      for k in range(indptr[node + 1] - 1, indptr[node] - 1, -1):  # reversed, so the first neighbour is explored first
        if indices[k] not in visited:
          stack.append((indices[k], cost + weights[k]))
    return result(False, np.inf)

  best_cost = {start: 0.0}  # cost of the best known path to each node
  parent = {start: None}  # the node each node was reached from, to rebuild the path
  # Ties in the priority go to the smaller heuristic for A* (the common choice: it dives toward the goal
  # instead of spreading sideways) and to the smaller cost for greedy search.
  frontier = [(heuristic[start], 0.0, 0.0, start)]
  closed = set()
  while frontier:
    _, _, cost, node = heapq.heappop(frontier)
    if node in closed or cost > best_cost[node]:
      continue
    closed.add(node)
    expansions += 1
    expanded.append(node)
    if node == goal:
      path = []
      while node is not None:
        path.append(node)
        node = parent[node]
      return result(True, cost, path[::-1])
    for k in range(indptr[node], indptr[node + 1]):
      neighbour, new_cost = indices[k], cost + weights[k]
      if planner == "greedy":
        if neighbour in closed or neighbour in best_cost:
          continue
        priority, tie = heuristic[neighbour], new_cost
      else:
        if new_cost >= best_cost.get(neighbour, np.inf):
          continue
        closed.discard(neighbour)  # a cheaper path was found: allow expanding it again (the heuristic may be inconsistent)
        priority, tie = new_cost + heuristic[neighbour], heuristic[neighbour]
      best_cost[neighbour] = new_cost
      parent[neighbour] = node
      heapq.heappush(frontier, (priority, tie, new_cost, neighbour))
  return result(False, np.inf)

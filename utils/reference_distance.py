"""
Wall-aware reference distance: the shortest path length on the grid of valid positions.

Each valid position is a node. It connects to its 8 neighbours that are also valid.
A sideways step costs 1 and a diagonal step costs sqrt(2). A diagonal step is not
allowed to cut the corner of a wall, so both cells next to it must be valid too.
Distances are measured in pixels, like the Euclidean label.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# One direction of each neighbour pair; the graph is treated as undirected.
_STEPS = ((1, 0, 1.0), (0, 1, 1.0), (1, 1, 2 ** 0.5), (1, -1, 2 ** 0.5))


def build_grid_graph(positions):
  """Sparse graph over positions (N, 2): node i is positions[i], edge weights are step costs."""
  size = positions.max(axis=0) + 2
  node_at_cell = np.full(tuple(size), -1)
  node_at_cell[positions[:, 0], positions[:, 1]] = np.arange(len(positions))

  def node_at(xs, ys):
    """Node index at each cell, or -1 for invalid cells and cells outside the grid."""
    inside = (xs >= 0) & (ys >= 0) & (xs < size[0]) & (ys < size[1])
    nodes = np.full(len(xs), -1)
    nodes[inside] = node_at_cell[xs[inside], ys[inside]]
    return nodes

  x, y = positions[:, 0], positions[:, 1]
  starts, ends, costs = [], [], []
  for dx, dy, cost in _STEPS:
    neighbour = node_at(x + dx, y + dy)
    connected = neighbour >= 0
    if dx and dy:
      connected &= (node_at(x + dx, y) >= 0) & (node_at(x, y + dy) >= 0)
    starts.append(np.flatnonzero(connected))
    ends.append(neighbour[connected])
    costs.append(np.full(connected.sum(), cost))

  n = len(positions)
  return csr_matrix((np.concatenate(costs), (np.concatenate(starts), np.concatenate(ends))), shape=(n, n))


def grid_labels(graph, agent, goal, chunk=500):
  """
  Shortest-path distance for each (agent, goal) index pair.
  Runs Dijkstra from each distinct agent, in chunks to bound memory.
  Returns a float array; unreachable pairs are inf.
  """
  labels = np.empty(len(agent))
  sources = np.unique(agent)
  for start in range(0, len(sources), chunk):
    batch = sources[start:start + chunk]
    distances = dijkstra(graph, directed=False, indices=batch)  # (len(batch), N)
    in_batch = np.isin(agent, batch)
    labels[in_batch] = distances[np.searchsorted(batch, agent[in_batch]), goal[in_batch]]
  return labels

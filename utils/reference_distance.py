"""
Wall-aware reference distances, in pixels, like the Euclidean label.

grid_labels: the shortest path on the grid of valid positions. Each valid position is a node. It
connects to its 8 neighbours that are also valid. A sideways step costs 1 and a diagonal step costs
sqrt(2). A diagonal step is not allowed to cut the corner of a wall, so both cells next to it must
be valid too.

formula_labels: the shortest path in the plane around the door (see the section below).
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


# ---------------------------------------------------------------------------
# Continuous formula: shortest path in the plane around the door.
#
# The agent is treated as a point and the wall as an obstacle grown by the agent radius, which gives
# it rounded corners (approximated by short straight pieces). The shortest path between two points is
# a straight line if it does not cross the grown wall, and otherwise it bends around the corner
# vertices of the grown wall at the door.
# ---------------------------------------------------------------------------
THIN_WALL_HALF_WIDTH = 1e-3  # a wall without thickness still needs a tiny width for the crossing test
ARC_SEGMENTS = 4  # straight pieces per quarter circle; 4 keeps the arc error below 0.15 px for radius 7
_FAR = 1e4  # stands in for "infinitely far" along the wall
_TOLERANCE = 1e-7  # rounding noise below this counts as zero in the crossing test


def _grown_upper_wall(x_left, x_right, lip, radius):
  """
  Vertices of the convex obstacle for the part of the wall above the door, grown by radius.
  The wall spans x_left..x_right and ends at y = lip (smaller y is farther from the door).
  Returns the vertices in order, and a boolean mask of those next to the door (the bend points).
  """
  def arc(cx, start, stop):
    angles = np.linspace(start, stop, ARC_SEGMENTS + 1)
    return [(cx + radius * np.cos(t), lip + radius * np.sin(t)) for t in angles]

  if radius == 0:
    gap_side = [(x_left, lip), (x_right, lip)]
  else:
    gap_side = arc(x_left, np.pi, np.pi / 2) + arc(x_right, np.pi / 2, 0)
  vertices = np.array([(gap_side[0][0], -_FAR)] + gap_side + [(gap_side[-1][0], -_FAR)])
  bend = np.zeros(len(vertices), dtype=bool)
  bend[1:-1] = True
  return vertices, bend


def _half_planes(vertices):
  """Outward normals (E, 2) and offsets (E,) of a convex polygon: a point x is inside if normals @ x < offsets."""
  edges = np.roll(vertices, -1, axis=0) - vertices
  area = (vertices[:, 0] * np.roll(vertices[:, 1], -1) - np.roll(vertices[:, 0], -1) * vertices[:, 1]).sum()
  normals = np.stack([edges[:, 1], -edges[:, 0]], axis=1) * (1 if area > 0 else -1)
  normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
  return normals, (normals * vertices).sum(axis=1)


def _crosses(p, q, obstacle):
  """True where the segment p-q passes through the inside of a convex obstacle. p and q are (..., 2)."""
  normals, offsets = obstacle
  direction = q - p
  rate = direction @ normals.T  # how fast each half-plane constraint is approached along the segment
  room = offsets - p @ normals.T  # distance to each boundary from p
  parallel = np.abs(rate) < _TOLERANCE
  with np.errstate(divide="ignore", invalid="ignore"):
    t = room / rate
  leave = np.minimum(np.where(~parallel & (rate > 0), t, np.inf).min(axis=-1), 1.0)
  enter = np.maximum(np.where(~parallel & (rate < 0), t, -np.inf).max(axis=-1), 0.0)
  on_boundary_line = (parallel & (room < _TOLERANCE)).any(axis=-1)  # runs along or outside an edge: no crossing
  return (leave - enter > _TOLERANCE) & ~on_boundary_line


def _free(p, q, obstacles):
  """True where the segment p-q crosses none of the obstacles."""
  free = np.ones(np.broadcast_shapes(p.shape, q.shape)[:-1], dtype=bool)
  for obstacle in obstacles:
    free &= ~_crosses(p, q, obstacle)
  return free


def door_geometry(env, inflated):
  """
  Grown wall obstacles and the shortest paths between their bend vertices, for the single door of the
  vertical wall. Call env.reset() first.
  inflated=True: wall thickness and agent radius are respected (the wall spans the centres of its pixels).
  inflated=False: a wall without thickness and a point agent.
  """
  assert env.wall_axis == 1 and env.num_doors == 1, "only a vertical wall with one door is supported"
  radius = float(env.variation_space["agent"]["radius"].value.item()) if inflated else 0.0
  half = env.wall_thickness // 2 if inflated else THIN_WALL_HALF_WIDTH
  door, size = float(env.door_positions[0]), float(env.door_sizes[0])
  edge = size + 1 if inflated else size  # distance from the door centre to the last wall pixel centre

  upper, bend = _grown_upper_wall(env.WALL_CENTER - half, env.WALL_CENTER + half, door - edge, radius)
  lower = upper.copy()
  lower[:, 1] = 2 * door - lower[:, 1]  # mirror about the door centre
  obstacles = [_half_planes(upper), _half_planes(lower)]
  nodes = np.concatenate([upper[bend], lower[bend]])

  # all-pairs shortest paths between the bend vertices (Floyd-Warshall on the visibility graph)
  straight = np.linalg.norm(nodes[:, None] - nodes[None], axis=2)
  paths = np.where(_free(nodes[:, None], nodes[None], obstacles), straight, np.inf)
  for k in range(len(nodes)):
    paths = np.minimum(paths, paths[:, k:k + 1] + paths[k:k + 1, :])
  return {"obstacles": obstacles, "nodes": nodes, "paths": paths}


def formula_labels(agent_xy, goal_xy, geometry, chunk=20_000):
  """
  Shortest-path length around the door for float coordinates (N, 2), in pixels.
  Never shorter than the straight line, and equal to it when nothing blocks the line.
  """
  obstacles, nodes, paths = geometry["obstacles"], geometry["nodes"], geometry["paths"]

  labels = np.empty(len(agent_xy))
  for start in range(0, len(agent_xy), chunk):
    a, g = agent_xy[start:start + chunk], goal_xy[start:start + chunk]

    to_node_a = np.linalg.norm(a[:, None] - nodes[None], axis=2)  # (n, V)
    to_node_g = np.linalg.norm(g[:, None] - nodes[None], axis=2)
    sees_a = _free(a[:, None], nodes[None], obstacles)
    sees_g = _free(g[:, None], nodes[None], obstacles)

    # agent -> vertex i -> (shortest path) -> vertex j -> goal; i == j is a path around a single vertex
    via = to_node_a[:, :, None] + paths[None] + to_node_g[:, None, :]
    via = np.where(sees_a[:, :, None] & sees_g[:, None, :], via, np.inf).min(axis=(1, 2))

    labels[start:start + chunk] = np.where(_free(a, g, obstacles), np.linalg.norm(a - g, axis=1), via)
  return labels

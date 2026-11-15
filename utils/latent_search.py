"""
Best-first search through the JEPA predictor.

A node is a latent that the predictor produced; an edge is one block (BLOCK environment steps) of one of
eight fixed directions. The predictor gives the child latent, a heuristic cost(latents) gives its estimated
remaining distance in pixels, and greedy / (weighted) A* decide which node to expand next. The search starts
from the encoding of the real current frame, returns the path to the most promising node, and run_episode
executes its first blocks before searching again (model-predictive control, like CEM).

The planner has the same call signature as cem_planner.plan, so run_episode can use either.
"""
import heapq

import numpy as np
import torch

from utils.cem_planner import ACTION_DIM
from utils.dynamics_probe import BLOCK, normalised_block

DIRECTIONS = np.array([(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)], dtype=float)
PX_PER_STEP = 5.0  # measured in the environment: one full-speed action moves 5 px along each axis
KINDS = ("greedy", "astar")


def make_search_planner(kind, action_mean, action_std, weight=1.0, expansions=100, max_depth=12,
                        merge_distance=12.0, stop_at=10.0):
  """
  kind: "greedy" (priority h) or "astar" (priority g + weight * h; g is the path length so far in px).
  expansions: nodes expanded per search; max_depth: blocks in the longest path; merge_distance: a predicted
  latent closer than this to an existing node (Euclidean, latent units) is dropped as a duplicate (it is a
  revisited position; the first path to it is kept, so A* here is not guaranteed to find the best path);
  stop_at: stop as soon as a node's estimated distance is below this (px).
  The cost passed to the returned planner must be in px.
  """
  if kind not in KINDS:
    raise ValueError(f"unknown search kind: {kind}")
  lengths = BLOCK * PX_PER_STEP * np.linalg.norm(DIRECTIONS, axis=1)
  moves = torch.tensor(np.stack([normalised_block(d, action_mean, action_std) for d in DIRECTIONS]), dtype=torch.float32)

  def planner(model, latent, cost, generator=None, device=None, horizon=None, num_samples=None):
    history = model.predictor.num_frames
    moves_here = moves.to(latent.device)
    with torch.no_grad():
      # per node: window of the last latents, window of the actions that led to them, h, g, parent, move
      first = cost(latent[None])[0].item()
      nodes = [(latent[None], latent.new_zeros(0, ACTION_DIM), first, 0.0, -1, -1, 0)]
      seen = latent[None]
      frontier = [(first, 0.0, 0)]
      best, expanded = 0, 0
      while frontier and expanded < expansions:
        _, _, index = heapq.heappop(frontier)
        emb, acts, h, g, _, _, depth = nodes[index]
        if h < stop_at:
          best = index
          break
        if depth >= max_depth:
          continue
        expanded += 1
        window = emb[None].expand(len(moves_here), -1, -1)
        action = torch.cat([acts[None].expand(len(moves_here), -1, -1), moves_here[:, None]], dim=1)
        child = model.predict(window, model.action_encoder(action))[:, -1]
        estimates = cost(child)
        for m in range(len(moves_here)):
          if torch.cdist(child[m][None], seen).min() < merge_distance:
            continue
          h_child, g_child = estimates[m].item(), g + lengths[m]
          nodes.append((torch.cat([emb, child[m][None]])[-history:], action[m][-(history - 1):] if history > 1 else action[m][:0],
                        h_child, g_child, index, m, depth + 1))
          seen = torch.cat([seen, child[m][None]])
          priority = h_child if kind == "greedy" else g_child + weight * h_child
          heapq.heappush(frontier, (priority, g_child if kind == "greedy" else h_child, len(nodes) - 1))
      else:
        # budget used up: take the node with the smallest estimated distance seen so far
        if len(nodes) > 1:
          best = min(range(1, len(nodes)), key=lambda i: nodes[i][2])
        if nodes[best][2] >= first:  # nothing looks better than here: still take the best first move
          best = min((i for i in range(1, len(nodes)) if nodes[i][4] == 0), key=lambda i: nodes[i][2], default=0)

    path = []
    while nodes[best][4] != -1:
      path.append(nodes[best][5])
      best = nodes[best][4]
    if not path:  # nothing to do: stand still for one block
      return torch.tensor(normalised_block((0.0, 0.0), action_mean, action_std), dtype=torch.float32, device=latent.device)[None]
    return moves_here[path[::-1]]

  return planner


def _stand_still(latent, action_mean, action_std):
  return torch.tensor(normalised_block((0.0, 0.0), action_mean, action_std), dtype=torch.float32, device=latent.device)[None]


def make_beam_planner(action_mean, action_std, width=5, max_depth=12, merge_distance=12.0, stop_at=10.0):
  """
  Beam search through the predictor: every level expands all nodes of the beam (8 moves each, one predictor
  batch), drops duplicates (latents closer than merge_distance to an earlier node), keeps the `width` children
  with the smallest estimated distance. Returns the path to the best node seen (greedy is width 1 with
  backtracking, A* is unlimited width with a path cost). The cost must be in px.
  """
  moves = torch.tensor(np.stack([normalised_block(d, action_mean, action_std) for d in DIRECTIONS]), dtype=torch.float32)

  def planner(model, latent, cost, generator=None, device=None, horizon=None, num_samples=None):
    history, k = model.predictor.num_frames, len(moves)
    moves_here = moves.to(latent.device)
    parents, best = [None], (float("inf"), 0, 0)  # per level: (index in previous level, move) of each beam node
    with torch.no_grad():
      if cost(latent[None])[0].item() < stop_at:
        return _stand_still(latent, action_mean, action_std)
      emb, acts, seen = latent[None, None], latent.new_zeros(1, 0, ACTION_DIM), latent[None]
      for level in range(1, max_depth + 1):
        w = emb.shape[0]
        window = emb[:, None].expand(w, k, *emb.shape[1:]).reshape(w * k, *emb.shape[1:])
        past = acts[:, None].expand(w, k, *acts.shape[1:]).reshape(w * k, *acts.shape[1:])
        action = torch.cat([past, moves_here.repeat(w, 1)[:, None]], dim=1)
        child = model.predict(window, model.action_encoder(action))[:, -1]
        h = cost(child)
        keep = []
        for i in torch.argsort(h).tolist():
          if len(keep) == width:
            break
          if torch.cdist(child[i][None], seen).min() < merge_distance:
            continue
          keep.append(i)
          seen = torch.cat([seen, child[i][None]])
        if not keep:
          break
        index = torch.tensor(keep, device=latent.device)
        parents.append([(i // k, i % k) for i in keep])
        emb = torch.cat([window[index], child[index][:, None]], dim=1)[:, -history:]
        acts = action[index][:, -(history - 1):] if history > 1 else action[index][:, :0]
        top = int(h[index].argmin())
        if h[index][top] < best[0]:
          best = (h[index][top].item(), level, top)
        if best[0] < stop_at:
          break
    path, level, j = [], best[1], best[2]
    while level > 0:
      j, move = parents[level][j]
      path.append(move)
      level -= 1
    return moves_here[path[::-1]] if path else _stand_still(latent, action_mean, action_std)

  return planner


def make_mcts_planner(action_mean, action_std, simulations=100, c=0.7, max_depth=12, stop_at=10.0):
  """
  Monte Carlo tree search (UCT) through the predictor over the 8 directions. One simulation walks down the
  tree by UCT (exploit: smaller mean estimated distance of what was backed up through the child, scaled to
  the range seen so far; explore: c * sqrt(ln N / n)), expands the leaf it reaches (8 children, one predictor
  batch) and backs up the leaf's best child estimate. There is no rollout: the heuristic stands in for it.
  The plan follows the most visited children, or the path to a node under stop_at if one was found.
  There is no duplicate merging (it is a tree). The cost must be in px.
  """
  moves = torch.tensor(np.stack([normalised_block(d, action_mean, action_std) for d in DIRECTIONS]), dtype=torch.float32)

  def planner(model, latent, cost, generator=None, device=None, horizon=None, num_samples=None):
    history = model.predictor.num_frames
    moves_here = moves.to(latent.device)
    with torch.no_grad():
      first = cost(latent[None])[0].item()
      if first < stop_at:
        return _stand_still(latent, action_mean, action_std)
      # per node: window of latents, window of actions, h, depth, parent, move, children, visits, sum of backed-up h
      emb, acts, hs, depth = [latent[None]], [latent.new_zeros(0, ACTION_DIM)], [first], [0]
      parent, move, kids, visits, total = [-1], [-1], [None], [0], [0.0]
      lo, hi, found = first, first, None
      for _ in range(simulations):
        node, path = 0, [0]
        while kids[node] is not None:
          n = sum(visits[k] for k in kids[node]) + 1
          scale = max(hi - lo, 1e-6)
          node = max(kids[node], key=lambda k: float("inf") if visits[k] == 0 else
                     (hi - total[k] / visits[k]) / scale + c * (np.log(n) / visits[k]) ** 0.5)
          path.append(node)
        if depth[node] < max_depth and found is None:
          window = emb[node][None].expand(len(moves_here), -1, -1)
          action = torch.cat([acts[node][None].expand(len(moves_here), -1, -1), moves_here[:, None]], dim=1)
          child = model.predict(window, model.action_encoder(action))[:, -1]
          estimates = cost(child).tolist()
          kids[node] = []
          for m in range(len(moves_here)):
            emb.append(torch.cat([emb[node], child[m][None]])[-history:])
            acts.append(action[m][-(history - 1):] if history > 1 else action[m][:0])
            hs.append(estimates[m])
            depth.append(depth[node] + 1)
            parent.append(node)
            move.append(m)
            kids.append(None)
            visits.append(0)
            total.append(0.0)
            kids[node].append(len(hs) - 1)
          value = min(estimates)
          lo, hi = min(lo, value), max(hi, max(estimates))
          if value < stop_at:
            found = kids[node][int(np.argmin(estimates))]
        else:
          value = hs[node]
        for visited in path:
          visits[visited] += 1
          total[visited] += value
    if found is not None:
      node, path = found, []
      while parent[node] != -1:
        path.append(move[node])
        node = parent[node]
      return moves_here[path[::-1]]
    path, node = [], 0
    while kids[node] and max(visits[k] for k in kids[node]) > 0:
      node = max(kids[node], key=lambda k: visits[k])
      path.append(move[node])
    return moves_here[path] if path else _stand_still(latent, action_mean, action_std)

  return planner

import heapq
import random

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from stable_worldmodel.envs.two_room import TwoRoomEnv

import utils.load_fix_pretrained as load_util


# all possible neighbor pairs for a given position
# (y-move, x-move, cost (euclidean distance))

# orthogonal neighbors, cost 1.0
_ORTHOGONAL_NEIGHBORS = [
  (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
]

# diagonal neighbors, cost sqrt(2)
# cost for diagonal move is √2
_DIAGONAL_NEIGHBORS = [
  (-1, -1, 2 ** 0.5), (-1, 1, 2 ** 0.5), (1, -1, 2 ** 0.5), (1, 1, 2 ** 0.5),
]
_NEIGHBORS = _ORTHOGONAL_NEIGHBORS + _DIAGONAL_NEIGHBORS # all possible neighbors 


def build_occupancy_grid(env: TwoRoomEnv):
  """
  returns 2d Numpy array in the format (height, width)
  with a boolean value 
  true  = accessible field 
  false = inaccessible field 
  """
  wall_mask, _door_mask = env._wall_and_door_masks()
  return (~wall_mask).numpy() # return !boolean value


def dijkstra_distance(walkable: np.ndarray, start_xy, goal_xy):
  """
  Shortest-path distance over the walkable grid via Dijkstra.
  """
  H, W = walkable.shape # boolean numpy array from above 
  sx, sy = int(round(start_xy[0])), int(round(start_xy[1])) 
  gx, gy = int(round(goal_xy[0])), int(round(goal_xy[1]))

  if (sx, sy) == (gx, gy): # if start == goal 
    return 0.0

  dist = np.full((H, W), np.inf, dtype=float) # distance to inf for 2d array
  dist[sy, sx] = 0.0 
  pq = [(0.0, sx, sy)] # current position with current distance 

  while pq:
    d, x, y = heapq.heappop(pq)
    if d > dist[y, x]: # if elements distance is larger then shortest known distance to element, then skip
      continue
    if (x, y) == (gx, gy): # if pos = goal return 
      return d
    for dy, dx, cost in _NEIGHBORS:
      nx, ny = x + dx, y + dy

      # pos is in grid height / width and is walkable 
      if 0 <= nx < W and 0 <= ny < H and walkable[ny, nx]:
        nd = d + cost # potential new distance 
        if nd < dist[ny, nx]: # if new distance is smaller then currently known distance 
          dist[ny, nx] = nd
          heapq.heappush(pq, (nd, nx, ny)) # new distance, x, y

  return float("inf")  # no path found


def _wall_geometry(env: TwoRoomEnv):
  """
  Read the current wall/door layout out of a TwoRoomEnv.
  Wall/door config stays the same for each train sample, only agent and target change
  """
  return {
    "axis": env.wall_axis,  # on which axis is the wall (x = 1 , y = 0)
    "center": float(env.WALL_CENTER), # where on the axis lays the wall (x=112)
    "door_positions": env.door_positions[: env.num_doors].tolist(), # y coordinates of the center of the door(s)
    "door_sizes": env.door_sizes[: env.num_doors].tolist(), # margin of the door 
  }


def true_distance_batch(pos_a, pos_g, geom):
  """
  Vectorized, exact shortest-path distance for a batch of (agent, goal)
  """
  pos_a = np.asarray(pos_a, dtype=float).reshape(-1, 2)
  pos_g = np.asarray(pos_g, dtype=float).reshape(-1, 2)

  axis_idx = 0 if geom["axis"] == 1 else 1 # axis of the wall (x)
  other_idx = 1 - axis_idx # y coordinate of the wall center
  center = geom["center"]

  # are agent and target in the same room
  same_room = (pos_a[:, axis_idx] < center) == (pos_g[:, axis_idx] < center)
  # euclidian distance goal and target 
  euclid = np.linalg.norm(pos_a - pos_g, axis=1)

  best_cross = np.full(len(pos_a), np.inf) # for every agent in batch, add inf 

  # loop over every door in the maze
  for door_c, door_s in zip(geom["door_positions"], geom["door_sizes"]):
    lo, hi = door_c - door_s, door_c + door_s # boarders for each door 


    # at which t do we reach the wall?
    # t is the progress (0-1) the agent has made from start to goal 

    # P(t) = a + t * (g - a) => returns x and y coodrinate of the agent 

    # now we re order the formula
    # center = a_x + t * (g_x - a_x)
    # center - a_x = t * (g_x - a_x)
    # t = (center - a_x) / (g_x - a_x)


    denom = pos_g[:, axis_idx] - pos_a[:, axis_idx] # g-a on axis coordinates (x-axis)
    denom = np.where(denom == 0, 1e-9, denom) # safety measure if g = a 
    t = (center - pos_a[:, axis_idx]) / denom # formula from above

    # P(t) = a + t * (g - a) => returns other axis coordinate (y)
    cross_other = pos_a[:, other_idx] + t * (pos_g[:, other_idx] - pos_a[:, other_idx])

    # put other axis (y) in space of the door (closest border of the door)
    gate_other = np.clip(cross_other, lo, hi)

    gate = np.empty_like(pos_a) # new empty array of shape pos a 
    gate[:, axis_idx] = center # x axis on border (112)
    gate[:, other_idx] = gate_other # y coordinate calculated above

    # euclidean distance from start to door + euclidean distance from door to target
    
    d = np.linalg.norm(pos_a - gate, axis=1) + np.linalg.norm(gate - pos_g, axis=1)
    # check if current door is the best door 
    best_cross = np.minimum(best_cross, d)

  return np.where(same_room, euclid, best_cross)


def true_distance(pos_a, pos_g, geom):
  """Single-pair convenience wrapper around true_distance_batch (reference/testing)."""
  return float(true_distance_batch([pos_a], [pos_g], geom)[0])


def _sample_valid_position(env: TwoRoomEnv, walkable: np.ndarray, rng: random.Random):
  """
  Sample a random walkable position within the env's valid bounds.
  Used for random start and target position
  """
  pos_min = float(env.BORDER_SIZE)
  pos_max = float(env.IMG_SIZE - env.BORDER_SIZE - 1)

  while True:
    x = rng.uniform(pos_min, pos_max)
    y = rng.uniform(pos_min, pos_max)
    if walkable[int(round(y)), int(round(x))]:
      return [x, y]


def generate_true_distance_dataset(
  num_samples=1000,
  checkpoint_path=None,
  seed=None,
  device="cpu",
  batch_size=1,
):
  """
  Generates a dataset of (agent_encoding, goal_encoding, heuristic), where
  `heuristic` is the TRUE shortest-path distance through the TwoRoom maze
  """
  print("Initializing Environment and Model...")
  env = TwoRoomEnv()
  env.reset()  # populate default wall/door layout
  walkable = build_occupancy_grid(env)
  geom = _wall_geometry(env)
  rng = random.Random(seed)

  # If the checkpoint_path is not specified it defaults to the HF pretrained
  if checkpoint_path is None:
    checkpoint_path = snapshot_download(
      repo_id="quentinll/lewm-tworooms",
      revision="77adaae0bc31deab21c93740d1f8bb947cd0bdec",
    )

  model = load_util.load_fix_pretrained(checkpoint_path)
  model.eval()
  model = model.to(device) 

  dataset_rows = []

  print(f"Generating {num_samples} samples...")
  with torch.no_grad(): # no gradient (we dont train embeddings)
    num_done = 0
    while num_done < num_samples:
      # current batch size formula
      current_batch = min(batch_size, num_samples - num_done)

      batch_pos_a = []
      batch_pos_g = []
      batch_imgs = []  # agent image, then goal image, per sample

      for _ in range(current_batch):
        pos_a = _sample_valid_position(env, walkable, rng)
        pos_g = _sample_valid_position(env, walkable, rng)
        batch_pos_a.append(pos_a)
        batch_pos_g.append(pos_g)

        env.reset(options={"state": pos_a})
        batch_imgs.append(env.render())

        env.reset(options={"state": pos_g})
        batch_imgs.append(env.render())

      # Stack all 2*(agent + goal) current_batch images into one (B, T=1, C, H, W) batch
      imgs_np = np.stack(batch_imgs)  # merge all images of batch into single array 
      t_imgs = ( # bring array into models required format 
        torch.from_numpy(imgs_np) # to tensor
        .permute(0, 3, 1, 2) # change axis (batch, C (color channels), H, W)
        .float()
        .unsqueeze(1) # Add T
        .to(device)
      ) 

      # create embeddings
      enc = model.encode({"pixels": t_imgs})["emb"] 
      enc = enc.squeeze(1).cpu().numpy()  # Remove T 

      vecs_a = enc[0::2]  # agent embeddings
      vecs_g = enc[1::2]  # goal embeddings

      # Vectorized distance for the whole batch at once (O(1) per pair,
      # no grid search - see true_distance_batch).
      batch_dists = true_distance_batch(batch_pos_a, batch_pos_g, geom)

      for i in range(current_batch):
        dataset_rows.append({
          # embedding
          "agent_pos_enc": vecs_a[i],
          "goal_pos_enc": vecs_g[i],
          # distance
          "heuristic": batch_dists[i],
          # coordinate
          "agent_pos": batch_pos_a[i],
          "goal_pos": batch_pos_g[i],
        })

      num_done += current_batch
      if num_done % 100 == 0 or num_done == num_samples:
        print(f"Progress: {num_done}/{num_samples}")

  return pd.DataFrame(dataset_rows)

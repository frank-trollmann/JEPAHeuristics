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


def bfs_distance(walkable: np.ndarray, start_xy, goal_xy):
  """
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

  return float("inf")  # no path found - shouldn't happen in TwoRoomEnv


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
):
  """
  Generates a dataset of (agent_encoding, goal_encoding, heuristic), where
  `heuristic` is the TRUE shortest-path distance through the TwoRoom maze
  (BFS/Dijkstra over the env's own wall mask)
  """
  print("Initializing Environment and Model...")
  env = TwoRoomEnv()
  env.reset()  # populate default wall/door layout
  walkable = build_occupancy_grid(env)
  rng = random.Random(seed)

  # If the checkpoint_path is not specified it defaults to the HF pretrained
  # tworooms checkpoint, matching the convention
  if checkpoint_path is None:
    checkpoint_path = snapshot_download(
      repo_id="quentinll/lewm-tworooms",
      revision="77adaae0bc31deab21c93740d1f8bb947cd0bdec",
    )

  model = load_util.load_fix_pretrained(checkpoint_path)
  model.eval()

  dataset_rows = []

  print(f"Generating {num_samples} samples...")
  with torch.no_grad(): # no inference (we dont train embeddings)
    for i in range(num_samples):
      pos_a = _sample_valid_position(env, walkable, rng)
      pos_g = _sample_valid_position(env, walkable, rng)

      env.reset(options={"state": pos_a})
      img_a = env.render()

      env.reset(options={"state": pos_g})
      img_g = env.render()

      # Model expects (Batch, Time, Channel, H, W) -> (1, 1, 3, H, W)
      t_img_a = torch.from_numpy(img_a).permute(2, 0, 1).float().unsqueeze(0).unsqueeze(0)
      t_img_g = torch.from_numpy(img_g).permute(2, 0, 1).float().unsqueeze(0).unsqueeze(0)

      enc_a = model.encode({"pixels": t_img_a})["emb"]
      enc_g = model.encode({"pixels": t_img_g})["emb"]

      vec_a = enc_a.squeeze().numpy()
      vec_g = enc_g.squeeze().numpy()

      dist = bfs_distance(walkable, pos_a, pos_g)

      dataset_rows.append({
        # embedding
        "agent_pos_enc": vec_a,
        "goal_pos_enc": vec_g,
        "heuristic": dist,
        # coordinate 
        "agent_pos": pos_a, 
        "goal_pos": pos_g,
      })

      if (i + 1) % 100 == 0:
        print(f"Progress: {i + 1}/{num_samples}")

  return pd.DataFrame(dataset_rows)

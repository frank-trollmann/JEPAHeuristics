import random

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from stable_worldmodel.envs.two_room import TwoRoomEnv

import utils.load_fix_pretrained as load_util
from utils.embedding_probing.experiments.shared.env_helpers import (
  _render_agent_at,
  _sample_valid_position,
  build_occupancy_grid,
)


def _door_geometry(env: TwoRoomEnv):
  """
  Read the current door's layout out of a TwoRoomEnv.
  """
  return {
    "door_y": float(env.door_positions[0]),
    "door_size": float(env.door_sizes[0]),
  }


def generate_random_obstacles_dataset(
  num_samples=1000,
  checkpoint_path=None,
  seed=42,
  device="cpu",
  batch_size=1,
):
  """
  Generates a dataset of (agent_encoding, goal_encoding, door_y, door_size)
  """
  print("Initializing Environment and Model...")
  env = TwoRoomEnv()
  env.reset()  # populate default wall/door layout
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
  with torch.no_grad():  # no gradient (we dont train embeddings)
    num_done = 0
    # per iteration : 1 train example
    while num_done < num_samples:
      current_batch = min(batch_size, num_samples - num_done)

      batch_pos_a = []
      batch_pos_g = []
      batch_door_geom = []
      batch_imgs = []  # agent image, then goal image, per sample

      # for every sample in a batch
      for _ in range(current_batch):
        # random door center
        door_seed = rng.randrange(2 ** 31)
        # reset everything except params in variatio array
        env.reset(options={"variation": ["door.position", "door.size"]}, seed=door_seed)

        # door center + door size
        door_geom = _door_geometry(env)

        walkable = build_occupancy_grid(env) 

        pos_a = _sample_valid_position(env, walkable, rng) # agent
        pos_g = _sample_valid_position(env, walkable, rng) # goal
        batch_pos_a.append(pos_a)
        batch_pos_g.append(pos_g)
        batch_door_geom.append(door_geom)

        # render agent and goal
        batch_imgs.append(_render_agent_at(env, pos_a))
        batch_imgs.append(_render_agent_at(env, pos_g))

      # Stack all 2*(agent + goal) current_batch images into one (B, T=1, C, H, W) batch
      imgs_np = np.stack(batch_imgs)
      t_imgs = (
        torch.from_numpy(imgs_np)
        .permute(0, 3, 1, 2)
        .float()
        .unsqueeze(1)
        .to(device)
      )

      # create embeddings
      enc = model.encode({"pixels": t_imgs})["emb"]
      enc = enc.squeeze(1).cpu().numpy()  # Remove T

      vecs_a = enc[0::2]  # agent embeddings
      vecs_g = enc[1::2]  # goal embeddings

      for i in range(current_batch):
        dataset_rows.append({
          # embedding
          "agent_pos_enc": vecs_a[i],
          "goal_pos_enc": vecs_g[i],
          # obstacle geometry (same for both embeddings of this sample)
          "door_y": batch_door_geom[i]["door_y"],
          "door_size": batch_door_geom[i]["door_size"],
          # coordinate
          "agent_pos": batch_pos_a[i],
          "goal_pos": batch_pos_g[i],
        })

      num_done += current_batch
      if num_done % 100 == 0 or num_done == num_samples:
        print(f"Progress: {num_done}/{num_samples}")

  return pd.DataFrame(dataset_rows)

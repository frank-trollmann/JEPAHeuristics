"""
Position dataset for learning latent heuristics.

Every valid position of the TwoRoom map is rendered and encoded exactly once.
(agent, goal) pairs are built later from these positions, within one split,
so no position can appear in more than one of train / val / test.
"""
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from stable_pretraining.data import dataset_stats

SPLITS = ("train", "val", "test")


def valid_positions(env):
  """
  All integer (x, y) positions inside the map border where the agent (a disc with
  the env's agent radius) touches neither the wall nor the border lines.
  Returns an int array of shape (N, 2).
  """
  # the wall layout is read from the env state that reset() builds
  env.reset()
  wall_mask, _door_mask = env._wall_and_door_masks()
  radius = float(env.variation_space["agent"]["radius"].value.item())

  # distance from every pixel to the nearest wall pixel
  clearance = distance_transform_edt(~wall_mask.numpy())

  # the mask also marks the area outside the border as free, so keep only the inside
  inside = np.zeros(clearance.shape, dtype=bool)
  low, high = env.BORDER_SIZE, env.IMG_SIZE - env.BORDER_SIZE
  inside[low:high, low:high] = True

  ys, xs = np.nonzero(inside & (clearance > radius))
  return np.stack([xs, ys], axis=1)


def split_positions(num_positions, seed, fractions=(0.7, 0.15, 0.15)):
  """
  Randomly assigns each position index to "train", "val" or "test".
  Returns an array of split names with length num_positions.
  """
  assert abs(sum(fractions) - 1.0) < 1e-9, "fractions must sum to 1"
  num_train = int(fractions[0] * num_positions)
  num_val = int(fractions[1] * num_positions)

  names = np.empty(num_positions, dtype="<U5")
  order = np.random.default_rng(seed).permutation(num_positions)
  names[order[:num_train]] = SPLITS[0]
  names[order[num_train:num_train + num_val]] = SPLITS[1]
  names[order[num_train + num_val:]] = SPLITS[2]
  return names


def encode_positions(env, model, positions, device, batch_size=256):
  """
  Renders the agent at every position and encodes the image with the JEPA encoder.
  Images are preprocessed like in the original LeWM evaluation:
  scaled to [0, 1], then ImageNet-normalised.
  Returns a float array of shape (N, latent_dim), in the order of positions.
  """
  mean = torch.tensor(dataset_stats.ImageNet["mean"]).view(1, 3, 1, 1)
  std = torch.tensor(dataset_stats.ImageNet["std"]).view(1, 3, 1, 1)

  encodings = []
  for start in range(0, len(positions), batch_size):
    frames = []
    for x, y in positions[start:start + batch_size]:
      env.reset(options={"state": [int(x), int(y)]})
      frames.append(env.render())  # (H, W, 3) uint8

    images = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
    images = ((images - mean) / std).unsqueeze(1).to(device)  # (B, Time=1, 3, H, W)

    with torch.no_grad():
      embeddings = model.encode({"pixels": images})["emb"]  # (B, 1, latent_dim)
    encodings.append(embeddings[:, 0].cpu().numpy())

  return np.concatenate(encodings)

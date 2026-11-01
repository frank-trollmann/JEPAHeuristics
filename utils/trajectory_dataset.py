"""
Trajectory dataset: encodings of frames from the original LeWM trajectories, and pairs of frames
labelled with the number of environment steps between them. No coordinates and no wall geometry
are used to build the labels, only the order of the frames in an episode.
"""
import h5py
import hdf5plugin  # noqa: F401  (registers the compression filter of the dataset file)
import numpy as np

from utils.heuristic_models import train_pair_model
from utils.position_dataset import encode_frames


def encode_trajectories(h5_path, model, device, episodes, stride=5, seed=0):
  """
  Encodes every stride-th frame of a random subset of episodes.
  Returns a dict: encodings (N, D), episode (N,) the episode number, step (N,) the environment step of
  the frame, positions (N, 2) the agent coordinates (kept only for analysis, never used for labels).
  """
  rng = np.random.default_rng(seed)
  with h5py.File(h5_path) as f:
    offsets, lengths = f["ep_offset"][:], f["ep_len"][:]
    chosen = np.sort(rng.choice(len(lengths), episodes, replace=False))
    parts = {"encodings": [], "episode": [], "step": [], "positions": []}
    for episode in chosen:
      start, length = int(offsets[episode]), int(lengths[episode])
      frames = f["pixels"][start:start + length:stride]
      parts["encodings"].append(encode_frames(model, frames, device))
      parts["episode"].append(np.full(len(frames), episode))
      parts["step"].append(np.arange(0, length, stride)[:len(frames)])
      parts["positions"].append(f["pos_agent"][start:start + length:stride])
  return {name: np.concatenate(values) for name, values in parts.items()}


def split_episodes(episode, seed, fractions=(0.8, 0.1, 0.1)):
  """Assigns whole episodes to train, val or test. Returns an array of split names, one per frame."""
  unique = np.unique(episode)
  order = np.random.default_rng(seed).permutation(len(unique))
  cut_train, cut_val = int(fractions[0] * len(unique)), int((fractions[0] + fractions[1]) * len(unique))
  names = np.empty(len(unique), dtype="<U5")
  names[order[:cut_train]] = "train"
  names[order[cut_train:cut_val]] = "val"
  names[order[cut_val:]] = "test"
  return names[np.searchsorted(unique, episode)]


def temporal_pairs(episode, step, mask, both_directions=True):
  """
  All pairs of frames from the same episode (restricted to frames where mask is True), as index arrays
  (agent, goal) into the frame arrays, and the number of steps from the earlier to the later frame.
  With both_directions the reversed pairs are added with the same label: the agent can walk back.
  """
  agents, goals, labels = [], [], []
  indices = np.flatnonzero(mask)
  starts = np.flatnonzero(np.diff(episode[indices], prepend=-1))  # first frame of each episode among the masked ones
  for first, last in zip(starts, list(starts[1:]) + [len(indices)]):
    frames = indices[first:last]
    earlier, later = np.triu_indices(len(frames), k=1)
    agents.append(frames[earlier])
    goals.append(frames[later])
    labels.append(step[frames[later]] - step[frames[earlier]])
  agent, goal, label = np.concatenate(agents), np.concatenate(goals), np.concatenate(labels).astype(float)
  if both_directions:
    agent, goal, label = np.concatenate([agent, goal]), np.concatenate([goal, agent]), np.concatenate([label, label])
  return agent, goal, label


def train_temporal_heuristic(data, device, seed, max_pairs=3_000_000):
  """
  Trains h'(z_agent, z_goal) = steps between two frames of the same trajectory, on training episodes only
  (validation episodes pick the best epoch). Returns the PairPredictor and the split name of every frame.
  """
  split = split_episodes(data["episode"], seed)
  train = temporal_pairs(data["episode"], data["step"], split == "train")
  val = temporal_pairs(data["episode"], data["step"], split == "val")
  if len(train[0]) > max_pairs:
    keep = np.random.default_rng(seed).choice(len(train[0]), max_pairs, replace=False)
    train = tuple(part[keep] for part in train)
  return train_pair_model(data["encodings"], train, val, device, seed, kind="mlp"), split

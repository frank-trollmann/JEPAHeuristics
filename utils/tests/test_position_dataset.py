import numpy as np
import pytest
import torch
from stable_worldmodel.envs.two_room import TwoRoomEnv

from utils.position_dataset import SPLITS, encode_positions, split_positions, valid_positions


@pytest.fixture(scope="module")
def env():
  return TwoRoomEnv()


def test_valid_positions_keep_the_agent_disc_clear_of_walls(env):
  positions = valid_positions(env)
  wall_mask, _ = env._wall_and_door_masks()
  wall_ys, wall_xs = np.nonzero(wall_mask.numpy())
  radius = float(env.variation_space["agent"]["radius"].value.item())

  # brute-force check on a sample, independent of the distance transform used in the code
  sample = positions[np.random.default_rng(0).choice(len(positions), 200, replace=False)]
  for x, y in sample:
    assert np.hypot(wall_xs - x, wall_ys - y).min() > radius


def test_valid_positions_exclude_wall_and_border_but_keep_the_door(env):
  as_set = {tuple(p) for p in valid_positions(env).tolist()}
  door_center_y = int(env.door_positions[0])

  assert (env.WALL_CENTER, 100) not in as_set  # inside the wall
  assert (env.BORDER_SIZE, env.BORDER_SIZE) not in as_set  # touching the border
  assert (env.WALL_CENTER, door_center_y) in as_set  # the door must stay walkable


def test_split_is_disjoint_complete_and_roughly_70_15_15():
  names = split_positions(1000, seed=1)

  assert set(names) == set(SPLITS)
  assert [int((names == s).sum()) for s in SPLITS] == [700, 150, 150]


def test_split_is_reproducible_per_seed():
  assert (split_positions(500, seed=3) == split_positions(500, seed=3)).all()
  assert (split_positions(500, seed=3) != split_positions(500, seed=4)).any()


class StubModel:
  """Stands in for JEPA: embeds an image as its mean value, and records the largest input seen."""

  def __init__(self):
    self.largest_input = 0.0

  def encode(self, info):
    pixels = info["pixels"]  # (B, 1, 3, H, W)
    self.largest_input = max(self.largest_input, pixels.max().item())
    return {"emb": pixels.mean(dim=(2, 3, 4)).unsqueeze(-1)}  # (B, 1, 1)


def test_encoding_keeps_order_across_batches_and_uses_normalised_input(env):
  positions = valid_positions(env)[:7]
  model = StubModel()

  in_one_batch = encode_positions(env, model, positions, device="cpu", batch_size=7)
  in_small_batches = encode_positions(env, model, positions, device="cpu", batch_size=3)

  assert in_one_batch.shape == (7, 1)
  assert np.allclose(in_one_batch, in_small_batches, atol=1e-6)
  assert len(np.unique(in_one_batch)) > 1  # different positions give different images
  assert model.largest_input < 3.0  # raw 0-255 pixels would be far above this

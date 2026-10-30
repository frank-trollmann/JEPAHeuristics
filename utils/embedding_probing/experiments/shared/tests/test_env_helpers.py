import random

import numpy as np
import torch
from stable_worldmodel.envs.two_room import TwoRoomEnv

from utils.embedding_probing.experiments.shared.env_helpers import (
  _render_agent_at,
  _sample_valid_position,
  build_occupancy_grid,
)


def test_sample_valid_position_is_always_walkable():
  env = TwoRoomEnv()
  env.reset()
  walkable = build_occupancy_grid(env)
  rng = random.Random(0)

  for _ in range(30):
    x, y = _sample_valid_position(env, walkable, rng)
    assert walkable[int(round(y)), int(round(x))]


def test_render_agent_at_matches_reset_and_render():
  # uses its own env instance since this mutates env.agent_position
  # directly, bypassing reset()
  env = TwoRoomEnv()
  env.reset()
  pos = [60.0, 112.0]

  env.reset(options={"state": pos})
  expected = env.render()

  actual = _render_agent_at(env, pos)

  np.testing.assert_array_equal(actual, expected)


def test_render_agent_at_differs_for_different_positions():
  env = TwoRoomEnv()
  env.reset()

  img_a = _render_agent_at(env, [60.0, 60.0])
  img_b = _render_agent_at(env, [160.0, 160.0])

  assert not np.array_equal(img_a, img_b)


def test_render_agent_at_does_not_reset_door_layout():
  # the correctness-critical property for callers that resample door
  # geometry per sample: rendering must not touch the variation space
  env = TwoRoomEnv()
  env.reset()
  env.reset(options={"variation": ["door.position", "door.size"]}, seed=1)

  door_y_before = float(env.door_positions[0])
  door_size_before = float(env.door_sizes[0])

  _render_agent_at(env, [60.0, 60.0])
  _render_agent_at(env, [160.0, 160.0])

  assert float(env.door_positions[0]) == door_y_before
  assert float(env.door_sizes[0]) == door_size_before


def test_wall_and_door_mask_caching_matches_fresh_computation():
  env = TwoRoomEnv()
  env.reset()

  fresh_wall_mask, fresh_door_mask = env._wall_and_door_masks()

  cached_wall_mask, cached_door_mask = env._wall_and_door_masks()
  env._wall_and_door_masks = lambda: (cached_wall_mask, cached_door_mask)

  patched_wall_mask, patched_door_mask = env._wall_and_door_masks()

  assert torch.equal(patched_wall_mask, fresh_wall_mask)
  assert torch.equal(patched_door_mask, fresh_door_mask)

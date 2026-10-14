import numpy as np
import pytest
import torch
from stable_worldmodel.envs.two_room import TwoRoomEnv

from utils.dynamics_probe import (
  BLOCK,
  BLOCKS,
  CONDITIONS,
  FULL_SPEED_BLOCK,
  MAP_MAX,
  MAP_MIN,
  WALL_STOP_LEFT,
  WALL_STOP_RIGHT,
  make_trials,
  normalised_block,
  run_trial,
)


class StubModel:
  """Stands in for JEPA: encodes an image as its mean value; P and the action encoder are identities."""

  def encode(self, info):
    return {"emb": info["pixels"].mean(dim=(2, 3, 4)).unsqueeze(-1)}

  def action_encoder(self, actions):
    return actions

  def predict(self, emb, act_emb):
    return emb


@pytest.fixture(scope="module")
def env():
  env = TwoRoomEnv()
  env.reset()
  return env


def test_normalised_block_repeats_the_zscored_action():
  block = normalised_block([1.0, 0.0], mean=np.array([0.0, 0.0]), std=np.array([0.5, 2.0]))

  assert block.shape == (2 * BLOCK,)
  assert np.allclose(block, [2.0, 0.0] * BLOCK)


@pytest.mark.parametrize("condition", ["wall", "wall_diagonal"])
def test_wall_trials_end_their_last_block_at_the_wall(env, condition):
  door_center = float(env.door_positions[0])
  trials = make_trials(condition, 20, door_center, np.random.default_rng(0))

  for (x, y), (ax, ay) in trials:
    last_start = x + ax * FULL_SPEED_BLOCK * (BLOCKS - 1)
    stop = WALL_STOP_LEFT if ax > 0 else WALL_STOP_RIGHT
    distance_to_stop = (stop - last_start) * ax
    # the last block starts 2 to 22 px before the stop position, so a full-speed block (25 px) must be blocked
    assert 2 <= distance_to_stop <= 22
    y_at_last_block = y + ay * FULL_SPEED_BLOCK * (BLOCKS - 1)
    assert abs(y_at_last_block - door_center) > 30  # far from the door


@pytest.mark.parametrize("condition", CONDITIONS)
def test_every_trial_path_stays_inside_the_map(env, condition):
  door_center = float(env.door_positions[0])

  for (x, y), (ax, ay) in make_trials(condition, 40, door_center, np.random.default_rng(0)):
    end_x = x + ax * FULL_SPEED_BLOCK * BLOCKS  # where the agent would end without a wall
    end_y = y + ay * FULL_SPEED_BLOCK * BLOCKS
    for coordinate in (x, y, end_x, end_y):
      assert MAP_MIN - 1e-9 <= coordinate <= MAP_MAX + 1e-9


def test_real_environment_blocks_at_the_wall_but_not_at_the_door_or_in_open_space(env):
  door_center = float(env.door_positions[0])
  mean, std = np.zeros(2), np.ones(2)
  moved = {}
  for condition in CONDITIONS:
    last_moves = []
    for start, action in make_trials(condition, 10, door_center, np.random.default_rng(1)):
      positions, _, _ = run_trial(env, StubModel(), "cpu", start, action, mean, std)
      last_moves.append(action[0] * (positions[-1, 0] - positions[-2, 0]))
    moved[condition] = np.mean(last_moves)

  assert moved["open"] == pytest.approx(FULL_SPEED_BLOCK)
  assert moved["open_diagonal"] == pytest.approx(FULL_SPEED_BLOCK)
  assert moved["door"] == pytest.approx(FULL_SPEED_BLOCK)
  assert moved["wall"] < 0.8 * FULL_SPEED_BLOCK
  assert moved["wall_diagonal"] < 0.8 * FULL_SPEED_BLOCK


def test_run_trial_returns_consistent_shapes(env):
  positions, latents, predicted = run_trial(
    env, StubModel(), "cpu", (50.0, 150.0), (1.0, 0.0), np.zeros(2), np.ones(2))

  assert positions.shape == (BLOCKS + 1, 2)
  assert latents.shape == (BLOCKS + 1, 1)
  assert predicted.shape == (1,)

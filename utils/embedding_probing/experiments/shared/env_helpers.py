import random

import numpy as np
import torch
from stable_worldmodel.envs.two_room import TwoRoomEnv


def build_occupancy_grid(env: TwoRoomEnv):
  """
  returns 2d Numpy array in the format (height, width)
  with a boolean value
  true  = accessible field
  false = inaccessible field
  """
  wall_mask, _door_mask = env._wall_and_door_masks()
  return (~wall_mask).numpy()


def _sample_valid_position(env: TwoRoomEnv, walkable: np.ndarray, rng: random.Random):
  """
  Sample a random walkable position within the env's valid bounds.
  Used for random start and target position.
  """
  # are in which agent can be placed
  pos_min = float(env.BORDER_SIZE)
  pos_max = float(env.IMG_SIZE - env.BORDER_SIZE - 1)

  # loop until walkable position got created
  while True:
    x = rng.uniform(pos_min, pos_max)
    y = rng.uniform(pos_min, pos_max)
    if walkable[int(round(y)), int(round(x))]:
      return [x, y]


def _render_agent_at(env: TwoRoomEnv, pos):
  """
  renders agent position without resetting the environment
  """
  # transform python list [x,y] to tensor, which is required by env lib
  env.agent_position = torch.as_tensor(pos, dtype=torch.float32)
  # call internal render_frame method directly without using .render(), this helps us to avoid env.reset
  img_chw = env._render_frame(agent_pos=env.agent_position).cpu().numpy() # must be hardcoded to cpu 
  return img_chw.transpose(1, 2, 0)

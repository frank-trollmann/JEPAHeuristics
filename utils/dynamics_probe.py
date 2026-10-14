"""
Dynamics probe: does the JEPA predictor P know about the wall?

The agent runs horizontally for three blocks in the real environment. P predicts the
latent after the third block from the first three real latents and actions. We decode
the position from that predicted latent and compare how far the agent was predicted to
move with how far it really moved, and with how far it would have moved without a wall.
"""
import numpy as np
import torch

from utils.position_dataset import encode_frames

BLOCK = 5  # environment steps per model step (action_block in the LeWM config)
BLOCKS = 3  # blocks run in the real environment; P predicts the last one from the earlier frames
AGENT_RADIUS = 7
FULL_SPEED_BLOCK = 25  # px moved in one block at action 1: speed 5 px/step * BLOCK steps

# (left-room x-range, right-room x-range) of the agent centre when it is stopped by the wall:
# the wall spans x = 107..117, so the agent centre stops at 107 - radius from the left, 117 + radius from the right.
WALL_STOP_LEFT = 107 - AGENT_RADIUS
WALL_STOP_RIGHT = 117 + AGENT_RADIUS

CONDITIONS = ("wall", "door", "open", "wall_diagonal", "open_diagonal")
MAP_MIN, MAP_MAX = 21, 203  # the agent centre stays inside the border, which is 14 px wide, plus the radius


def action_statistics(actions):
  """Mean and standard deviation of the actions (N, 2), ignoring rows with NaN."""
  actions = actions[~np.isnan(actions).any(axis=1)]
  return actions.mean(axis=0), actions.std(axis=0)


def normalised_block(action, mean, std):
  """The model input for one block: the z-scored action repeated BLOCK times, shape (2 * BLOCK,)."""
  return np.tile((np.asarray(action) - mean) / std, BLOCK)


def make_trials(condition, num_trials, door_center, rng):
  """
  Start position and action for each trial, as a list of ((x, y), (ax, ay)).
  Every trial runs BLOCKS blocks of full speed and stays inside the map.

  wall, door, wall_diagonal: move toward the wall, so that the last block starts 2 to 22 px before it.
  'wall' and 'door' move horizontally (door at the door height); 'wall_diagonal' also moves down at full speed.
  open, open_diagonal: move away from the wall inside one room, so nothing can block the agent.
  The diagonal conditions use actions (+-1, +1), which is the most common kind of action in the LeWM data.
  """
  trials = []
  for i in range(num_trials):
    left_room = i % 2 == 0
    if condition in ("wall", "door", "wall_diagonal"):
      direction = 1.0 if left_room else -1.0
      stop = WALL_STOP_LEFT if left_room else WALL_STOP_RIGHT
      last_start = stop - direction * rng.uniform(2, 22)
      start_x = last_start - direction * FULL_SPEED_BLOCK * (BLOCKS - 1)
      if condition == "door":
        y, vertical = door_center + rng.uniform(-5, 5), 0.0
      elif condition == "wall":
        y, vertical = rng.uniform(100, 190), 0.0
      else:
        y, vertical = rng.uniform(100, 150) - FULL_SPEED_BLOCK * (BLOCKS - 1), 1.0
    else:
      direction = -1.0 if left_room else 1.0
      start_x = WALL_STOP_LEFT - rng.uniform(0, 4) if left_room else WALL_STOP_RIGHT + rng.uniform(0, 4)
      if condition == "open":
        y, vertical = rng.uniform(100, 190), 0.0
      else:
        y, vertical = rng.uniform(30, 120), 1.0
    trials.append(((start_x, y), (direction, vertical)))
  return trials


def run_trial(env, model, device, start, action, mean, std):
  """
  Runs one trial. Returns the real positions after each block (BLOCKS + 1, 2),
  the real latents (BLOCKS + 1, D) and the latent P predicts for the last frame (D,).
  """
  env.reset(options={"state": list(start)})
  frames, positions = [env.render()], [env.agent_position.numpy().copy()]
  for _ in range(BLOCKS):
    for _ in range(BLOCK):
      env.step(np.array(action, dtype=np.float32))
    frames.append(env.render())
    positions.append(env.agent_position.numpy().copy())
  latents = encode_frames(model, np.stack(frames), device)

  block = torch.tensor(normalised_block(action, mean, std), dtype=torch.float32, device=device)
  actions = block.repeat(1, BLOCKS, 1)  # (1, BLOCKS, 2 * BLOCK): the action taken at each of the first BLOCKS frames
  history = torch.tensor(latents[:BLOCKS], dtype=torch.float32, device=device).unsqueeze(0)
  with torch.no_grad():
    predicted = model.predict(history, model.action_encoder(actions))[0, -1].cpu().numpy()
  return np.stack(positions), latents, predicted


def summarise(condition, results, decoder):
  """
  Moves along x in the last block (px, signed in the direction of motion):
  real, without a wall, as predicted by P, and as decoded from the real latent (decoder check).
  """
  real, free, predicted, decoded_real = [], [], [], []
  for positions, latents, predicted_latent, direction in results:
    before = decoder.predict(latents[BLOCKS - 1:BLOCKS])[0, 0]
    real.append(direction * (positions[-1, 0] - positions[-2, 0]))
    free.append(FULL_SPEED_BLOCK)
    predicted.append(direction * (decoder.predict(predicted_latent[None])[0, 0] - before))
    decoded_real.append(direction * (decoder.predict(latents[-1:])[0, 0] - before))
  real, free, predicted = np.array(real), np.array(free), np.array(predicted)
  varies = real.std() > 0  # only the wall trials have different real moves, so only there a correlation exists
  return {
    "condition": condition,
    "real_move": real.mean(),
    "move_without_wall": free.mean(),
    "predicted_move": predicted.mean(),
    "predicted_move_std": predicted.std(),
    "decoded_real_move": np.mean(decoded_real),
    "abs_error_vs_real": np.abs(predicted - real).mean(),
    "abs_error_vs_no_wall": np.abs(predicted - free).mean(),
    "correlation_predicted_real": np.corrcoef(predicted, real)[0, 1] if varies else np.nan,
  }

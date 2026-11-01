"""
CEM planning with the JEPA model, following the LeWM evaluation
(le-wm/config/eval/tworoom.yaml and solver/cem.yaml; stable_worldmodel/solver/cem.py).

A plan is HORIZON blocks. One block is BLOCK environment actions, stored as one vector of
2 * BLOCK numbers in the z-scored action space the model was trained in.
"""
import numpy as np
import torch

from utils.dynamics_probe import BLOCK
from utils.position_dataset import encode_frames

HORIZON = 5
NUM_SAMPLES = 300
ITERATIONS = 30
TOP_K = 30
VAR_SCALE = 1.0
ACTION_DIM = 2 * BLOCK


def rollout(model, latent, actions):
  """
  The latent that the predictor gives after each plan in actions (S, HORIZON, ACTION_DIM), starting
  from one latent (D,). Only the current frame is used as history. Returns (S, D).
  """
  history = model.predictor.num_frames
  samples = actions.shape[0]
  emb = latent.reshape(1, 1, -1).expand(samples, 1, -1)
  act = actions[:, :1]
  for t in range(actions.shape[1]):
    predicted = model.predict(emb[:, -history:], model.action_encoder(act)[:, -history:])[:, -1:]
    emb = torch.cat([emb, predicted], dim=1)
    if t + 1 < actions.shape[1]:
      act = torch.cat([act, actions[:, t + 1:t + 2]], dim=1)
  return emb[:, -1]


def jepa_cost(final_latents, goal_latent):
  """The cost JEPA planning uses (JEPA.criterion): squared distance of the final predicted latent to the goal latent."""
  return ((final_latents - goal_latent) ** 2).sum(dim=1)


def plan(model, latent, cost, generator, device, horizon=HORIZON):
  """
  Cross-entropy method. cost(final_latents) gives one cost per candidate (lower is better).
  Returns the final mean of the action distribution, shape (horizon, ACTION_DIM).
  """
  mean = torch.zeros(horizon, ACTION_DIM, device=device)
  var = VAR_SCALE * torch.ones(horizon, ACTION_DIM, device=device)
  with torch.no_grad():
    for _ in range(ITERATIONS):
      candidates = torch.randn(NUM_SAMPLES, horizon, ACTION_DIM, generator=generator, device=device) * var + mean
      candidates[0] = mean  # like the original solver, the current mean is always a candidate
      costs = cost(rollout(model, latent, candidates))
      elites = candidates[torch.topk(costs, TOP_K, largest=False).indices]
      mean, var = elites.mean(dim=0), elites.std(dim=0)
  return mean


def to_environment_actions(plan_blocks, action_mean, action_std):
  """Turns a plan (blocks, ACTION_DIM) in z-scored space into environment actions (blocks * BLOCK, 2) in [-1, 1]."""
  actions = plan_blocks.cpu().numpy().reshape(-1, 2) * action_std + action_mean
  return np.clip(actions, -1.0, 1.0).astype(np.float32)


def run_episode(env, model, device, start, goal, make_cost, action_mean, action_std, budget_steps, seed):
  """
  Plans with CEM and acts in the real environment until the goal is reached (the environment's own
  goal test, within 16 px) or budget_steps environment steps are used. All HORIZON planned blocks are
  executed before replanning, like receding_horizon in the LeWM evaluation.

  make_cost(goal_latent, goal_xy) returns the cost function used by CEM for this goal.
  Returns a dict: success, steps used, final distance to the goal and the smallest distance reached.
  """
  generator = torch.Generator(device=device).manual_seed(seed)

  env.reset(options={"state": list(goal)})
  goal_latent = torch.tensor(encode_frames(model, np.stack([env.render()]), device)[0], device=device)
  cost = make_cost(goal_latent, goal)

  env.reset(options={"state": list(start), "target_state": list(goal)})
  steps, closest, success = 0, float(np.linalg.norm(np.asarray(start) - np.asarray(goal))), False
  while steps < budget_steps and not success:
    latent = torch.tensor(encode_frames(model, np.stack([env.render()]), device)[0], device=device)
    blocks = plan(model, latent, cost, generator, device)
    for action in to_environment_actions(blocks, action_mean, action_std):
      _, _, success, _, info = env.step(action)
      steps += 1
      closest = min(closest, info["distance_to_target"])
      if success or steps >= budget_steps:
        break

  return {
    "success": bool(success),
    "steps": steps,
    "final_distance": float(info["distance_to_target"]) if steps else closest,
    "closest_distance": closest,
  }

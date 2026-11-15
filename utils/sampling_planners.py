"""
Two more sampling planners next to CEM (cem_planner.plan), with the same interface, settings and
rollout, following stable_worldmodel/solver/mppi.py and icem.py. They differ from CEM only in how the
action distribution is updated, so any difference in success is down to the planner, not the cost.

mppi: the new mean is a softmax-weighted average of the elite plans; the variance stays fixed.
icem: colored (low-frequency) noise, five elites kept from the previous iteration, momentum on mean and std,
      candidates clipped to the action bounds.
"""
import numpy as np
import torch

from utils.cem_planner import ACTION_DIM, HORIZON, ITERATIONS, NUM_SAMPLES, TOP_K, VAR_SCALE, rollout
from utils.dynamics_probe import normalised_block
from utils.latent_search import DIRECTIONS

TEMPERATURE = 0.5  # MPPI, in units of the spread of the elite costs (see plan_mppi)
NOISE_BETA = 2.0  # iCEM colored noise exponent; 0 is white noise like CEM
MOMENTUM = 0.1  # iCEM: weight of the old mean and std
KEEP_ELITES = 5  # iCEM: elites carried over to the next iteration


def plan_mppi(model, latent, cost, generator, device, horizon=HORIZON, num_samples=NUM_SAMPLES, temperature=TEMPERATURE):
  """
  MPPI. The library weights elites by softmax(-(cost - min cost) / 0.5), which only makes sense on the
  scale of the JEPA cost. Here the temperature is in units of the elite cost spread (their standard
  deviation), so multiplying the cost by any constant gives the same plan and all costs behave alike.
  """
  mean = torch.zeros(horizon, ACTION_DIM, device=device)
  var = VAR_SCALE * torch.ones(horizon, ACTION_DIM, device=device)
  with torch.no_grad():
    for _ in range(ITERATIONS):
      candidates = torch.randn(num_samples, horizon, ACTION_DIM, generator=generator, device=device) * var + mean
      candidates[0] = mean
      values, index = torch.topk(cost(rollout(model, latent, candidates)), TOP_K, largest=False)
      spread = values.std().clamp(min=1e-8)
      weights = torch.softmax(-(values - values.min()) / (temperature * spread), dim=0)
      mean = (weights[:, None, None] * candidates[index]).sum(dim=0)
  return mean


def colored_noise(num_samples, horizon, generator, device, beta=NOISE_BETA):
  """Noise (num_samples, horizon, ACTION_DIM) with power 1/f^beta over the horizon, scaled to unit std."""
  white = torch.randn(num_samples, ACTION_DIM, horizon, generator=generator, device=device)
  scale = torch.fft.rfftfreq(horizon, device=device)
  scale[0] = 1.0
  scale = scale.pow(-beta / 2)
  scale[0] = scale[1] if len(scale) > 1 else 1.0
  colored = torch.fft.irfft(torch.fft.rfft(white, dim=-1) * scale, n=horizon, dim=-1)
  return (colored / colored.std(dim=-1, keepdim=True).clamp(min=1e-8)).transpose(-1, -2)


def plan_icem(model, latent, cost, generator, device, horizon=HORIZON, num_samples=NUM_SAMPLES, bounds=None):
  """
  iCEM. bounds = (low, high), each (ACTION_DIM,), are the environment's action limits in the z-scored
  action space (no clipping if None).
  """
  mean = torch.zeros(horizon, ACTION_DIM, device=device)
  var = VAR_SCALE * torch.ones(horizon, ACTION_DIM, device=device)
  kept = None
  with torch.no_grad():
    for _ in range(ITERATIONS):
      candidates = colored_noise(num_samples, horizon, generator, device) * var + mean
      candidates[0] = mean
      if kept is not None:
        candidates[1:1 + KEEP_ELITES] = kept[:KEEP_ELITES]
      if bounds is not None:
        candidates = torch.minimum(torch.maximum(candidates, bounds[0]), bounds[1])
      index = torch.topk(cost(rollout(model, latent, candidates)), TOP_K, largest=False).indices
      kept = candidates[index]
      mean = MOMENTUM * mean + (1 - MOMENTUM) * kept.mean(dim=0)
      var = MOMENTUM * var + (1 - MOMENTUM) * kept.std(dim=0)
  return mean


def make_snapped_cem(action_mean, action_std):
  """
  CEM whose candidates are snapped, block by block, to the nearest of the 8 full-speed directions that the
  search planners use (utils/latent_search.DIRECTIONS). Same action set as the search, so a difference to
  the search is not due to the actions it may take.
  """
  moves = torch.tensor(np.stack([normalised_block(d, action_mean, action_std) for d in DIRECTIONS]), dtype=torch.float32)

  def snap(candidates):
    table = moves.to(candidates.device)
    flat = candidates.reshape(-1, ACTION_DIM)
    return table[torch.cdist(flat, table).argmin(dim=1)].reshape(candidates.shape)

  def planner(model, latent, cost, generator, device, horizon=HORIZON, num_samples=NUM_SAMPLES):
    mean = torch.zeros(horizon, ACTION_DIM, device=device)
    var = VAR_SCALE * torch.ones(horizon, ACTION_DIM, device=device)
    with torch.no_grad():
      for _ in range(ITERATIONS):
        candidates = torch.randn(num_samples, horizon, ACTION_DIM, generator=generator, device=device) * var + mean
        candidates[0] = mean
        candidates = snap(candidates)
        elites = candidates[torch.topk(cost(rollout(model, latent, candidates)), TOP_K, largest=False).indices]
        mean, var = elites.mean(dim=0), elites.std(dim=0)
    return snap(mean)

  return planner


def plan_random(model, latent, cost, generator, device, horizon=HORIZON, num_samples=NUM_SAMPLES):
  """
  Random shooting: the same number of plans as CEM draws in total (num_samples * ITERATIONS), all from the
  initial distribution, no refitting; the best one wins. The floor that shows what refitting adds.
  """
  best, best_cost = None, float("inf")
  with torch.no_grad():
    for _ in range(ITERATIONS):
      candidates = torch.randn(num_samples, horizon, ACTION_DIM, generator=generator, device=device) * VAR_SCALE
      costs = cost(rollout(model, latent, candidates))
      i = costs.argmin()
      if costs[i] < best_cost:
        best, best_cost = candidates[i].clone(), costs[i].item()
  return best


def plan_gradient(model, latent, cost, generator, device, horizon=HORIZON, num_samples=NUM_SAMPLES, starts=30, steps=ITERATIONS, lr=0.3):
  """
  Gradient descent on the action plan through the predictor (the cost must be differentiable): Adam on
  `starts` random initial plans, `steps` steps, the plan with the lowest final cost wins. num_samples is
  not used (starts is the batch).
  """
  plans = (torch.randn(starts, horizon, ACTION_DIM, generator=generator, device=device) * VAR_SCALE).requires_grad_()
  optimiser = torch.optim.Adam([plans], lr=lr)
  with torch.enable_grad():
    for _ in range(steps):
      loss = cost(rollout(model, latent, plans)).sum()  # starts are independent, so the sum gives each its own gradient
      plans.grad = torch.autograd.grad(loss, plans)[0]
      optimiser.step()
  with torch.no_grad():
    return plans[cost(rollout(model, latent, plans)).argmin()].detach()

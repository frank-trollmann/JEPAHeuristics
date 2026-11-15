import torch

from utils.cem_planner import ACTION_DIM, HORIZON, jepa_cost, rollout
import numpy as np

from utils.latent_search import DIRECTIONS
from utils.dynamics_probe import normalised_block
from utils.sampling_planners import colored_noise, make_snapped_cem, plan_icem, plan_mppi
from utils.tests.test_cem_planner import AdditiveModel


def _reaches(planner, start, goal, seed, **kwargs):
  model = AdditiveModel()
  best = planner(model, torch.tensor([start]), lambda final: jepa_cost(final, torch.tensor([goal])),
                 torch.Generator().manual_seed(seed), device="cpu", **kwargs)
  return rollout(model, torch.tensor([start]), best[None]).item()


def test_mppi_and_icem_find_a_plan_that_reaches_the_goal():
  assert abs(_reaches(plan_mppi, 0.0, 3.0, seed=0) - 3.0) < 0.3
  assert abs(_reaches(plan_icem, 0.0, 3.0, seed=0) - 3.0) < 0.3


def test_mppi_plan_does_not_depend_on_the_scale_of_the_cost():
  model = AdditiveModel()
  goal = torch.tensor([3.0])
  plans = [
    plan_mppi(model, torch.tensor([0.0]), lambda final, s=scale: s * jepa_cost(final, goal),
              torch.Generator().manual_seed(2), device="cpu")
    for scale in (1.0, 1000.0)
  ]

  assert torch.allclose(plans[0], plans[1], atol=1e-3)


def test_icem_keeps_its_plan_inside_the_action_bounds():
  model = AdditiveModel()
  bounds = (torch.full((ACTION_DIM,), -1.0), torch.full((ACTION_DIM,), 1.0))

  best = plan_icem(model, torch.tensor([0.0]), lambda final: -final[:, 0],  # wants a large final latent
                   torch.Generator().manual_seed(3), device="cpu", bounds=bounds)

  assert best.max() <= 1.0 + 1e-6
  assert best[:, 0].mean() > 0.9  # pushes against the bound


def test_colored_noise_is_smoother_over_the_horizon_than_white_noise():
  def lag_one_correlation(noise):
    return torch.corrcoef(torch.stack([noise[:, :-1].flatten(), noise[:, 1:].flatten()]))[0, 1].item()

  generator = torch.Generator().manual_seed(4)
  colored = colored_noise(4000, 8, generator, "cpu", beta=2.0)
  white = colored_noise(4000, 8, generator, "cpu", beta=0.0)

  assert colored.shape == (4000, 8, ACTION_DIM)
  assert lag_one_correlation(colored) > 0.5
  assert abs(lag_one_correlation(white)) < 0.1


def test_snapped_cem_only_returns_the_eight_directions_and_reaches_the_goal():
  mean, std = np.zeros(2), np.ones(2)
  planner = make_snapped_cem(mean, std)
  model = AdditiveModel()
  best = planner(model, torch.tensor([0.0]), lambda final: jepa_cost(final, torch.tensor([3.0])),
                 torch.Generator().manual_seed(0), device="cpu")
  table = torch.tensor(np.stack([normalised_block(d, mean, std) for d in DIRECTIONS]), dtype=torch.float32)

  assert all((table - block).abs().sum(dim=1).min() < 1e-5 for block in best)

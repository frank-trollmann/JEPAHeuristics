import numpy as np
import torch

from utils.cem_planner import ACTION_DIM, jepa_cost, rollout
from utils.latent_search import make_beam_planner, make_mcts_planner
from utils.sampling_planners import plan_gradient, plan_random
from utils.tests.test_cem_planner import AdditiveModel
from utils.tests.test_latent_search import CountingModel, px_cost

MEAN, STD = np.zeros(2), np.ones(2)


def _final(model, start, blocks):
  return rollout(model, torch.tensor([start]), blocks[None]).item()


def test_random_shooting_gets_close_to_the_goal():
  model = AdditiveModel()
  best = plan_random(model, torch.tensor([0.0]), lambda final: jepa_cost(final, torch.tensor([3.0])),
                     torch.Generator().manual_seed(0), device="cpu")

  assert abs(_final(model, 0.0, best) - 3.0) < 0.5


def test_gradient_descent_follows_the_gradient_to_the_goal():
  model = AdditiveModel()
  best = plan_gradient(model, torch.tensor([0.0]), lambda final: jepa_cost(final, torch.tensor([8.0])),
                       torch.Generator().manual_seed(0), device="cpu")

  assert best.shape[-1] == ACTION_DIM
  assert abs(_final(model, 0.0, best) - 8.0) < 0.5  # random starts reach about +-2 at most: only the gradient gets there


def test_beam_search_walks_to_the_goal_one_level_per_predictor_call():
  model = CountingModel()
  blocks = make_beam_planner(MEAN, STD, width=1, merge_distance=0.1)(model, torch.tensor([0.0]), px_cost(3.0))

  assert blocks.shape[0] == 3 and blocks[:, 0].sum().item() == 3.0
  assert model.calls == 3  # one batched call per level


def test_beam_search_heads_toward_an_unreachable_goal_and_respects_the_depth():
  model = CountingModel()
  blocks = make_beam_planner(MEAN, STD, width=1, max_depth=4, merge_distance=0.1)(model, torch.tensor([0.0]), px_cost(100.0))

  assert blocks.shape[0] == 4 and blocks[:, 0].sum().item() == 4.0


def test_mcts_finds_the_goal_and_spends_the_budget_on_the_promising_side():
  model = CountingModel()
  blocks = make_mcts_planner(MEAN, STD, simulations=100)(model, torch.tensor([0.0]), px_cost(3.0))

  assert blocks[:, 0].sum().item() == 3.0
  assert model.calls < 100  # stops expanding once a node is under stop_at


def test_mcts_moves_toward_an_unreachable_goal_by_visit_count():
  model = CountingModel()
  blocks = make_mcts_planner(MEAN, STD, simulations=60)(model, torch.tensor([0.0]), px_cost(100.0))

  assert model.calls == 60 and blocks[0, 0].item() == 1.0

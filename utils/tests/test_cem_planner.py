import numpy as np
import torch

from utils.cem_planner import ACTION_DIM, HORIZON, NUM_SAMPLES, jepa_cost, plan, rollout, to_environment_actions
from utils.dynamics_probe import BLOCK


class AdditiveModel:
  """Stands in for JEPA: the latent is 1-D and every block moves it by the first number of the block."""

  class predictor:
    num_frames = 3

  def action_encoder(self, actions):
    return actions

  def predict(self, emb, act_emb):
    return emb + act_emb[..., :1]


def test_rollout_adds_up_the_blocks():
  actions = torch.ones(7, HORIZON, ACTION_DIM)

  final = rollout(AdditiveModel(), torch.tensor([2.0]), actions)

  assert final.shape == (7, 1)
  assert torch.allclose(final, torch.full((7, 1), 2.0 + HORIZON))


def test_jepa_cost_is_the_squared_distance_to_the_goal():
  cost = jepa_cost(torch.tensor([[0.0, 0.0], [3.0, 4.0]]), torch.tensor([0.0, 0.0]))

  assert cost.tolist() == [0.0, 25.0]


def test_cem_finds_a_plan_that_reaches_the_goal():
  generator = torch.Generator().manual_seed(0)
  goal = torch.tensor([3.0])
  model = AdditiveModel()

  best = plan(model, torch.tensor([0.0]), lambda final: jepa_cost(final, goal), generator, device="cpu")

  reached = rollout(model, torch.tensor([0.0]), best[None])
  assert abs(reached.item() - 3.0) < 0.1


def test_cem_minimises_whatever_cost_it_is_given_and_scores_all_candidates():
  generator = torch.Generator().manual_seed(1)
  model = AdditiveModel()
  candidates_scored = []

  def cost(final):
    candidates_scored.append(final.shape[0])
    return final[:, 0] ** 2  # best final latent is 0, which needs the blocks to add up to -5

  best = plan(model, torch.tensor([5.0]), cost, generator, device="cpu")

  assert set(candidates_scored) == {NUM_SAMPLES}
  assert abs(rollout(model, torch.tensor([5.0]), best[None]).item()) < 0.2


def test_environment_actions_are_unscaled_and_clipped():
  blocks = torch.zeros(2, ACTION_DIM)
  blocks[0] = 1.0
  blocks[1] = 100.0

  actions = to_environment_actions(blocks, action_mean=np.array([0.0, 0.5]), action_std=np.array([0.5, 0.5]))

  assert actions.shape == (2 * BLOCK, 2)
  assert np.allclose(actions[0], [0.5, 1.0])  # 1 * 0.5 + 0, 1 * 0.5 + 0.5
  assert np.allclose(actions[BLOCK], [1.0, 1.0])  # clipped to the action range


def test_run_episode_stops_at_the_budget_and_reports_distances():
  from stable_worldmodel.envs.two_room import TwoRoomEnv
  from utils.cem_planner import run_episode

  class StubJepa(AdditiveModel):
    def encode(self, info):
      return {"emb": info["pixels"].mean(dim=(2, 3, 4)).unsqueeze(-1)}

  env = TwoRoomEnv()
  env.reset()

  result = run_episode(env, StubJepa(), "cpu", start=(40, 150), goal=(80, 150),
                       make_cost=lambda goal, xy: (lambda final: jepa_cost(final, goal)),
                       action_mean=np.zeros(2), action_std=np.ones(2), budget_steps=30, seed=0)

  assert set(result) == {"success", "steps", "final_distance", "closest_distance"}
  assert 0 < result["steps"] <= 30
  assert result["closest_distance"] <= 40  # the start distance

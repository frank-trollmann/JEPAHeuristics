import numpy as np
import torch

from utils.latent_search import DIRECTIONS, make_search_planner
from utils.tests.test_cem_planner import AdditiveModel

MEAN, STD = np.zeros(2), np.ones(2)


class CountingModel(AdditiveModel):
  """The 1-D stub: a block moves the latent by the x part of its direction. Counts predictor calls."""

  def __init__(self):
    self.calls = 0

  def predict(self, emb, act_emb):
    self.calls += 1
    return super().predict(emb, act_emb)


def px_cost(goal):
  return lambda latents: (latents[:, 0] - goal).abs() * 25.0


def search(kind, goal, **kwargs):
  model = CountingModel()
  settings = dict(merge_distance=0.1, stop_at=10.0)
  settings.update(kwargs)
  blocks = make_search_planner(kind, MEAN, STD, **settings)(model, torch.tensor([0.0]), px_cost(goal))
  return blocks, model.calls


def test_greedy_and_astar_walk_to_the_goal_with_the_fewest_expansions():
  for kind in ("greedy", "astar"):
    blocks, calls = search(kind, goal=3.0)

    assert blocks.shape[0] == 3 and blocks[:, 0].sum().item() == 3.0
    assert calls == 3  # root, then the two nodes on the way; the goal node is never expanded


def test_astar_without_a_heuristic_expands_more_than_greedy():
  assert search("astar", goal=3.0, weight=0.0)[1] > search("greedy", goal=3.0)[1]


def test_unreachable_goal_stops_at_the_budget_and_heads_toward_it():
  blocks, calls = search("greedy", goal=100.0, expansions=5)

  assert calls == 5
  assert blocks[0, 0].item() == 1.0  # first move goes toward the goal


def test_duplicate_latents_are_merged():
  _, calls = search("greedy", goal=100.0, expansions=1000, max_depth=12)

  assert calls <= 30  # on the 1-D line there are only 25 distinct positions within 12 blocks


def test_path_is_never_longer_than_max_depth():
  blocks, _ = search("greedy", goal=100.0, expansions=1000, max_depth=4)

  assert blocks.shape[0] <= 4


def test_already_at_the_goal_stands_still():
  blocks, calls = search("greedy", goal=0.0)

  assert calls == 0 and blocks.shape[0] == 1 and blocks.abs().sum().item() == 0.0

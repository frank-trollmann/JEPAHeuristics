import numpy as np

from utils.trajectory_dataset import split_episodes, temporal_pairs


def make_frames():
  # two episodes of 4 frames each, 5 environment steps apart
  episode = np.array([7, 7, 7, 7, 9, 9, 9, 9])
  step = np.array([0, 5, 10, 15, 0, 5, 10, 15])
  return episode, step


def test_pairs_stay_inside_one_episode_and_count_the_steps_between_frames():
  episode, step = make_frames()

  agent, goal, label = temporal_pairs(episode, step, np.ones(8, dtype=bool), both_directions=False)

  assert len(agent) == 2 * 6  # 4 frames give 6 pairs per episode
  assert (episode[agent] == episode[goal]).all()
  assert (step[goal] > step[agent]).all()
  assert np.allclose(label, step[goal] - step[agent])


def test_reversed_pairs_are_added_with_the_same_label():
  episode, step = make_frames()

  agent, goal, label = temporal_pairs(episode, step, np.ones(8, dtype=bool), both_directions=True)

  assert len(agent) == 24
  forward = {(a, g): l for a, g, l in zip(agent[:12], goal[:12], label[:12])}
  for a, g, l in zip(agent[12:], goal[12:], label[12:]):
    assert forward[(g, a)] == l


def test_pairs_only_use_masked_frames():
  episode, step = make_frames()
  mask = np.array([True, True, False, False, False, False, True, True])

  agent, goal, _ = temporal_pairs(episode, step, mask, both_directions=False)

  assert set(agent) | set(goal) <= {0, 1, 6, 7}
  assert len(agent) == 2  # one pair in each episode


def test_episode_split_gives_every_frame_of_an_episode_the_same_split():
  episode = np.repeat(np.arange(100), 5)

  split = split_episodes(episode, seed=0)

  for e in range(100):
    assert len(set(split[episode == e])) == 1
  assert {"train", "val", "test"} == set(split)
  assert (split == "train").sum() == 400

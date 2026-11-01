import random

from stable_worldmodel.envs.two_room import TwoRoomEnv

from utils.embedding_probing.experiments.obstacle_probing.generate_random_obstacles import (
  _door_geometry,
  _render_agent_at,
)


def _door_sequence(seed, n):
  """
  Mirrors the per-sample door-resampling pattern in
  generate_random_obstacles_dataset: draw a per-sample seed from a
  random.Random(seed) instance, reset the env's door geometry with it,
  and record the resulting (door_y, door_size) sequence.
  """
  env = TwoRoomEnv()
  env.reset()
  rng = random.Random(seed)

  sequence = []
  for _ in range(n):
    door_seed = rng.randrange(2 ** 31)
    env.reset(options={"variation": ["door.position", "door.size"]}, seed=door_seed)
    geom = _door_geometry(env)
    sequence.append((geom["door_y"], geom["door_size"]))
  return sequence


def test_door_geometry_returns_door_y_and_size():
  env = TwoRoomEnv()
  env.reset()

  geom = _door_geometry(env)

  assert set(geom.keys()) == {"door_y", "door_size"}
  assert isinstance(geom["door_y"], float)
  assert isinstance(geom["door_size"], float)


def test_resampling_door_variation_changes_geometry_across_samples():
  env = TwoRoomEnv()
  env.reset()

  geometries = set()
  for i in range(10):
    env.reset(options={"variation": ["door.position", "door.size"]}, seed=i)
    geom = _door_geometry(env)
    geometries.add((geom["door_y"], geom["door_size"]))

  # resampling 10 times should produce more than just one fixed value
  assert len(geometries) > 1


def test_door_geometry_stays_fixed_across_renders_within_one_sample():
  # the whole point of _render_agent_at: rendering agent and goal images
  # must NOT change the door layout chosen for that sample - a plain
  # env.reset(options={"state": ...}) would reset door.position/size back
  # to their init value before this sample's agent/goal renders happen
  env = TwoRoomEnv()
  env.reset()
  env.reset(options={"variation": ["door.position", "door.size"]}, seed=1)

  geom_before = _door_geometry(env)

  _render_agent_at(env, [60.0, 60.0])
  geom_after_agent_render = _door_geometry(env)

  _render_agent_at(env, [160.0, 160.0])
  geom_after_goal_render = _door_geometry(env)

  assert geom_before == geom_after_agent_render == geom_after_goal_render


def test_door_sequence_is_reproducible_with_same_seed():
  sequence_a = _door_sequence(seed=42, n=10)
  sequence_b = _door_sequence(seed=42, n=10)

  assert sequence_a == sequence_b


def test_door_sequence_differs_with_different_seed():
  sequence_a = _door_sequence(seed=42, n=10)
  sequence_b = _door_sequence(seed=7, n=10)

  assert sequence_a != sequence_b


def test_door_sequence_varies_across_samples_within_one_seed():
  sequence = _door_sequence(seed=42, n=10)
  assert len(set(sequence)) > 1

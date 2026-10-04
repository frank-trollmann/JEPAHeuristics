import random

import numpy as np
import pytest
from stable_worldmodel.envs.two_room import TwoRoomEnv

from utils.true_distance.generate_true_distance_dataset import (
    _sample_valid_position,
    _wall_geometry,
    build_occupancy_grid,
    dijkstra_distance,
    true_distance,
    true_distance_batch,
)


@pytest.fixture(scope="module")
def env_geom_walkable():
    env = TwoRoomEnv()
    env.reset()
    geom = _wall_geometry(env)
    walkable = build_occupancy_grid(env)
    return env, geom, walkable


def test_same_room_pairs_equal_euclidean(env_geom_walkable):
    env, geom, walkable = env_geom_walkable
    # both points left of the wall (WALL_CENTER=112) -> same room
    pos_a = [30.0, 30.0]
    pos_g = [60.0, 190.0]

    expected = float(np.linalg.norm(np.array(pos_a) - np.array(pos_g)))
    assert true_distance(pos_a, pos_g, geom) == pytest.approx(expected)


def test_zero_distance_for_identical_points(env_geom_walkable):
    _, geom, _ = env_geom_walkable
    pos = [100.0, 150.0]
    assert true_distance(pos, pos, geom) == pytest.approx(0.0, abs=1e-9)


def test_cross_room_pairs_route_through_door(env_geom_walkable):
    env, geom, walkable = env_geom_walkable
    # left room vs right room (WALL_CENTER=112), aligned with the default
    # door (position ~49, size ~14) so the straight line already passes
    # through the door -> distance should equal plain euclidean distance.
    door_y = geom["door_positions"][0]
    pos_a = [50.0, door_y]
    pos_g = [174.0, door_y]

    expected = float(np.linalg.norm(np.array(pos_a) - np.array(pos_g)))
    assert true_distance(pos_a, pos_g, geom) == pytest.approx(expected, rel=1e-6)


def test_cross_room_pair_is_never_shorter_than_euclidean(env_geom_walkable):
    env, geom, walkable = env_geom_walkable
    rng = random.Random(1)
    for _ in range(30):
        pos_a = _sample_valid_position(env, walkable, rng)
        pos_g = _sample_valid_position(env, walkable, rng)
        d = true_distance(pos_a, pos_g, geom)
        euclid = float(np.linalg.norm(np.array(pos_a) - np.array(pos_g)))
        # detouring through a door can never be shorter than a straight line
        assert d >= euclid - 1e-9


def test_closed_form_matches_dijkstra_within_discretization_margin(env_geom_walkable):
    """Dijkstra operates on a rounded pixel grid, so it's expected to be a
    little larger than the exact closed-form value - but not wildly off."""
    env, geom, walkable = env_geom_walkable
    rng = random.Random(42)
    for _ in range(30):
        pos_a = _sample_valid_position(env, walkable, rng)
        pos_g = _sample_valid_position(env, walkable, rng)

        closed = true_distance(pos_a, pos_g, geom)
        grid = dijkstra_distance(walkable, pos_a, pos_g)

        assert grid >= closed - 1.0  # grid distance shouldn't undercut the exact value (modulo rounding)
        assert grid <= closed * 1.2 + 2.0  # and shouldn't overshoot it by much either


def test_batch_matches_single_pair_loop(env_geom_walkable):
    env, geom, walkable = env_geom_walkable
    rng = random.Random(0)
    pos_a_list = [_sample_valid_position(env, walkable, rng) for _ in range(25)]
    pos_g_list = [_sample_valid_position(env, walkable, rng) for _ in range(25)]

    batch_result = true_distance_batch(pos_a_list, pos_g_list, geom)
    single_results = [true_distance(a, g, geom) for a, g in zip(pos_a_list, pos_g_list)]

    np.testing.assert_allclose(batch_result, single_results, rtol=1e-9)


def test_batch_handles_single_pair():
    geom = {"axis": 1, "center": 112.0, "door_positions": [49.0], "door_sizes": [14.0]}
    result = true_distance_batch([[30.0, 30.0]], [[60.0, 190.0]], geom)
    assert result.shape == (1,)

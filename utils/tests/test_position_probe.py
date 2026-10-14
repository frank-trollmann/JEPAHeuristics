import numpy as np

from utils.position_probe import fit_position_probe, position_errors


def make_encodings(positions, seed=0):
  """Stand-in encodings: a fixed linear mix of the position into 20 dimensions."""
  mix = np.random.default_rng(seed).normal(size=(2, 20))
  return positions @ mix


def test_linear_probe_recovers_a_position_that_is_linearly_encoded():
  positions = np.random.default_rng(1).uniform(20, 200, size=(500, 2))
  encodings = make_encodings(positions)

  model = fit_position_probe(encodings[:400], positions[:400])
  errors = position_errors(model, encodings[400:], positions[400:])

  assert errors["mean_position_error"] < 0.01


def test_mlp_probe_reads_the_position_much_better_than_guessing():
  positions = np.random.default_rng(2).uniform(20, 200, size=(1500, 2))
  encodings = make_encodings(positions)

  model = fit_position_probe(encodings[:1200], positions[:1200], hidden=(64, 64))
  errors = position_errors(model, encodings[1200:], positions[1200:])

  mean_guess_error = np.linalg.norm(positions[1200:] - positions[:1200].mean(axis=0), axis=1).mean()
  assert errors["mean_position_error"] < 0.2 * mean_guess_error


def test_position_errors_are_zero_for_a_perfect_model():
  positions = np.array([[1.0, 2.0], [3.0, 4.0]])

  class Perfect:
    def predict(self, encodings):
      return encodings

  errors = position_errors(Perfect(), positions, positions)

  assert errors == {"mae_x": 0.0, "mae_y": 0.0, "mean_position_error": 0.0}

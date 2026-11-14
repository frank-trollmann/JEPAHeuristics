import pytest
import torch

from utils.embedding_probing.experiments.shared.normalization import normalise_images


def test_normalise_images_preserves_shape():
  images = torch.randint(0, 256, (2, 1, 3, 4, 4), dtype=torch.uint8)
  out = normalise_images(images)
  assert out.shape == images.shape


def test_normalise_images_returns_float_tensor():
  images = torch.randint(0, 256, (2, 3, 4, 4), dtype=torch.uint8)
  out = normalise_images(images)
  assert out.dtype == torch.float32


def test_normalise_images_matches_known_imagenet_values_at_max_pixel():
  # input value 255 (max) should normalise to (1 - mean) / std per channel
  images = torch.full((1, 3, 1, 1), 255.0)
  out = normalise_images(images)

  expected_r = (1.0 - 0.485) / 0.229
  expected_g = (1.0 - 0.456) / 0.224
  expected_b = (1.0 - 0.406) / 0.225

  assert out[0, 0, 0, 0].item() == pytest.approx(expected_r, rel=1e-4)
  assert out[0, 1, 0, 0].item() == pytest.approx(expected_g, rel=1e-4)
  assert out[0, 2, 0, 0].item() == pytest.approx(expected_b, rel=1e-4)


def test_normalise_images_matches_known_imagenet_values_at_zero_pixel():
  # input value 0 (min) should normalise to (0 - mean) / std per channel
  images = torch.zeros((1, 3, 1, 1))
  out = normalise_images(images)

  expected_r = (0.0 - 0.485) / 0.229
  expected_g = (0.0 - 0.456) / 0.224
  expected_b = (0.0 - 0.406) / 0.225

  assert out[0, 0, 0, 0].item() == pytest.approx(expected_r, rel=1e-4)
  assert out[0, 1, 0, 0].item() == pytest.approx(expected_g, rel=1e-4)
  assert out[0, 2, 0, 0].item() == pytest.approx(expected_b, rel=1e-4)


def test_normalise_images_handles_uint8_input():
  # rendered frames come as uint8 - make sure the division doesn't truncate
  # to integers before the float conversion
  images = torch.full((1, 3, 1, 1), 128, dtype=torch.uint8)
  out = normalise_images(images)

  expected_r = (128.0 / 255.0 - 0.485) / 0.229
  assert out[0, 0, 0, 0].item() == pytest.approx(expected_r, rel=1e-4)

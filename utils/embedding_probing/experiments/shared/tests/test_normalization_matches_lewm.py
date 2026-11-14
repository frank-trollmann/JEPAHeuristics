import numpy as np
import pytest
import torch
from torchvision.transforms import v2 as transforms
from stable_pretraining.data import dataset_stats

from utils.embedding_probing.experiments.shared.normalization import normalise_images


def test_matches_the_transform_of_the_official_lewm_evaluation():
  # le-wm/eval.py img_transform: ToImage, ToDtype(float32, scale=True), Normalize(ImageNet)
  official = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(**dataset_stats.ImageNet),
  ])
  frame = np.random.default_rng(0).integers(0, 256, size=(224, 224, 3), dtype=np.uint8)

  expected = official(frame)
  actual = normalise_images(torch.from_numpy(frame).permute(2, 0, 1))

  assert torch.allclose(actual, expected, atol=1e-5)


def test_black_and_white_and_the_batch_shape_the_generators_use():
  mean = torch.tensor(dataset_stats.ImageNet["mean"])
  std = torch.tensor(dataset_stats.ImageNet["std"])
  batch = torch.zeros(2, 1, 3, 4, 4)  # (batch, time, channel, H, W) like the generators
  batch[1] = 255.0

  out = normalise_images(batch)

  assert out.shape == batch.shape and out.dtype == torch.float32
  assert torch.allclose(out[0, 0, :, 0, 0], -mean / std)
  assert torch.allclose(out[1, 0, :, 0, 0], (1 - mean) / std)

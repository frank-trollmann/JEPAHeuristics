"""
Image preparation for the LeWM encoder.

The encoder was trained and is evaluated on images that are scaled to [0, 1]
and then normalised with the ImageNet mean and standard deviation per colour
channel (le-wm/utils.py get_img_preprocessor, le-wm/eval.py img_transform).
Rendered frames have values 0-255, and LeWM.encode does not rescale them.
"""
import torch
from stable_pretraining.data import dataset_stats


def normalise_images(images):
    """
    images: colour channels at dimension -3 (shape (..., 3, H, W)), values 0-255, any numeric dtype.
    Returns a float tensor of the same shape on the same device.
    """
    mean = torch.tensor(dataset_stats.ImageNet["mean"], device=images.device).view(3, 1, 1)
    std = torch.tensor(dataset_stats.ImageNet["std"], device=images.device).view(3, 1, 1)
    return (images.float() / 255.0 - mean) / std

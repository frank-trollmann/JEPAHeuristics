import os
import stable_worldmodel as swm
import h5py
import matplotlib.pyplot as plt
from einops import rearrange
from torch.utils.data import DataLoader
import torch
import random
import numpy as np

if not "working_directory_corrected" in vars():
    %cd ..
    working_directory_corrected = True

cache_dir = os.environ.get("/Users/richard/Projects/CODE/JEPA Planning Project/JEPAHeuristics/datasets", None)
dataset = swm.data.load_dataset(
  "tworoom.h5",
  transform= None,
  cache_dir=cache_dir,
  format = "hdf5"
)

print(dataset[0].keys())
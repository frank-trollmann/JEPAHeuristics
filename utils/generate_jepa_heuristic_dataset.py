import random
import torch
import numpy as np
import pandas as pd
from stable_worldmodel.envs.two_room import TwoRoomEnv
import utils.load_fix_pretrained as load_util
from huggingface_hub import snapshot_download

def generate_jepa_heuristic_dataset(num_samples=1000, checkpoint_path=None):
  """
  Generates a dataset of (agent_encoding, goal_encoding, euclidean_distance).
  """
  # Setup Environment and Model
  print("Initializing Environment and Model...")
  env = TwoRoomEnv()

  # If the checkpoint_path is not specified it will be defaulted to the users huggingface cache, where huggingface will have dowloaded the model
  if checkpoint_path is None:
    checkpoint_path = snapshot_download(
    repo_id="quentinll/lewm-tworooms",
    revision="77adaae0bc31deab21c93740d1f8bb947cd0bdec",
    )

  model = load_util.load_fix_pretrained(checkpoint_path)
  model.eval()  # Set to evaluation mode

  dataset_rows = []

  print(f"Generating {num_samples} samples...")
  with torch.no_grad():
    for i in range(num_samples):
      # Sample random coordinates (0-244 range)
      pos_a = [random.randint(0, 244), random.randint(0, 244)]
      pos_g = [random.randint(0, 244), random.randint(0, 244)]

      # Extract images from environment
      # Agent Position
      env.reset(options={"state": pos_a})
      img_a = env.render()  # Expected shape: (244, 244, 3)

      # Goal Position
      env.reset(options={"state": pos_g})
      img_g = env.render()  # Expected shape: (244, 244, 3)

      # Preprocess images for the model
      # Model expects (Batch, Time, Channel, H, W) -> (1, 1, 3, 244, 244)
      # Convert numpy array to tensor, permute to (C, H, W),
      # then unsqueeze twice: once for Batch, once for Time.
      t_img_a = torch.from_numpy(img_a).permute(2, 0, 1).float().unsqueeze(0).unsqueeze(0)
      t_img_g = torch.from_numpy(img_g).permute(2, 0, 1).float().unsqueeze(0).unsqueeze(0)

      # Encode images to Latent Space
      # We wrap in a dict because model.encode expects the format used by the DataLoader
      enc_a = model.encode({"pixels": t_img_a})["emb"]  # Shape: (1, 1, 3, 192) or similar
      enc_g = model.encode({"pixels": t_img_g})["emb"]

      # Squeeze the tensors to get a flat 192-dim vector
      vec_a = enc_a.squeeze().numpy()
      vec_g = enc_g.squeeze().numpy()

      # Calculate Ground Truth Heuristic (Euclidean Distance)
      dist = np.linalg.norm(np.array(pos_a) - np.array(pos_g))

      dataset_rows.append({
        "agent_pos_enc": vec_a,
        "goal_pos_enc": vec_g,
        "heuristic": dist
      })

      if (i + 1) % 100 == 0:
        print(f"Progress: {i + 1}/{num_samples}")

  return pd.DataFrame(dataset_rows)
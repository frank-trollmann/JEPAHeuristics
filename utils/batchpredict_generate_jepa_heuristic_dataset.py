import random
import torch
import numpy as np
import pandas as pd
from stable_worldmodel.envs.two_room import TwoRoomEnv
import utils.load_fix_pretrained as load_util
from huggingface_hub import snapshot_download

def batchpredict_generate_jepa_heuristic_dataset(num_samples=1000, batch_size=32, checkpoint_path=None):
  """
  Generates a dataset of (agent_encoding, goal_encoding, euclidean_distance) by using batch-prediction of the JEPA model.
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

  if torch.cuda.is_available():
    device = torch.device("cuda")
  elif torch.backends.mps.is_available():
    device = torch.device("mps")
  else:
    device = torch.device("cpu")
  model = load_util.load_fix_pretrained(checkpoint_path)
  model.to(device)
  model.eval()  # Set to evaluation mode

  dataset_rows = []

  # Process in chunks of batch_size
  total_batches = (num_samples + batch_size - 1) // batch_size
  min_c, max_c = env.BORDER_SIZE, env.IMG_SIZE - env.BORDER_SIZE
  for i in range(0, num_samples, batch_size):
    current_batch_size = min(batch_size, num_samples - i)
    print(f"Processing batch {i//batch_size + 1}/{total_batches} | Samples {i+1} to {min(i + batch_size, num_samples)}")

    agent_images = []
    goal_images = []
    coords_a = []
    coords_g = []

    # Sample only within the safe content area (excluding borders)
    for _ in range(current_batch_size):
      pos_a = [random.randint(min_c, max_c), random.randint(min_c, max_c)]
      pos_g = [random.randint(min_c, max_c), random.randint(min_c, max_c)]

      env.reset(options={"state": pos_a})
      agent_images.append(env.render())
            
      env.reset(options={"state": pos_g})
      goal_images.append(env.render())
            
      coords_a.append(pos_a)
      coords_g.append(pos_g)

    # 2. Convert lists to Batched Tensors
    # Shape: (Batch, Time=1, Channel, H, W) -> (batch_size, 1, 3, 224, 224)
    t_batch_a = torch.from_numpy(np.array(agent_images)).permute(0, 3, 1, 2).float().unsqueeze(1).to(device)
    t_batch_g = torch.from_numpy(np.array(goal_images)).permute(0, 3, 1, 2).float().unsqueeze(1).to(device)

    # 3. Batch Encode (The GPU "Magic" happens here)
    with torch.no_grad():
      enc_a_batch = model.encode({"pixels": t_batch_a})["emb"]  # Shape: (batch_size, 1, 3, 192)
      enc_g_batch = model.encode({"pixels": t_batch_g})["emb"]

    # 4. Move back to CPU and unpack
    vecs_a = enc_a_batch.detach().cpu().squeeze().numpy()
    vecs_g = enc_g_batch.detach().cpu().squeeze().numpy()

    # 5. Save to dataset_rows
    for j in range(current_batch_size):
      dist = np.linalg.norm(np.array(coords_a[j]) - np.array(coords_g[j]))
      dataset_rows.append({
        "agent_pos_enc": vecs_a[j],
        "goal_pos_enc": vecs_g[j],
        "heuristic": dist
      })

  return pd.DataFrame(dataset_rows)
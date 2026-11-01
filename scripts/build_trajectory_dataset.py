import argparse
import os
from datetime import datetime

import numpy as np
import torch
from huggingface_hub import snapshot_download

import utils.load_fix_pretrained as load_util
from utils.trajectory_dataset import encode_trajectories


def main():
    parser = argparse.ArgumentParser(description="Encode frames of the original LeWM trajectories")
    parser.add_argument("--checkpoint", type=str, default=None, help="Folder with config.json and weights.pt. Defaults to the pinned HuggingFace checkpoint.")
    parser.add_argument("--h5", type=str, default="stable_worldmodel/datasets/tworoom/tworoom.h5")
    parser.add_argument("--episodes", type=int, default=4000, help="Random subset of the 10,000 episodes")
    parser.add_argument("--stride", type=int, default=5, help="Keep every stride-th frame (the model step is 5 environment steps)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or snapshot_download(
        repo_id="quentinll/lewm-tworooms",
        revision="77adaae0bc31deab21c93740d1f8bb947cd0bdec",
    )
    model = load_util.load_fix_pretrained(checkpoint).to(device).eval()

    data = encode_trajectories(args.h5, model, device, args.episodes, args.stride, args.seed)
    os.makedirs("data", exist_ok=True)
    path = f"data/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_trajectory_dataset_{args.episodes}.npz"
    np.savez_compressed(path, **data)
    print(f"Saved {len(data['encodings'])} frames from {args.episodes} episodes to {path}")


if __name__ == "__main__":
    main()

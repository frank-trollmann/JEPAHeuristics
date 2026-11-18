import argparse
import os
from datetime import datetime

import numpy as np
import torch
from huggingface_hub import snapshot_download
from stable_worldmodel.envs.two_room import TwoRoomEnv

import utils.load_fix_pretrained as load_util
from utils.position_dataset import encode_positions, split_positions, valid_positions


def main():
    parser = argparse.ArgumentParser(
        description="Render and encode every valid TwoRoom position once, with a train/val/test split by position"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Folder with config.json and weights.pt. Defaults to the pinned HuggingFace checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the position split")
    parser.add_argument("--batch-size", type=int, default=256, help="Images encoded per GPU batch")
    args = parser.parse_args()

    checkpoint = args.checkpoint or snapshot_download(
        repo_id="quentinll/lewm-tworooms",
        revision="77adaae0bc31deab21c93740d1f8bb947cd0bdec",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_util.load_fix_pretrained(checkpoint).to(device).eval()

    env = TwoRoomEnv()
    positions = valid_positions(env)
    splits = split_positions(len(positions), seed=args.seed)
    print(f"{len(positions)} valid positions, device: {device}")

    encodings = encode_positions(env, model, positions, device, args.batch_size)

    os.makedirs("data", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = f"data/{timestamp}_position_dataset_seed{args.seed}.npz"
    np.savez_compressed(path, positions=positions, encodings=encodings, splits=splits)
    print(f"Saved {len(positions)} encoded positions to {path}")


if __name__ == "__main__":
    main()

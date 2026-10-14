import argparse
import glob
import os
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from stable_worldmodel.envs.two_room import TwoRoomEnv

import utils.load_fix_pretrained as load_util
from utils.dynamics_probe import CONDITIONS, action_statistics, make_trials, run_trial, summarise
from utils.pair_dataset import load_position_dataset
from utils.position_probe import fit_position_probe


def main():
    parser = argparse.ArgumentParser(description="Test whether the JEPA predictor knows about the wall")
    parser.add_argument("--dataset", type=str, default=None, help="Position dataset .npz (for the position decoder). Defaults to the newest in data/")
    parser.add_argument("--checkpoint", type=str, default=None, help="Folder with config.json and weights.pt. Defaults to the pinned HuggingFace checkpoint.")
    parser.add_argument("--actions-h5", type=str, default="stable_worldmodel/datasets/tworoom/tworoom.h5", help="LeWM dataset, only used for the action mean and std")
    parser.add_argument("--trials", type=int, default=200, help="Trials per condition")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or snapshot_download(
        repo_id="quentinll/lewm-tworooms",
        revision="77adaae0bc31deab21c93740d1f8bb947cd0bdec",
    )
    model = load_util.load_fix_pretrained(checkpoint).to(device).eval()

    with h5py.File(args.actions_h5) as f:
        mean, std = action_statistics(f["action"][:])

    # decoder from a latent to (x, y), fitted on the training positions only
    path = args.dataset or sorted(glob.glob("data/*_position_dataset_*.npz"))[-1]
    positions, encodings, splits = load_position_dataset(path)
    train = splits == "train"
    decoder = fit_position_probe(encodings[train], positions[train], hidden=(256, 256))

    env = TwoRoomEnv()
    env.reset()
    door_center = float(env.door_positions[0])
    rng = np.random.default_rng(args.seed)

    rows = []
    for condition in CONDITIONS:
        results = []
        for start, action in make_trials(condition, args.trials, door_center, rng):
            positions_run, latents, predicted = run_trial(env, model, device, start, action, mean, std)
            results.append((positions_run, latents, predicted, action[0]))
        rows.append(summarise(condition, results, decoder))
        print(f"done: {condition}")

    table = pd.DataFrame(rows)
    os.makedirs("data/results", exist_ok=True)
    out = f"data/results/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_dynamics_probe.csv"
    table.to_csv(out, index=False)
    print(table.round(2).to_string(index=False))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

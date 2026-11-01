import argparse
import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from stable_worldmodel.envs.two_room import TwoRoomEnv

from utils.heuristic_models import latent_distance, metrics
from utils.pair_dataset import crosses_wall, euclidean_labels, load_position_dataset, sample_pairs
from utils.reference_distance import door_geometry, formula_labels
from utils.trajectory_dataset import temporal_pairs, train_temporal_heuristic


def residual(values, explained_by):
    """What is left of values after a straight line in explained_by has been removed."""
    slope, intercept = np.polyfit(explained_by, values, 1)
    return values - (slope * explained_by + intercept)


def main():
    parser = argparse.ArgumentParser(description="Train a heuristic from the temporal order of frames and test how wall-aware it is")
    parser.add_argument("--trajectory-dataset", type=str, default=None, help="Newest data/*_trajectory_dataset_*.npz by default")
    parser.add_argument("--dataset", type=str, default=None, help="Position dataset .npz. Defaults to the newest in data/")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trajectory = dict(np.load(args.trajectory_dataset or sorted(glob.glob("data/*_trajectory_dataset_*.npz"))[-1]))
    heuristic, split = train_temporal_heuristic(trajectory, device, args.seed)

    # 1. how well does it predict the steps between two frames of unseen episodes?
    agent, goal, steps = temporal_pairs(trajectory["episode"], trajectory["step"], split == "test")
    print("unseen episodes:", metrics(steps, heuristic(agent, goal)), "(steps)")

    # 2. is it wall-aware? Compare with the wall-aware distance on pairs of fresh positions (never used for training)
    positions, encodings, splits = load_position_dataset(args.dataset or sorted(glob.glob("data/*_position_dataset_*.npz"))[-1])
    agent, goal = sample_pairs(splits, "test", 50_000, seed=0)
    env = TwoRoomEnv()
    env.reset()
    formula = formula_labels(positions[agent].astype(float), positions[goal].astype(float), door_geometry(env, inflated=True))
    straight = euclidean_labels(positions, agent, goal)
    cross = crosses_wall(positions, agent, goal, TwoRoomEnv.WALL_CENTER)

    to_torch = lambda z: torch.tensor(z, dtype=torch.float32, device=device)
    scores = {
        "temporal heuristic (steps)": heuristic.on_latents(to_torch(encodings[agent]), to_torch(encodings[goal])).cpu().numpy(),
        "squared latent distance (JEPA cost)": latent_distance(encodings, agent, goal, squared=True)[:, 0],
        "latent distance": latent_distance(encodings, agent, goal, squared=False)[:, 0],
        "Euclidean distance (coordinates)": straight,
    }
    rows = []
    for name, score in scores.items():
        row = {"measure": name,
               "pearson_all": np.corrcoef(score, formula)[0, 1],
               "spearman_all": spearmanr(score, formula)[0],
               "spearman_same_room": spearmanr(score[~cross], formula[~cross])[0],
               "spearman_cross_wall": spearmanr(score[cross], formula[cross])[0],
               # beyond what the straight-line distance explains: does the measure see the detour?
               "detour_correlation": np.corrcoef(residual(score, straight), residual(formula, straight))[0, 1]}
        rows.append(row)
    table = pd.DataFrame(rows)
    os.makedirs("data/results", exist_ok=True)
    out = f"data/results/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_temporal_heuristic.csv"
    table.to_csv(out, index=False)
    print(table.round(3).to_string(index=False))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

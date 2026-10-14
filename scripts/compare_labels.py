import argparse
import glob
import os
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd

from scripts.run_label_experiment import LABELS, make_labeler
from utils.pair_dataset import crosses_wall, load_position_dataset, sample_pairs
from stable_worldmodel.envs.two_room import TwoRoomEnv


def main():
    parser = argparse.ArgumentParser(description="How much do the distance labels differ from each other?")
    parser.add_argument("--dataset", type=str, default=None, help="Position dataset .npz. Defaults to the newest in data/")
    parser.add_argument("--pairs", type=int, default=200_000)
    args = parser.parse_args()

    path = args.dataset or sorted(glob.glob("data/*_position_dataset_*.npz"))[-1]
    positions, _, splits = load_position_dataset(path)
    agent, goal = sample_pairs(splits, "test", args.pairs, seed=0)
    cross = crosses_wall(positions, agent, goal, TwoRoomEnv.WALL_CENTER)

    labels = {name: make_labeler(name, positions)(agent, goal) for name in LABELS}

    rows = []
    for first, second in combinations(LABELS, 2):
        difference = labels[second] - labels[first]  # positive: the second label is longer
        rows.append({
            "from": first,
            "to": second,
            "mean_difference": difference.mean(),
            "mae": np.abs(difference).mean(),
            "mae_same_room": np.abs(difference)[~cross].mean(),
            "mae_cross_wall": np.abs(difference)[cross].mean(),
            "max_abs": np.abs(difference).max(),
            "mean_relative_cross_wall": (difference / np.maximum(labels[first], 1))[cross].mean(),
        })

    table = pd.DataFrame(rows)
    os.makedirs("data/results", exist_ok=True)
    out = f"data/results/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_label_comparison.csv"
    table.to_csv(out, index=False)
    print(table.round(2).to_string(index=False))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

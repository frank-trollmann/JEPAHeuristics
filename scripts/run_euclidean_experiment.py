import argparse
import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from stable_worldmodel.envs.two_room import TwoRoomEnv

from utils.heuristic_models import (
    fit_linear,
    fit_mean_baseline,
    latent_distance,
    metrics,
    train_mlp,
)
from utils.pair_dataset import (
    concat_features,
    crosses_wall,
    euclidean_labels,
    load_position_dataset,
    sample_pairs,
)

PAIRS_PER_SPLIT = {"train": 200_000, "val": 50_000, "test": 50_000}
PAIR_SEED = 0


def build_split(positions, encodings, splits, split):
    """Pairs of one split with all inputs the models need."""
    agent, goal = sample_pairs(splits, split, PAIRS_PER_SPLIT[split], PAIR_SEED)
    return {
        "agent": agent,
        "goal": goal,
        "concat": concat_features(encodings, agent, goal),
        "latent_distance": latent_distance(encodings, agent, goal, squared=False),
        "latent_distance_squared": latent_distance(encodings, agent, goal, squared=True),
        "label": euclidean_labels(positions, agent, goal),
        "crosses_wall": crosses_wall(positions, agent, goal, TwoRoomEnv.WALL_CENTER),
    }


def score(model_name, seed, split, data, prediction):
    """One result row: overall metrics plus MAE for same-room and across-the-wall pairs."""
    cross = data["crosses_wall"]
    label = data["label"]
    return {
        "model": model_name,
        "seed": seed,
        "split": split,
        **metrics(label, prediction),
        "mae_same_room": np.abs(label - prediction)[~cross].mean(),
        "mae_cross_wall": np.abs(label - prediction)[cross].mean(),
    }


def main():
    parser = argparse.ArgumentParser(description="Learn the Euclidean distance from latent encodings and compare to baselines")
    parser.add_argument("--dataset", type=str, default=None, help="Position dataset .npz. Defaults to the newest in data/")
    parser.add_argument("--seeds", type=int, default=3, help="Number of MLP seeds")
    args = parser.parse_args()

    path = args.dataset or sorted(glob.glob("data/*_position_dataset_*.npz"))[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dataset: {path}, device: {device}")

    positions, encodings, splits = load_position_dataset(path)
    data = {s: build_split(positions, encodings, splits, s) for s in PAIRS_PER_SPLIT}
    train = data["train"]

    # (name, seed, function mapping a split's data to predictions)
    fitted = []

    mean_model = fit_mean_baseline(train["label"])
    fitted.append(("baseline: mean", 0, lambda d: mean_model.predict(np.zeros((len(d["label"]), 1)))))

    for name, key in [("baseline: latent distance", "latent_distance"),
                      ("baseline: squared latent distance (JEPA cost)", "latent_distance_squared")]:
        line = fit_linear(train[key], train["label"])
        fitted.append((name, 0, lambda d, line=line, key=key: line.predict(d[key])))

    linear = fit_linear(train["concat"], train["label"])
    fitted.append(("linear regression on concat", 0, lambda d: linear.predict(d["concat"])))

    for seed in range(args.seeds):
        predict = train_mlp(train["concat"], train["label"], data["val"]["concat"], data["val"]["label"], device, seed)
        fitted.append(("MLP on concat", seed, lambda d, predict=predict: predict(d["concat"])))

    rows = [
        score(name, seed, split, data[split], predict(data[split]))
        for name, seed, predict in fitted
        for split in ("val", "test")
    ]
    results = pd.DataFrame(rows)

    os.makedirs("data/results", exist_ok=True)
    out = f"data/results/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_euclidean_experiment.csv"
    results.to_csv(out, index=False)

    summary = results.groupby(["split", "model"]).mean(numeric_only=True).drop(columns="seed")
    print(summary.round(3).to_string())
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

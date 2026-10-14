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
from utils.reference_distance import build_grid_graph, grid_labels

LABELS = ("euclidean", "grid")
BASELINE_PAIRS = 200_000  # baselines and linear regression always train on this many pairs
EVAL_PAIRS = 50_000
PAIR_SEED = 0


def make_labeler(label, positions):
    """Returns a function mapping (agent, goal) index arrays to label values in pixels."""
    if label == "euclidean":
        return lambda agent, goal: euclidean_labels(positions, agent, goal)
    graph = build_grid_graph(positions)
    return lambda agent, goal: grid_labels(graph, agent, goal)


def make_pairs(agent, goal, label, positions):
    return {
        "agent": agent,
        "goal": goal,
        "label": label,
        "crosses_wall": crosses_wall(positions, agent, goal, TwoRoomEnv.WALL_CENTER),
    }


def score(model_name, train_pairs, seed, split, pairs, prediction):
    """One result row: overall metrics plus MAE for same-room and across-the-wall pairs."""
    cross = pairs["crosses_wall"]
    error = np.abs(pairs["label"] - prediction)
    return {
        "model": model_name,
        "train_pairs": train_pairs,
        "seed": seed,
        "split": split,
        **metrics(pairs["label"], prediction),
        "mae_same_room": error[~cross].mean(),
        "mae_cross_wall": error[cross].mean(),
    }


def main():
    parser = argparse.ArgumentParser(description="Learn a distance label from latent encodings and compare to baselines")
    parser.add_argument("--dataset", type=str, default=None, help="Position dataset .npz. Defaults to the newest in data/")
    parser.add_argument("--label", choices=LABELS, default="euclidean", help="Which distance the models must predict")
    parser.add_argument("--train-pairs", type=int, nargs="+", default=[200_000], help="MLP training set sizes (a learning curve if several)")
    parser.add_argument("--seeds", type=int, default=3, help="Number of MLP seeds")
    args = parser.parse_args()

    path = args.dataset or sorted(glob.glob("data/*_position_dataset_*.npz"))[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dataset: {path}, label: {args.label}, device: {device}")

    positions, encodings, splits = load_position_dataset(path)
    labeler = make_labeler(args.label, positions)

    # Training pairs are one large sample; smaller training sets are its first n pairs.
    largest = max(args.train_pairs + [BASELINE_PAIRS])
    train_agent, train_goal = sample_pairs(splits, "train", largest, PAIR_SEED)
    train_label = labeler(train_agent, train_goal)
    print(f"Labelled {largest} training pairs")

    evaluation = {}
    for split in ("val", "test"):
        agent, goal = sample_pairs(splits, split, EVAL_PAIRS, PAIR_SEED)
        evaluation[split] = make_pairs(agent, goal, labeler(agent, goal), positions)

    base = slice(0, BASELINE_PAIRS)
    base_agent, base_goal, base_label = train_agent[base], train_goal[base], train_label[base]

    # (name, train pairs, seed, function mapping pairs to predictions)
    fitted = []

    mean_model = fit_mean_baseline(base_label)
    fitted.append(("baseline: mean", BASELINE_PAIRS, 0,
                   lambda p: mean_model.predict(np.zeros((len(p["label"]), 1)))))

    if args.label != "euclidean":
        fitted.append(("baseline: Euclidean distance as the guess", BASELINE_PAIRS, 0,
                       lambda p: euclidean_labels(positions, p["agent"], p["goal"])))

    for name, squared in [("baseline: latent distance", False),
                          ("baseline: squared latent distance (JEPA cost)", True)]:
        line = fit_linear(latent_distance(encodings, base_agent, base_goal, squared), base_label)
        fitted.append((name, BASELINE_PAIRS, 0,
                       lambda p, line=line, squared=squared: line.predict(latent_distance(encodings, p["agent"], p["goal"], squared))))

    linear = fit_linear(concat_features(encodings, base_agent, base_goal), base_label)
    fitted.append(("linear regression on concat", BASELINE_PAIRS, 0,
                   lambda p: linear.predict(concat_features(encodings, p["agent"], p["goal"]))))

    for size in args.train_pairs:
        n = slice(0, size)
        for seed in range(args.seeds):
            predict = train_mlp(
                encodings,
                (train_agent[n], train_goal[n], train_label[n]),
                (evaluation["val"]["agent"], evaluation["val"]["goal"], evaluation["val"]["label"]),
                device, seed,
            )
            fitted.append(("MLP on concat", size, seed, lambda p, predict=predict: predict(p["agent"], p["goal"])))
            print(f"Trained MLP: {size} pairs, seed {seed}")

    results = pd.DataFrame([
        score(name, size, seed, split, evaluation[split], predict(evaluation[split]))
        for name, size, seed, predict in fitted
        for split in ("val", "test")
    ])

    os.makedirs("data/results", exist_ok=True)
    out = f"data/results/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.label}_experiment.csv"
    results.to_csv(out, index=False)

    summary = results.groupby(["split", "model", "train_pairs"]).mean(numeric_only=True).drop(columns="seed")
    print(summary.round(3).to_string())
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

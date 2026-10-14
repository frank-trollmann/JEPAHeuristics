import argparse
import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from stable_worldmodel.envs.two_room import TwoRoomEnv

from utils.heuristic_models import (
    MODEL_KINDS,
    fit_linear_in_chunks,
    latent_distance,
    metrics,
    train_pair_model,
)
from utils.pair_dataset import (
    concat_features,
    crosses_wall,
    euclidean_labels,
    load_position_dataset,
    sample_pairs,
)
from utils.position_probe import fit_position_probe
from utils.reference_distance import build_grid_graph, door_geometry, formula_labels, grid_labels

LABELS = ("euclidean", "grid", "formula", "thin_formula")
EVAL_PAIRS = 50_000
PAIR_SEED = 0
OVERESTIMATE_TOLERANCE = 2.0  # px; a guess this much above the label counts as an overestimate
MODEL_NAMES = {
    "mlp": "MLP on concat",
    "mlp_diff": "MLP on concat + difference",
    "metric": "metric model ||f(z_a) - f(z_g)||",
}


def make_labeler(label, positions):
    """Returns a function mapping (agent, goal) index arrays to label values in pixels."""
    if label == "euclidean":
        return lambda agent, goal: euclidean_labels(positions, agent, goal)
    if label == "grid":
        graph = build_grid_graph(positions)
        return lambda agent, goal: grid_labels(graph, agent, goal)

    env = TwoRoomEnv()
    env.reset()
    geometry = door_geometry(env, inflated=(label == "formula"))
    coordinates = positions.astype(float)
    return lambda agent, goal: formula_labels(coordinates[agent], coordinates[goal], geometry)


def make_pairs(agent, goal, label, positions):
    return {
        "agent": agent,
        "goal": goal,
        "label": label,
        "crosses_wall": crosses_wall(positions, agent, goal, TwoRoomEnv.WALL_CENTER),
    }


def score(label_name, model_name, train_pairs, seed, split, pairs, prediction):
    """
    One result row: overall metrics, MAE for same-room and across-the-wall pairs, and how often and
    by how much the guess is too high (a heuristic that overestimates is not admissible for A*).
    """
    cross = pairs["crosses_wall"]
    error = prediction - pairs["label"]
    return {
        "label": label_name,
        "model": model_name,
        "train_pairs": train_pairs,
        "seed": seed,
        "split": split,
        **metrics(pairs["label"], prediction),
        "mae_same_room": np.abs(error)[~cross].mean(),
        "mae_cross_wall": np.abs(error)[cross].mean(),
        "overestimate_share": (error > OVERESTIMATE_TOLERANCE).mean(),
        "mean_overestimate": np.maximum(error, 0).mean(),
    }


def main():
    parser = argparse.ArgumentParser(description="Learn a distance label from latent encodings and compare to baselines")
    parser.add_argument("--dataset", type=str, default=None, help="Position dataset .npz. Defaults to the newest in data/")
    parser.add_argument("--label", choices=LABELS, default="euclidean", help="Which distance the models must predict")
    parser.add_argument("--train-pairs", type=int, nargs="+", default=[200_000], help="Training set sizes (a learning curve if several)")
    parser.add_argument("--models", choices=MODEL_KINDS, nargs="+", default=["mlp"], help="Learned models to train")
    parser.add_argument("--decode", action="store_true", help="Also score the baselines that decode (x, y) from each encoding first")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds per learned model")
    args = parser.parse_args()

    path = args.dataset or sorted(glob.glob("data/*_position_dataset_*.npz"))[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dataset: {path}, label: {args.label}, device: {device}")

    positions, encodings, splits = load_position_dataset(path)
    labeler = make_labeler(args.label, positions)

    # Training pairs are one large sample; smaller training sets are its first n pairs.
    largest = max(args.train_pairs)
    train_agent, train_goal = sample_pairs(splits, "train", largest, PAIR_SEED)
    train_label = labeler(train_agent, train_goal)
    print(f"Labelled {largest} training pairs")

    evaluation = {}
    for split in ("val", "test"):
        agent, goal = sample_pairs(splits, split, EVAL_PAIRS, PAIR_SEED)
        evaluation[split] = make_pairs(agent, goal, labeler(agent, goal), positions)

    # (name, train pairs, seed, function mapping pairs to predictions)
    fitted = []

    if args.label != "euclidean":
        # needs no training, so train_pairs is 0
        fitted.append(("baseline: Euclidean distance as the guess", 0, 0,
                       lambda p: euclidean_labels(positions, p["agent"], p["goal"])))

    if args.decode:
        # decode (x, y) from every encoding with a model fitted on the training positions only
        decoder = fit_position_probe(encodings[splits == "train"], positions[splits == "train"], hidden=(256, 256))
        decoded = decoder.predict(encodings)
        snapped = cKDTree(positions).query(decoded)[1]  # nearest valid position, so every label is defined
        fitted.append(("baseline: decode position, then Euclidean", 0, 0,
                       lambda p: np.linalg.norm(decoded[p["agent"]] - decoded[p["goal"]], axis=1)))
        fitted.append(("baseline: decode position, then reference distance", 0, 0,
                       lambda p: labeler(snapped[p["agent"]], snapped[p["goal"]])))

    latent_features = {
        "baseline: latent distance": lambda a, g: latent_distance(encodings, a, g, squared=False),
        "baseline: squared latent distance (JEPA cost)": lambda a, g: latent_distance(encodings, a, g, squared=True),
        "linear regression on concat": lambda a, g: concat_features(encodings, a, g),
    }

    for size in args.train_pairs:
        n = slice(0, size)
        agent_n, goal_n, label_n = train_agent[n], train_goal[n], train_label[n]

        mean = label_n.mean()
        fitted.append(("baseline: mean", size, 0, lambda p, mean=mean: np.full(len(p["label"]), mean)))

        for name, features in latent_features.items():
            predict = fit_linear_in_chunks(features, agent_n, goal_n, label_n)
            fitted.append((name, size, 0,
                           lambda p, predict=predict, features=features: predict(features(p["agent"], p["goal"]))))

        for kind in args.models:
            for seed in range(args.seeds):
                predict = train_pair_model(
                    encodings,
                    (agent_n, goal_n, label_n),
                    (evaluation["val"]["agent"], evaluation["val"]["goal"], evaluation["val"]["label"]),
                    device, seed, kind=kind,
                )
                fitted.append((MODEL_NAMES[kind], size, seed, lambda p, predict=predict: predict(p["agent"], p["goal"])))
                print(f"Trained {kind}: {size} pairs, seed {seed}")

    results = pd.DataFrame([
        score(args.label, name, size, seed, split, evaluation[split], predict(evaluation[split]))
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

import argparse
import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from scipy.sparse.csgraph import dijkstra

from scripts.run_label_experiment import make_labeler
from scripts.run_planning_eval import sample_tasks
from utils.heuristic_models import fit_linear_in_chunks, latent_distance, train_pair_model
from utils.pair_dataset import load_position_dataset, sample_pairs
from utils.reference_distance import build_grid_graph
from utils.search import PLANNERS, make_adjacency, search

OPTIMAL_TOLERANCE = 1e-6


def build_heuristics(positions, encodings, splits, device, label, train_pairs, seed):
    """
    Returns {name: function(goal index) -> heuristic value of every node (px)}.
    Everything learned is fitted on training positions only; tasks come from test positions.
    """
    everyone = np.arange(len(positions))
    labeler = make_labeler(label, positions)
    train_agent, train_goal = sample_pairs(splits, "train", train_pairs, seed)
    train_label = labeler(train_agent, train_goal)
    val_agent, val_goal = sample_pairs(splits, "val", 50_000, seed)

    # the latent distance is not in pixels, so it is fitted to the label first (a straight line)
    lines = {
        squared: fit_linear_in_chunks(lambda a, g, squared=squared: latent_distance(encodings, a, g, squared),
                                      train_agent[:200_000], train_goal[:200_000], train_label[:200_000])
        for squared in (False, True)
    }
    mlp = train_pair_model(encodings, (train_agent, train_goal, train_label),
                           (val_agent, val_goal, labeler(val_agent, val_goal)), device, seed, kind="mlp")
    graph = build_grid_graph(positions)

    def latent(squared):
        return lambda goal: lines[squared](latent_distance(encodings, everyone, np.full(len(everyone), goal), squared))

    return {
        "none (all zero)": lambda goal: np.zeros(len(positions)),
        "Euclidean": lambda goal: np.linalg.norm(positions - positions[goal], axis=1),
        "latent distance (fitted)": latent(False),
        "squared latent distance, JEPA cost (fitted)": latent(True),
        "learned MLP": lambda goal: mlp(everyone, np.full(len(everyone), goal)),
        "perfect (grid distance)": lambda goal: dijkstra(graph, directed=False, indices=[goal])[0],
    }, graph


def main():
    parser = argparse.ArgumentParser(description="DFS, greedy best-first and A* with different heuristics on the position grid")
    parser.add_argument("--dataset", type=str, default=None, help="Position dataset .npz. Defaults to the newest in data/")
    parser.add_argument("--episodes", type=int, default=100, help="Tasks per task kind")
    parser.add_argument("--heuristic-label", choices=("grid", "formula"), default="formula")
    parser.add_argument("--heuristic-pairs", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = args.dataset or sorted(glob.glob("data/*_position_dataset_*.npz"))[-1]
    positions, encodings, splits = load_position_dataset(path)
    tasks = sample_tasks(positions, splits, args.episodes, args.seed)
    print({kind: len(items) for kind, items in tasks.items()})

    heuristics, graph = build_heuristics(positions, encodings, splits, device, args.heuristic_label, args.heuristic_pairs, args.seed)
    adjacency = make_adjacency(graph)
    print("Heuristics ready")

    rows = []
    for kind, items in tasks.items():
        for index, (_, _, _, start, goal) in enumerate(items):
            optimal = dijkstra(graph, directed=False, indices=[start])[0][goal]
            tables = {name: make(goal) for name, make in heuristics.items()}

            runs = [("dfs", "none (all zero)")] + [(planner, name) for planner in ("greedy", "astar") for name in tables]
            for planner, name in runs:
                result = search(adjacency, start, goal, tables[name], planner)
                rows.append({"kind": kind, "task": index, "planner": planner, "heuristic": name,
                             "optimal_cost": optimal, **result})
        print(f"done: {kind}")

    table = pd.DataFrame(rows)
    table["suboptimality"] = table["cost"] / table["optimal_cost"]
    table["optimal"] = table["suboptimality"] <= 1 + OPTIMAL_TOLERANCE
    os.makedirs("data/results", exist_ok=True)
    out = f"data/results/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_search_eval.csv"
    table.to_csv(out, index=False)

    summary = table.groupby(["kind", "planner", "heuristic"]).agg(
        found=("found", "mean"), expansions=("expansions", "median"), expansions_mean=("expansions", "mean"),
        suboptimality=("suboptimality", "mean"), optimal=("optimal", "mean"))
    print(summary.round(3).to_string())
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

import argparse
import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from scipy.sparse.csgraph import dijkstra

from scripts.run_planning_eval import sample_tasks
from scripts.run_search_eval import build_heuristics
from utils.pair_dataset import load_position_dataset
from utils.search import make_adjacency, search

KEEP = ("Euclidean", "learned MLP", "perfect (grid distance)")


def main():
    parser = argparse.ArgumentParser(description="Weighted A* and admissibility of learned heuristics on the position grid")
    parser.add_argument("--dataset", type=str, default=None, help="Position dataset .npz. Defaults to the newest in data/")
    parser.add_argument("--episodes", type=int, default=100, help="Tasks per task kind")
    parser.add_argument("--weights", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0, 5.0])
    parser.add_argument("--heuristic-pairs", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = args.dataset or sorted(glob.glob("data/*_position_dataset_*.npz"))[-1]
    positions, encodings, splits = load_position_dataset(path)
    tasks = sample_tasks(positions, splits, args.episodes, args.seed)

    # the same MLP trained on two labels: the continuous door formula (shorter than the grid) and the grid itself
    heuristics = {}
    for label in ("formula", "grid"):
        made, graph = build_heuristics(positions, encodings, splits, device, label, args.heuristic_pairs, args.seed)
        for name in KEEP:
            if name == "learned MLP":
                heuristics[f"MLP trained on {label}"] = made[name]
            else:
                heuristics[name] = made[name]
    adjacency = make_adjacency(graph)
    print("Heuristics ready")

    rows = []
    for kind, items in tasks.items():
        for index, (_, _, _, start, goal) in enumerate(items):
            true_distance = dijkstra(graph, directed=False, indices=[goal])[0]  # exact distance of every node to the goal
            optimal = true_distance[start]
            for name, make in heuristics.items():
                table = make(goal)
                over = table - true_distance
                stats = {"over_share_strict": float((over > 1e-6).mean()), "over_share_2px": float((over > 2).mean()),
                         "max_over": float(over.max())}
                for weight in args.weights:
                    result = search(adjacency, start, goal, table, "astar", weight=weight)
                    rows.append({"kind": kind, "task": index, "heuristic": name, "weight": weight, "optimal_cost": optimal,
                                 **stats, **result})
        print(f"done: {kind}")

    table = pd.DataFrame(rows)
    table["suboptimality"] = table["cost"] / table["optimal_cost"]
    table["optimal"] = table["suboptimality"] <= 1 + 1e-6
    table["within_bound"] = table["suboptimality"] <= table["weight"] + 1e-6
    os.makedirs("data/results", exist_ok=True)
    out = f"data/results/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_weighted_astar.csv"
    table.to_csv(out, index=False)

    summary = table.groupby(["kind", "heuristic", "weight"]).agg(
        expansions=("expansions", "median"), suboptimality=("suboptimality", "mean"), worst=("suboptimality", "max"),
        optimal=("optimal", "mean"), within_bound=("within_bound", "mean"))
    print(summary.round(3).to_string())
    over = table[table.weight == args.weights[0]].groupby(["kind", "heuristic"]).agg(
        over_strict=("over_share_strict", "mean"), over_2px=("over_share_2px", "mean"), max_over=("max_over", "max"))
    print(over.round(3).to_string())
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

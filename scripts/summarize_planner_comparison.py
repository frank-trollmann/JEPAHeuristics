import argparse
import glob

import numpy as np
import pandas as pd


def wilson(successes, n, z=1.96):
    """95% Wilson interval for a success rate; returns (low, high)."""
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def load(pattern):
    """All result CSVs matching pattern, with the file tag (which run) and the seed (from the tag or 2)."""
    frames = []
    for path in sorted(glob.glob(pattern)):
        frame = pd.read_csv(path)
        frame["run"] = path.split("planner_comparison")[-1].removesuffix(".csv").strip("_") or "main"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Success, interval, compute per (run, planner, heuristic, kind) from planner-comparison CSVs")
    parser.add_argument("--pattern", default="data/results/*_planner_comparison_*.csv")
    parser.add_argument("--exclude", nargs="*", default=["pilot", "smoke"], help="Runs whose tag contains one of these are skipped")
    args = parser.parse_args()

    table = load(args.pattern)
    table = table[~table["run"].apply(lambda run: any(word in run for word in args.exclude))]
    rows = []
    for (run, planner, heuristic, execute_blocks, kind), group in table.groupby(["run", "planner", "heuristic", "execute_blocks", "kind"]):
        wins, n = int(group["success"].sum()), len(group)
        low, high = wilson(wins, n)
        rows.append({"run": run, "planner": planner, "heuristic": heuristic, "exec": execute_blocks, "kind": kind, "n": n,
                     "success": wins / n, "ci_low": low, "ci_high": high, "steps": group["steps"].mean(),
                     "transitions_k": group["transitions"].mean() / 1000, "plan_s": group["plan_seconds"].mean()})
    summary = pd.DataFrame(rows)
    print(summary.round(2).to_string(index=False))
    summary.to_csv("data/results/planner_comparison_summary.csv", index=False)


if __name__ == "__main__":
    main()

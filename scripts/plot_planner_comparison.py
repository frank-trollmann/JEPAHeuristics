import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.summarize_planner_comparison import wilson

# categorical slots 1-4 of the reference palette (light mode), marker shape as second encoding
FAMILIES = {
    "search": ("#2a78d6", "o", ["greedy", "astar", "wastar"], "search_5blocks"),
    "iCEM": ("#eb6834", "s", ["icem"], "icem"),
    "CEM, 8 directions": ("#1baf7a", "^", ["dcem"], "C_snapped_cem"),
    "MPPI": ("#eda100", "D", ["mppi"], "mppi"),
    "beam": ("#6f42c1", "P", ["beam"], "M1_beam_mcts"),
    "MCTS": ("#c2185b", "v", ["mcts"], "M1_beam_mcts"),
    "gradient descent": ("#7a6a4f", "<", ["gd"], "M7_gd_far"),
    "random": ("#8a8a85", ">", ["random"], "M8_random_far"),
}
LABEL_OFFSET = {"beam": (10, -4), "MCTS": (8, 9), "gradient descent": (8, 9), "random": (8, 9)}
CEM_EARLIER = {"jepa": 0.02, "learned": 0.48}  # CEM1 / CEM2 on the same tasks, about 270k calls (research log)
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def main():
    parser = argparse.ArgumentParser(description="Success vs predictor calls on the far wall-crossing tasks, per heuristic")
    parser.add_argument("--summary", default="data/results/planner_comparison_summary.csv")
    parser.add_argument("--out", default="data/results/planner_comparison_far.png")
    args = parser.parse_args()

    table = pd.read_csv(args.summary)
    table = table[table["kind"] == "cross_far"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, facecolor=SURFACE)
    for ax, heuristic, title in zip(axes, ("jepa", "learned"), ("JEPA's own cost", "Learned heuristic")):
        ax.set_facecolor(SURFACE)
        for name, (color, marker, planners, run) in FAMILIES.items():
            rows = table[(table["run"] == run) & table["planner"].isin(planners) & (table["heuristic"] == heuristic)]
            if rows.empty:
                continue
            x, y = rows["transitions_k"] * 1000, rows["success"]
            ax.errorbar(x, y, yerr=[y - rows["ci_low"], rows["ci_high"] - y], fmt="none", ecolor=color, alpha=0.5, lw=1.2, capsize=0)
            ax.scatter(x, y, s=60, color=color, marker=marker, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            if name in ("search", "beam", "MCTS", "gradient descent") or heuristic == "learned":  # with the JEPA cost the failing samplers overlap: one shared label
                ax.annotate(name, (x.mean(), y.mean()), xytext=LABEL_OFFSET.get(name, (-6, -18) if name == "iCEM" else (8, 9)), textcoords="offset points", fontsize=9.5, color=INK)
        ax.scatter([270_000], [CEM_EARLIER[heuristic]], s=60, color=MUTED, marker="x", linewidth=2, zorder=3)
        if heuristic == "learned":
            ax.annotate("CEM", (270_000, CEM_EARLIER[heuristic]), xytext=(-8, 9), textcoords="offset points", fontsize=9.5, color=MUTED, ha="right")
        else:
            ax.annotate("CEM, MPPI, iCEM, CEM (8 dir.), random:\nall between 0.00 and 0.06", (265_000, 0.06), xytext=(-14, 34),
                        textcoords="offset points", fontsize=9.5, color=INK, ha="right",
                        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
        ax.set_xscale("log")
        ax.set_xlim(8e2, 6e5)
        ax.set_ylim(-0.04, 1.06)
        ax.set_title(title, fontsize=11.5, color=INK, loc="left")
        ax.set_xlabel("predictor calls per episode (log scale)", color=MUTED, fontsize=9.5)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=9)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[0].set_ylabel("success on far wall-crossing tasks", color=MUTED, fontsize=9.5)
    fig.suptitle("Search through the predictor reaches ~95% with either cost; sampling planners need the learned heuristic",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.01, "50 episodes per point, 95% Wilson intervals; one map, one checkpoint. Search = greedy, A*, weighted A* (three points). "
             "CEM = earlier runs, calls approximate.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(args.out, dpi=170, facecolor=SURFACE)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

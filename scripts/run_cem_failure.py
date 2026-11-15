import argparse
import glob
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from stable_worldmodel.envs.two_room import TwoRoomEnv

import utils.load_fix_pretrained as load_util
from scripts.run_label_experiment import make_labeler
from scripts.run_planning_eval import sample_tasks
from utils.cem_planner import run_episode
from utils.heuristic_models import train_pair_model
from utils.pair_dataset import load_position_dataset, sample_pairs

WALL_X = 112  # centre of the wall; the door is centred at y = 49

# name: settings of run_episode that differ from the LeWM evaluation (horizon 5, execute 5 blocks, 300 samples, 150 steps)
CONFIGS = {
    "baseline": {},
    "replan_every_block": {"execute_blocks": 1},
    "replan_every_2_blocks": {"execute_blocks": 2},
    "horizon_8": {"horizon": 8, "execute_blocks": 5},
    "horizon_12": {"horizon": 12, "execute_blocks": 5},
    "samples_1000": {"num_samples": 1000},
    "budget_300": {"budget_steps": 300},
}


def describe_failure(path, start):
    """Where did the episode end: which room, and did the agent ever cross the wall line?"""
    path = np.asarray(path)
    side = lambda x: "left" if x < WALL_X else "right"
    crossed = bool((np.sign(path[:, 0] - WALL_X) != np.sign(start[0] - WALL_X)).any())
    final = path[-1]
    near_wall = abs(final[0] - WALL_X) < 25
    return crossed, side(final[0]), near_wall, float(final[1])


def main():
    parser = argparse.ArgumentParser(description="Why do far wall-crossing tasks fail with a learned heuristic as the CEM cost?")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--action-stats", type=str, default="data/action_stats.npz", help="mean and std of the LeWM dataset actions (made from tworoom.h5, see the research log)")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=["baseline"])
    parser.add_argument("--kind", choices=("same_room", "cross_near", "cross_far"), default="cross_far")
    parser.add_argument("--heuristic-pairs", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=2, help="2 reproduces the tasks and the heuristic of experiment CEM1")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or snapshot_download(repo_id="quentinll/lewm-tworooms", revision="77adaae0bc31deab21c93740d1f8bb947cd0bdec")
    model = load_util.load_fix_pretrained(checkpoint).to(device).eval()
    stats = np.load(args.action_stats)
    action_mean, action_std = stats["mean"], stats["std"]

    path = args.dataset or sorted(glob.glob("data/*_position_dataset_*.npz"))[-1]
    positions, encodings, splits = load_position_dataset(path)
    tasks = sample_tasks(positions, splits, args.episodes, args.seed)[args.kind]

    labeler = make_labeler("formula", positions)
    train_agent, train_goal = sample_pairs(splits, "train", args.heuristic_pairs, args.seed)
    val_agent, val_goal = sample_pairs(splits, "val", 50_000, args.seed)
    heuristic = train_pair_model(encodings, (train_agent, train_goal, labeler(train_agent, train_goal)),
                                 (val_agent, val_goal, labeler(val_agent, val_goal)), device, seed=args.seed, kind="mlp")
    cost = lambda goal, xy: (lambda final: heuristic.on_latents(final, goal.expand_as(final)))
    print("Trained the learned heuristic", flush=True)

    env = TwoRoomEnv()
    env.reset()
    rows = []
    for name in args.configs:
        settings = {"budget_steps": 150, **CONFIGS[name]}
        started = time.time()
        for index, (start, goal, path_length, *_) in enumerate(tasks):
            result = run_episode(env, model, device, start, goal, cost, action_mean, action_std,
                                 seed=args.seed * 1000 + index, trace=True, **settings)
            crossed, final_side, near_wall, final_y = describe_failure(result["path"], start)
            rows.append({"config": name, "kind": args.kind, "episode": index, "start": start, "goal": goal, "path_length": path_length,
                         "crossed_wall_line": crossed, "final_side": final_side, "final_near_wall": near_wall, "final_y": final_y,
                         **{k: v for k, v in result.items() if k != "path"}, "path": json.dumps(result["path"])})
        done = pd.DataFrame([r for r in rows if r["config"] == name])
        print(f"{name:22s} success {done['success'].mean():.2f} ({len(done)} episodes, {time.time() - started:.0f} s)", flush=True)

    table = pd.DataFrame(rows)
    os.makedirs("data/results", exist_ok=True)
    out = f"data/results/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_cem_failure.csv"
    table.to_csv(out, index=False)
    summary = table.groupby("config").agg(success=("success", "mean"), steps=("steps", "mean"), crossed=("crossed_wall_line", "mean"),
                                          closest=("closest_distance", "mean"), episodes=("success", "size"))
    print(summary.round(3).to_string())
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

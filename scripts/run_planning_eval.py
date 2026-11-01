import argparse
import glob
import os
import time
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from scipy.spatial import cKDTree
from stable_worldmodel.envs.two_room import TwoRoomEnv

import utils.load_fix_pretrained as load_util
from scripts.run_label_experiment import make_labeler
from utils.cem_planner import jepa_cost, run_episode
from utils.dynamics_probe import action_statistics
from utils.heuristic_models import train_pair_model
from utils.pair_dataset import euclidean_labels, load_position_dataset, sample_pairs
from utils.position_probe import fit_position_probe
from utils.trajectory_dataset import train_temporal_heuristic
from utils.reference_distance import door_geometry, formula_labels

TASK_KINDS = ("same_room", "cross_near", "cross_far")
MIN_START_DISTANCE = 60  # px; the goal test is 16 px, so shorter tasks are almost solved at the start


def sample_tasks(positions, splits, per_kind, seed):
    """
    Start and goal coordinates from test-split positions, in three kinds:
    same_room: both in the same room;
    cross_near: across the wall, and the shortest path is at most 15% longer than the straight line;
    cross_far: across the wall, and the shortest path is at least 50% longer (the wall is in the way).
    Returns {kind: list of (start, goal, path length, start index, goal index)}.
    """
    env = TwoRoomEnv()
    env.reset()
    geometry = door_geometry(env, inflated=True)

    agent, goal = sample_pairs(splits, "test", 400_000, seed)
    straight = euclidean_labels(positions, agent, goal)
    path = formula_labels(positions[agent].astype(float), positions[goal].astype(float), geometry)
    left_of_wall = lambda index: positions[index, 0] < geometry["nodes"][:, 0].min()
    right_of_wall = lambda index: positions[index, 0] > geometry["nodes"][:, 0].max()
    same_room = (left_of_wall(agent) & left_of_wall(goal)) | (right_of_wall(agent) & right_of_wall(goal))
    across = (left_of_wall(agent) & right_of_wall(goal)) | (right_of_wall(agent) & left_of_wall(goal))
    far_enough = straight >= MIN_START_DISTANCE

    masks = {
        "same_room": same_room & far_enough,
        "cross_near": across & far_enough & (path <= 1.15 * straight),
        "cross_far": across & far_enough & (path >= 1.5 * straight),
    }
    tasks = {}
    for kind, mask in masks.items():
        chosen = np.flatnonzero(mask)[:per_kind]
        tasks[kind] = [(tuple(positions[agent[i]]), tuple(positions[goal[i]]), float(path[i]), int(agent[i]), int(goal[i]))
                       for i in chosen]
    return tasks


def sample_paper_style_tasks(h5_path, count, seed, offset_steps=25):
    """
    Tasks like the LeWM evaluation (goal_offset_steps 25): the start is a frame of a dataset trajectory
    and the goal is the agent position offset_steps frames later in the same trajectory.
    """
    rng = np.random.default_rng(seed)
    with h5py.File(h5_path) as f:
        episode_offset, episode_length = f["ep_offset"][:], f["ep_len"][:]
        tasks = []
        for episode in rng.choice(len(episode_length), count, replace=False):
            t = int(rng.integers(0, episode_length[episode] - offset_steps))
            first = int(episode_offset[episode]) + t
            start, goal = f["pos_agent"][first], f["pos_agent"][first + offset_steps]
            tasks.append((tuple(float(v) for v in start), tuple(float(v) for v in goal), float(np.linalg.norm(start - goal))))
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Compare CEM planning costs on same-room and across-the-wall tasks")
    parser.add_argument("--dataset", type=str, default=None, help="Position dataset .npz. Defaults to the newest in data/")
    parser.add_argument("--checkpoint", type=str, default=None, help="Folder with config.json and weights.pt. Defaults to the pinned HuggingFace checkpoint.")
    parser.add_argument("--actions-h5", type=str, default="stable_worldmodel/datasets/tworoom/tworoom.h5", help="LeWM dataset, only used for the action mean and std")
    parser.add_argument("--episodes", type=int, default=60, help="Episodes per task kind and method")
    parser.add_argument("--budget-steps", type=int, default=150, help="Environment steps per episode")
    parser.add_argument("--heuristic-label", choices=("euclidean", "grid", "formula"), default="formula", help="Label the learned heuristic is trained on")
    parser.add_argument("--heuristic-pairs", type=int, default=1_000_000)
    parser.add_argument("--methods", nargs="+", choices=("jepa", "learned", "decode_formula", "temporal"), default=["jepa", "learned"])
    parser.add_argument("--paper-tasks", type=int, default=0, help="Also run this many tasks like the LeWM evaluation (goal 25 steps ahead, 50-step budget)")
    parser.add_argument("--only-paper-tasks", action="store_true", help="Run only the LeWM-style tasks (to check this CEM against the published result)")
    parser.add_argument("--trajectory-dataset", type=str, default=None, help="For the temporal method: data/*_trajectory_dataset_*.npz (newest by default)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or snapshot_download(
        repo_id="quentinll/lewm-tworooms",
        revision="77adaae0bc31deab21c93740d1f8bb947cd0bdec",
    )
    model = load_util.load_fix_pretrained(checkpoint).to(device).eval()
    with h5py.File(args.actions_h5) as f:
        action_mean, action_std = action_statistics(f["action"][:])

    path = args.dataset or sorted(glob.glob("data/*_position_dataset_*.npz"))[-1]
    positions, encodings, splits = load_position_dataset(path)
    tasks = {} if args.only_paper_tasks else sample_tasks(positions, splits, args.episodes, args.seed)
    budgets = {kind: args.budget_steps for kind in tasks}
    if args.paper_tasks or args.only_paper_tasks:
        tasks["paper_25"] = sample_paper_style_tasks(args.actions_h5, args.paper_tasks or args.episodes, args.seed)
        budgets["paper_25"] = 50
    print({kind: len(items) for kind, items in tasks.items()})

    heuristic = None
    if "learned" in args.methods:
        # the learned heuristic is trained on training positions only; tasks come from test positions
        labeler = make_labeler(args.heuristic_label, positions)
        train_agent, train_goal = sample_pairs(splits, "train", args.heuristic_pairs, args.seed)
        val_agent, val_goal = sample_pairs(splits, "val", 50_000, args.seed)
        heuristic = train_pair_model(
            encodings,
            (train_agent, train_goal, labeler(train_agent, train_goal)),
            (val_agent, val_goal, labeler(val_agent, val_goal)),
            device, seed=args.seed, kind="mlp",
        )
        print("Trained the learned heuristic")

    temporal = None
    if "temporal" in args.methods:
        # trained on the order of frames in the original trajectories only: no coordinates, no wall geometry
        trajectory = dict(np.load(args.trajectory_dataset or sorted(glob.glob("data/*_trajectory_dataset_*.npz"))[-1]))
        temporal, _ = train_temporal_heuristic(trajectory, device, args.seed)
        print("Trained the temporal heuristic")

    decoder = None
    if "decode_formula" in args.methods:
        # decode (x, y) from a latent, snap it to the nearest valid position, and use the formula distance to the goal
        decoder = fit_position_probe(encodings[splits == "train"], positions[splits == "train"], hidden=(256, 256))
        valid = cKDTree(positions)
        env_for_geometry = TwoRoomEnv()
        env_for_geometry.reset()
        geometry = door_geometry(env_for_geometry, inflated=True)

    def decode_formula_cost(goal_latent, goal_xy):
        goal_xy = np.asarray(goal_xy, dtype=float)[None]

        def cost(final):
            decoded = decoder.predict(final.cpu().numpy())
            snapped = positions[valid.query(decoded)[1]].astype(float)
            distance = formula_labels(snapped, np.repeat(goal_xy, len(snapped), axis=0), geometry)
            return torch.tensor(distance, dtype=torch.float32, device=final.device)

        return cost

    costs = {
        "jepa": lambda goal, xy: (lambda final: jepa_cost(final, goal)),
        "learned": lambda goal, xy: (lambda final: heuristic.on_latents(final, goal.expand_as(final))),
        "decode_formula": decode_formula_cost,
        "temporal": lambda goal, xy: (lambda final: temporal.on_latents(final, goal.expand_as(final))),
    }

    env = TwoRoomEnv()
    env.reset()
    rows = []
    for method in args.methods:
        for kind, items in tasks.items():
            started = time.time()
            for index, (start, goal, path_length, *_) in enumerate(items):
                result = run_episode(env, model, device, start, goal, costs[method], action_mean, action_std,
                                     budgets[kind], seed=args.seed * 1000 + index)
                rows.append({"method": method, "heuristic_label": args.heuristic_label if method == "learned" else "", "kind": kind, "episode": index, "start": start, "goal": goal,
                             "path_length": path_length, **result})
            done = pd.DataFrame([r for r in rows if r["method"] == method and r["kind"] == kind])
            print(f"{method:8s} {kind:11s} success {done['success'].mean():.2f} "
                  f"({len(done)} episodes, {time.time() - started:.0f} s)")

    table = pd.DataFrame(rows)
    os.makedirs("data/results", exist_ok=True)
    out = f"data/results/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_planning_eval.csv"
    table.to_csv(out, index=False)
    summary = table.groupby(["method", "kind"]).agg(
        success=("success", "mean"), steps=("steps", "mean"),
        closest=("closest_distance", "mean"), episodes=("success", "size"))
    print(summary.round(3).to_string())
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

import argparse
import glob
import os
import time
from datetime import datetime
from functools import partial

import h5py
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from stable_worldmodel.envs.two_room import TwoRoomEnv

import utils.load_fix_pretrained as load_util
from scripts.run_label_experiment import make_labeler
from scripts.run_planning_eval import TASK_KINDS, sample_tasks
from utils.cem_planner import ACTION_DIM, plan, run_episode
from utils.dynamics_probe import BLOCK, action_statistics
from utils.heuristic_models import train_pair_model
from utils.latent_search import make_beam_planner, make_mcts_planner, make_search_planner
from utils.pair_dataset import load_position_dataset, sample_pairs
from utils.sampling_planners import make_snapped_cem, plan_gradient, plan_icem, plan_mppi, plan_random
from utils.trajectory_dataset import train_temporal_heuristic

SAMPLING = ("cem", "mppi", "icem", "dcem", "random", "gd")  # dcem = CEM snapped to the 8 search directions; gd needs a differentiable cost (jepa, learned)
SEARCH = ("greedy", "astar", "wastar", "beam", "mcts")  # wastar = A* with weight 2; mcts uses --expansions as its simulations
WEIGHTED = 2.0


def fit_to_pixels(heuristic, encodings, positions, splits, labeler, device, seed):
    """
    The search planners add a cost in pixels to the heuristic, so a heuristic that is not in pixels (the JEPA
    cost, the frame-order heuristic) is first fitted to the wall-aware distance with a straight line, on
    validation-split pairs. Returns h(latents, goal) -> pixels.
    """
    agent, goal = sample_pairs(splits, "val", 50_000, seed)
    z = torch.tensor(encodings, device=device)
    values = heuristic(z[agent], z[goal]).cpu().numpy()
    slope, intercept = np.polyfit(values, labeler(agent, goal), 1)
    return lambda latents, goal_latent: (slope * heuristic(latents, goal_latent) + intercept).clamp(min=0.0)


def main():
    parser = argparse.ArgumentParser(description="Compare planners (CEM, MPPI, iCEM, greedy / A* search through the predictor) under the same heuristics")
    parser.add_argument("--planners", nargs="+", choices=SAMPLING + SEARCH, required=True)
    parser.add_argument("--heuristics", nargs="+", choices=("jepa", "learned", "temporal"), default=["learned"])
    parser.add_argument("--kinds", nargs="+", choices=TASK_KINDS, default=list(TASK_KINDS))
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per task kind")
    parser.add_argument("--budget-steps", type=int, default=150)
    parser.add_argument("--execute-blocks", type=int, default=5, help="Blocks executed before replanning (5 = LeWM evaluation)")
    parser.add_argument("--expansions", type=int, default=100, help="Search: nodes expanded per plan")
    parser.add_argument("--max-depth", type=int, default=12, help="Search: longest path in blocks")
    parser.add_argument("--beam-width", type=int, default=5, help="Beam search: nodes kept per level")
    parser.add_argument("--mcts-c", type=float, default=0.7, help="MCTS: exploration constant")
    parser.add_argument("--merge-distance", type=float, default=12.0, help="Search: latents closer than this are duplicates")
    parser.add_argument("--stop-at", type=float, default=10.0, help="Search: stop when the estimated distance is below this (px)")
    parser.add_argument("--mppi-temperature", type=float, default=0.5, help="MPPI temperature in units of the elite cost spread")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--actions-h5", type=str, default="stable_worldmodel/datasets/tworoom/tworoom.h5")
    parser.add_argument("--trajectory-dataset", type=str, default=None)
    parser.add_argument("--heuristic-pairs", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=2, help="2 = the tasks and heuristics of CEM1 / CEM2")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or snapshot_download(
        repo_id="quentinll/lewm-tworooms",
        revision="77adaae0bc31deab21c93740d1f8bb947cd0bdec",
    )
    model = load_util.load_fix_pretrained(checkpoint).to(device).eval()
    if os.path.exists(args.actions_h5):
        with h5py.File(args.actions_h5) as f:
            action_mean, action_std = action_statistics(f["action"][:])
    else:  # the 12 GB dataset is not on every machine
        saved = np.load("data/action_stats.npz")
        action_mean, action_std = saved["mean"], saved["std"]

    transitions = {"count": 0}
    predict = model.predict

    def counting_predict(emb, act_emb):
        transitions["count"] += emb.shape[0]
        return predict(emb, act_emb)

    model.predict = counting_predict

    path = args.dataset or sorted(glob.glob("data/*_position_dataset_*.npz"))[-1]
    positions, encodings, splits = load_position_dataset(path)
    all_tasks = sample_tasks(positions, splits, args.episodes, args.seed)
    tasks = {kind: all_tasks[kind] for kind in args.kinds}
    print({kind: len(items) for kind, items in tasks.items()})

    # heuristics h(latents, goal_latent); lower is closer
    heuristics = {"jepa": lambda a, g: ((a - g) ** 2).sum(dim=1)}
    differentiable = {"jepa": heuristics["jepa"]}  # for gradient descent; the learned model's on_latents runs under no_grad
    if "learned" in args.heuristics:
        labeler = make_labeler("formula", positions)
        train_agent, train_goal = sample_pairs(splits, "train", args.heuristic_pairs, args.seed)
        val_agent, val_goal = sample_pairs(splits, "val", 50_000, args.seed)
        learned = train_pair_model(
            encodings, (train_agent, train_goal, labeler(train_agent, train_goal)),
            (val_agent, val_goal, labeler(val_agent, val_goal)), device, seed=args.seed, kind="mlp")
        heuristics["learned"] = learned.on_latents
        standardise = lambda z: (z - learned.encodings_mean) / learned.encodings_std
        differentiable["learned"] = lambda a, g: learned.model(standardise(a), standardise(g)) * learned.y_std + learned.y_mean
        print("Trained the learned heuristic")
    if "temporal" in args.heuristics:
        trajectory = dict(np.load(args.trajectory_dataset or sorted(glob.glob("data/*_trajectory_dataset_*.npz"))[-1]))
        temporal, _ = train_temporal_heuristic(trajectory, device, args.seed)
        heuristics["temporal"] = temporal.on_latents
        print("Trained the temporal heuristic")
    labeler = make_labeler("formula", positions)
    in_pixels = {"learned": heuristics.get("learned")}
    for name in ("jepa", "temporal"):
        if name in args.heuristics and any(p in SEARCH for p in args.planners):
            in_pixels[name] = fit_to_pixels(heuristics[name], encodings, positions, splits, labeler, device, args.seed)

    low = torch.tensor(np.tile((-1.0 - action_mean) / action_std, BLOCK), dtype=torch.float32, device=device)
    high = torch.tensor(np.tile((1.0 - action_mean) / action_std, BLOCK), dtype=torch.float32, device=device)
    assert low.shape == (ACTION_DIM,)

    def build(planner):
        if planner == "cem":
            return plan
        if planner == "mppi":
            return partial(plan_mppi, temperature=args.mppi_temperature)
        if planner == "dcem":
            return make_snapped_cem(action_mean, action_std)
        if planner == "random":
            return plan_random
        if planner == "gd":
            return plan_gradient
        if planner == "beam":
            return make_beam_planner(action_mean, action_std, args.beam_width, args.max_depth, args.merge_distance, args.stop_at)
        if planner == "mcts":
            return make_mcts_planner(action_mean, action_std, args.expansions, args.mcts_c, args.max_depth, args.stop_at)
        if planner == "icem":
            return partial(plan_icem, bounds=(low, high))
        kind, weight = ("astar", WEIGHTED) if planner == "wastar" else (planner, 1.0)
        return make_search_planner(kind, action_mean, action_std, weight, args.expansions, args.max_depth,
                                   args.merge_distance, args.stop_at)

    env = TwoRoomEnv()
    env.reset()
    os.makedirs("data/results", exist_ok=True)
    out = f"data/results/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_planner_comparison{args.tag}.csv"
    rows = []
    for planner_name in args.planners:
        planner = build(planner_name)
        for heuristic_name in args.heuristics:
            if planner_name == "gd" and heuristic_name not in differentiable:
                print(f"gd skipped for {heuristic_name}: no differentiable version")
                continue
            h = in_pixels[heuristic_name] if planner_name in SEARCH else (differentiable if planner_name == "gd" else heuristics)[heuristic_name]
            make_cost = lambda goal, xy, h=h: (lambda final: h(final, goal.expand_as(final)))
            for kind, items in tasks.items():
                started = time.time()
                for index, (start, goal, path_length, *_) in enumerate(items):
                    transitions["count"] = 0
                    result = run_episode(env, model, device, start, goal, make_cost, action_mean, action_std,
                                         args.budget_steps, seed=args.seed * 1000 + index,
                                         execute_blocks=args.execute_blocks, planner=planner)
                    rows.append({"planner": planner_name, "heuristic": heuristic_name, "kind": kind, "episode": index,
                                 "start": start, "goal": goal, "path_length": path_length,
                                 "execute_blocks": args.execute_blocks, "transitions": transitions["count"], **result})
                done = pd.DataFrame([r for r in rows if (r["planner"], r["heuristic"], r["kind"]) == (planner_name, heuristic_name, kind)])
                print(f"{planner_name:7s} {heuristic_name:9s} {kind:11s} success {done['success'].mean():.2f} "
                      f"({len(done)} episodes, {time.time() - started:.0f} s, {done['transitions'].mean():,.0f} transitions/episode)", flush=True)
                pd.DataFrame(rows).to_csv(out, index=False)

    table = pd.DataFrame(rows)
    summary = table.groupby(["planner", "heuristic", "kind"]).agg(
        success=("success", "mean"), steps=("steps", "mean"), closest=("closest_distance", "mean"),
        plan_seconds=("plan_seconds", "mean"), transitions=("transitions", "mean"), episodes=("success", "size"))
    print(summary.round(3).to_string())
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

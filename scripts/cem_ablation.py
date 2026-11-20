"""CEM ablation: batch_size, generations, elite_fraction across horizons."""
from __future__ import annotations
import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch

from utils.cem_path_sampler import CEMPathSampler
from utils.heuristic_planner import load_pretrained_mlp
from utils.image_preprocessing import normalise_images
from stable_worldmodel.envs.two_room import TwoRoomEnv


@dataclass
class AblationConfig:
    batch_size: int
    num_generations: int
    elite_fraction: float


def generate_problem(rng: np.random.Generator) -> tuple[list, list, float]:
    env = TwoRoomEnv()
    agent_pos = rng.integers(10, 200, size=2).tolist()
    goal_pos = rng.integers(10, 200, size=2).tolist()
    while abs(agent_pos[0] - goal_pos[0]) < 20 and abs(agent_pos[1] - goal_pos[1]) < 20:
        goal_pos = rng.integers(10, 200, size=2).tolist()
    env.reset(options={"state": agent_pos})
    initial_dist = float(torch.norm(torch.tensor(agent_pos, dtype=torch.float) - torch.tensor(goal_pos, dtype=torch.float)).item())
    return agent_pos, goal_pos, initial_dist


def make_images(agent_pos, goal_pos, device):
    env = TwoRoomEnv()
    env.reset(options={"state": agent_pos})
    img_a = env.render()
    t_img_a = normalise_images(torch.from_numpy(img_a).permute(2, 0, 1).float().unsqueeze(0)).squeeze(0).to(device)
    env.reset(options={"state": goal_pos})
    img_g = env.render()
    t_img_g = normalise_images(torch.from_numpy(img_g).permute(2, 0, 1).float().unsqueeze(0)).squeeze(0).to(device)
    return t_img_a, t_img_g


def run_cem(model, t_img_a, t_img_g, horizon, device, rng, cfg: AblationConfig):
    sampler = CEMPathSampler(
        model=model, history_size=3, action_dim=10,
        action_low=-1.0, action_high=1.0,
        elite_fraction=cfg.elite_fraction, rng=rng,
    )
    t0 = time.time()
    plan = sampler.sample_path(
        agent_image=t_img_a, goal_image=t_img_g,
        horizon=horizon, num_generations=cfg.num_generations,
        batch_size=cfg.batch_size, device=device,
    )
    plan_np = plan.cpu().numpy()
    plan_time = time.time() - t0

    # Evaluate latent
    H = 3
    emb = model.encode({"pixels": t_img_a.unsqueeze(0).unsqueeze(0)})["emb"]
    emb_g = model.encode({"pixels": t_img_g.unsqueeze(0).unsqueeze(0)})["emb"]
    history = []
    for action_10d in plan_np:
        n_hist = min(len(history), H)
        hist_arr = np.zeros((0, 10), dtype=np.float32) if not history else np.array(history[-n_hist:], dtype=np.float32)
        if len(history) < H:
            hist_arr = np.concatenate([np.zeros((H - len(history), 10), dtype=np.float32), hist_arr], axis=0)
        hist_tensor = torch.from_numpy(hist_arr).float().to(device)
        cand = torch.from_numpy(action_10d).float().to(device).unsqueeze(0)
        history_for_pred = torch.cat([hist_tensor.unsqueeze(0)[:, :-1, :], cand.unsqueeze(1)], dim=1)
        with torch.no_grad():
            act_emb = model.action_encoder(history_for_pred)
            emb = model.predict(emb, act_emb)[:, -1:, :]
    pred_final = emb.squeeze()
    goal_final = emb_g.squeeze()
    latent_mse = float((pred_final - goal_final).pow(2).mean().item())
    latent_dist = float((pred_final - goal_final).norm().item())

    # Evaluate env
    agent_pos, goal_pos, _ = (None, None, None)  # will be passed from outside
    return latent_dist, latent_mse, plan_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-problems", type=int, default=10)
    parser.add_argument("--horizon", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data/planning_benchmark")
    args = parser.parse_args()

    print("Loading models...")
    from utils.load_fix_pretrained import load_fix_pretrained
    from huggingface_hub import snapshot_download
    cp_path = snapshot_download(repo_id='quentinll/lewm-tworooms', revision='77adaae0bc31deab21c93740d1f8bb947cd0bdec')
    model = load_fix_pretrained(cp_path)
    model = model.to(args.device).eval()
    print(f"  JEPA loaded on {args.device}")

    configs = [
        AblationConfig(16, 5, 0.2),   # current default
        AblationConfig(32, 5, 0.2),
        AblationConfig(64, 5, 0.2),
        AblationConfig(16, 10, 0.2),
        AblationConfig(16, 5, 0.1),
        AblationConfig(16, 5, 0.3),
        AblationConfig(32, 10, 0.2),
    ]

    results = []
    rng = np.random.default_rng(args.seed)

    for h in args.horizon:
        print(f"\n{'='*60}")
        print(f"HORIZON={h}  flat_dim={h*10}")
        print(f"{'='*60}")
        for ci, cfg in enumerate(configs):
            total_samples = cfg.batch_size * cfg.num_generations
            samples_per_dim = total_samples / (h * 10)
            print(f"\n  [{ci+1}/{len(configs)}] bs={cfg.batch_size} gen={cfg.num_generations} elite={cfg.elite_fraction}  "
                  f"total_samples={total_samples}  samples_per_dim={samples_per_dim:.1f}")

            latent_dists = []
            latent_mses = []
            times = []

            for pi in range(args.num_problems):
                agent_pos, goal_pos, initial_dist = generate_problem(rng)
                t_img_a, t_img_g = make_images(agent_pos, goal_pos, args.device)
                try:
                    ld, lm, pt = run_cem(model, t_img_a, t_img_g, h, args.device, rng, cfg)
                    latent_dists.append(ld)
                    latent_mses.append(lm)
                    times.append(pt)
                except Exception as e:
                    print(f"    P{pi+1}: ERROR {e}")
                    continue

            if latent_dists:
                avg_ld = np.mean(latent_dists)
                avg_lm = np.mean(latent_mses)
                avg_t = np.mean(times)
                print(f"    → latent_dist={avg_ld:.2f}  mse={avg_lm:.4f}  time={avg_t:.2f}s")
                for pi in range(args.num_problems):
                    results.append({
                        "horizon": h,
                        "config_idx": ci,
                        "batch_size": cfg.batch_size,
                        "num_generations": cfg.num_generations,
                        "elite_fraction": cfg.elite_fraction,
                        "total_samples": total_samples,
                        "samples_per_dim": round(samples_per_dim, 1),
                        "flat_dim": h * 10,
                        "latent_dist": round(latent_dists[pi], 4),
                        "latent_mse": round(latent_mses[pi], 4),
                        "plan_time": round(times[pi], 4),
                    })

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join(args.output_dir, f"{ts}_cem_ablation.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "config": vars(args), "configs": [
            {"batch_size": c.batch_size, "num_generations": c.num_generations, "elite_fraction": c.elite_fraction}
            for c in configs
        ]}, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Summary
    print("\n" + "="*70)
    print("ABLATION SUMMARY (latent_dist, lower=better)")
    print("="*70)
    hdr = f"{'Config':<35} {'h=5':>8} {'h=10':>8} {'Δ':>7} {'h=5t':>7} {'h=10t':>7}"
    print(hdr)
    print("-"*70)
    for ci, cfg in enumerate(configs):
        r5 = [r for r in results if r["horizon"] == 5 and r["config_idx"] == ci]
        r10 = [r for r in results if r["horizon"] == 10 and r["config_idx"] == ci]
        if r5 and r10:
            a5 = np.mean([r["latent_dist"] for r in r5])
            a10 = np.mean([r["latent_dist"] for r in r10])
            t5 = np.mean([r["plan_time"] for r in r5])
            t10 = np.mean([r["plan_time"] for r in r10])
            label = f"bs={cfg.batch_size} gen={cfg.num_generations} elite={cfg.elite_fraction}"
            print(f"{label:<35} {a5:8.2f} {a10:8.2f} {a10-a5:+7.2f} {t5:7.1f} {t10:7.1f}")


if __name__ == "__main__":
    main()

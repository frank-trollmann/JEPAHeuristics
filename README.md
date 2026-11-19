# Planning with JEPA
### A CODE University of Applied Sciences semester project.

We study planning with a frozen JEPA world model (LeWorldModel, LeWM) in the TwoRoom environment: a wall with one door separates two rooms. The question is whether a better heuristic can be learned on the frozen latent space, and how sampling planners (CEM, MPPI, iCEM, ...) and search planners (greedy, A*, weighted A*, beam, MCTS) compare under different costs, including JEPA's own.

- **Design decisions and results (SE_38 hand-in):** [`docs/SE38-design-decisions.md`](docs/SE38-design-decisions.md)
- **Experiment record** (question, setup, result, limits of every experiment): [`docs/research-log.md`](docs/research-log.md)

## Repository structure (planning work, branch `adi/planning`)

| Path | What it contains |
|---|---|
| `utils/cem_planner.py` | CEM (LeWM's planner), the predictor rollout, JEPA's cost, `run_episode` (receding-horizon execution in the real environment) |
| `utils/sampling_planners.py` | MPPI, iCEM, snapped CEM (8 directions), random shooting, gradient descent on the plan |
| `utils/latent_search.py` | Greedy / A* / weighted A*, beam search and MCTS through the predictor (nodes are imagined latents) |
| `utils/search.py` | DFS, greedy, A*, weighted A* on the exact position grid (reference with true transitions) |
| `utils/heuristic_models.py` | Learned heuristics (pair MLP, metric model), baselines, metrics |
| `utils/reference_distance.py` | Wall-aware reference distances: grid Dijkstra and the continuous door formula |
| `utils/position_dataset.py`, `utils/pair_dataset.py` | Valid positions, split by position, encoding with the preprocessing the encoder expects, pair sampling |
| `utils/trajectory_dataset.py` | Frame-order heuristic (trained without coordinates) |
| `utils/position_probe.py`, `utils/dynamics_probe.py` | Read (x, y) from a latent; test whether the predictor knows the wall |
| `utils/tests/` | Unit tests for all of the above |
| `scripts/` | One runner per experiment (`run_planner_comparison.py` is the main one); exact commands in the research log |
| `docs/` | Design decisions, research log, figures |

Other folders (`utils/embedding_probing/`, `scripts/embedding_probing/`, dataset generators, notebooks) are teammates' work on probing and dataset generation.

## How to run

Requirements: Python 3.13+, [uv](https://docs.astral.sh/uv/), a CUDA GPU recommended (developed on an RTX 4050, 6 GB).

```
uv sync                                    # installs dependencies (CPU torch by default; install CUDA torch for GPU)
python -m pytest utils/tests -q            # unit tests
python -m scripts.generate_position_dataset --seed 0   # renders and encodes all valid positions into data/ (~3 min on GPU)
python -m scripts.run_planner_comparison --planners astar icem --heuristics learned jepa --kinds cross_far --tag _demo
python -m scripts.summarize_planner_comparison         # success rates with 95% intervals from data/results/
```

The LeWM checkpoint is downloaded from Hugging Face on first use (or pass `--checkpoint <folder>`). The planners need the mean and standard deviation of the dataset's actions (the predictor expects z-scored actions). They are read from the original LeWM `tworoom.h5` dataset (default path `stable_worldmodel/datasets/tworoom/tworoom.h5`, or `--actions-h5`), or, if that file is absent, from `data/action_stats.npz`. Neither is in git (`data/` is ignored for size), so a fresh clone needs one of them. `run_planner_comparison.py --help` lists all planners, heuristics and settings.

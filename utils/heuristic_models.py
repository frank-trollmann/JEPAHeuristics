"""
Baselines, models and metrics for learning a heuristic from (agent, goal) encodings.
"""
import numpy as np
import torch
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn


def metrics(y_true, y_pred):
  """Mean absolute error and root mean squared error (both in pixels) and R^2."""
  return {
    "mae": mean_absolute_error(y_true, y_pred),
    "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
    "r2": r2_score(y_true, y_pred),
  }


def latent_distance(encodings, agent, goal, squared):
  """
  One feature per pair: the distance between the agent and goal encodings, shape (num_pairs, 1).
  The squared version is the cost JEPA planning uses (JEPA.criterion), without the predictor.
  """
  distance = np.linalg.norm(encodings[agent] - encodings[goal], axis=1)
  return (distance ** 2 if squared else distance).reshape(-1, 1)


def fit_mean_baseline(y_train):
  """Always predicts the mean training label."""
  return DummyRegressor(strategy="mean").fit(np.zeros((len(y_train), 1)), y_train)


def fit_linear(x_train, y_train):
  return LinearRegression().fit(x_train, y_train)


def fit_linear_in_chunks(features, agent, goal, label, chunk=100_000):
  """
  Least squares with an intercept, solved from sums over chunks of pairs.
  Gives the same fit as LinearRegression, but never builds all features at once,
  so millions of pairs fit in memory.

  features(agent, goal) returns the feature matrix of those pairs.
  Returns predict(features_matrix) -> predictions.
  """
  xtx, xty = 0, 0
  for start in range(0, len(agent), chunk):
    stop = start + chunk
    x = features(agent[start:stop], goal[start:stop]).astype(np.float64)
    x = np.hstack([x, np.ones((len(x), 1))])  # last column is the intercept
    xtx = xtx + x.T @ x
    xty = xty + x.T @ label[start:stop]

  weights = np.linalg.lstsq(xtx, xty, rcond=None)[0]
  return lambda x: x @ weights[:-1] + weights[-1]


MODEL_KINDS = ("mlp", "mlp_diff", "metric")


class PairMLP(nn.Module):
  """MLP on the agent and goal encodings side by side, optionally with their difference as extra input."""

  def __init__(self, dim, with_difference):
    super().__init__()
    self.with_difference = with_difference
    inputs = (3 if with_difference else 2) * dim
    self.layers = nn.Sequential(nn.Linear(inputs, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

  def forward(self, agent, goal):
    parts = [agent, goal] + ([agent - goal] if self.with_difference else [])
    return self.layers(torch.cat(parts, dim=1)).squeeze(1)


class MetricNet(nn.Module):
  """
  h'(z_a, z_g) = ||f(z_a) - f(z_g)||: symmetric, never negative and obeys the triangle inequality,
  like a shortest-path distance does.
  """

  def __init__(self, dim, embed_dim=64):
    super().__init__()
    self.f = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, embed_dim))

  def forward(self, agent, goal):
    return ((self.f(agent) - self.f(goal)) ** 2).sum(dim=1).add(1e-8).sqrt()


def build_model(kind, dim):
  if kind == "mlp":
    return PairMLP(dim, with_difference=False)
  if kind == "mlp_diff":
    return PairMLP(dim, with_difference=True)
  if kind == "metric":
    return MetricNet(dim)
  raise ValueError(f"unknown model kind: {kind}")


class PairPredictor:
  """
  A trained pair model. Call it with index arrays (agent, goal) into the encodings it was trained on,
  or use on_latents for latent vectors that are not in that table, e.g. latents predicted by JEPA.
  Predictions are in label units (pixels).
  """

  def __init__(self, model, table, encodings_stats, y_mean, y_std):
    self.model, self.table, self.y_mean, self.y_std = model, table, y_mean, y_std
    self.device = table.device
    self.encodings_mean = torch.tensor(encodings_stats[0], dtype=torch.float32, device=self.device)
    self.encodings_std = torch.tensor(encodings_stats[1], dtype=torch.float32, device=self.device)

  def __call__(self, agent, goal, chunk=65536):
    agent_t, goal_t = torch.as_tensor(agent, device=self.device), torch.as_tensor(goal, device=self.device)
    with torch.no_grad():
      outputs = [self.model(self.table[agent_t[i:i + chunk]], self.table[goal_t[i:i + chunk]])
                 for i in range(0, len(agent_t), chunk)]
    return torch.cat(outputs).cpu().numpy() * self.y_std + self.y_mean

  def on_latents(self, agent_latents, goal_latents):
    """Predictions for latent vectors (N, D) given as torch tensors on the model's device; returns a torch tensor (N,)."""
    with torch.no_grad():
      standardised = [(z - self.encodings_mean) / self.encodings_std for z in (agent_latents, goal_latents)]
      return self.model(*standardised) * self.y_std + self.y_mean


def train_pair_model(encodings, train, val, device, seed, kind="mlp", epochs=30, batch_size=1024, lr=1e-3):
  """
  Trains a model that maps an (agent, goal) pair of encodings to a distance, and keeps the
  epoch with the lowest validation loss. kind is one of MODEL_KINDS.

  train and val are (agent, goal, label): index arrays into encodings, and the labels.
  Pairs are looked up by index inside each batch, so millions of pairs need no extra memory.
  Returns a PairPredictor.
  """
  torch.manual_seed(seed)
  train_agent, train_goal, train_label = train
  used = encodings[np.unique(np.concatenate([train_agent, train_goal]))]
  # a metric is never negative, so its target is only scaled, not centred
  y_mean = 0.0 if kind == "metric" else train_label.mean()
  y_std = train_label.std()

  # all encodings, standardised with the statistics of the training positions
  table = torch.tensor((encodings - used.mean(0)) / (used.std(0) + 1e-8), dtype=torch.float32, device=device)

  def to_tensors(agent, goal, label):
    return (
      torch.as_tensor(agent, device=device),
      torch.as_tensor(goal, device=device),
      torch.tensor((label - y_mean) / y_std, dtype=torch.float32, device=device),
    )

  train_t, val_t = to_tensors(*train), to_tensors(*val)

  model = build_model(kind, encodings.shape[1]).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  best_loss, best_state = float("inf"), None
  for _ in range(epochs):
    model.train()
    for batch in torch.randperm(len(train_t[0]), device=device).split(batch_size):
      optimizer.zero_grad()
      prediction = model(table[train_t[0][batch]], table[train_t[1][batch]])
      nn.functional.mse_loss(prediction, train_t[2][batch]).backward()
      optimizer.step()

    model.eval()
    with torch.no_grad():
      val_loss = nn.functional.mse_loss(model(table[val_t[0]], table[val_t[1]]), val_t[2]).item()
    if val_loss < best_loss:
      best_loss = val_loss
      best_state = {k: v.clone() for k, v in model.state_dict().items()}

  model.load_state_dict(best_state)
  model.eval()
  return PairPredictor(model, table, encodings_stats=(used.mean(0), used.std(0) + 1e-8), y_mean=y_mean, y_std=y_std)

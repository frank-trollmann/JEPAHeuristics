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


def train_mlp(encodings, train, val, device, seed, epochs=30, batch_size=1024, lr=1e-3):
  """
  Trains a small MLP on the concatenated agent and goal encodings and keeps the epoch
  with the lowest validation loss.

  train and val are (agent, goal, label): index arrays into encodings, and the labels.
  Pairs are looked up by index inside each batch, so millions of pairs need no extra memory.
  Returns predict(agent, goal), which gives predictions (numpy, in label units).
  """
  torch.manual_seed(seed)
  train_agent, train_goal, train_label = train
  used = encodings[np.unique(np.concatenate([train_agent, train_goal]))]
  y_mean, y_std = train_label.mean(), train_label.std()

  # all encodings, standardised with the statistics of the training positions
  table = torch.tensor((encodings - used.mean(0)) / (used.std(0) + 1e-8), dtype=torch.float32, device=device)

  def to_tensors(agent, goal, label):
    return (
      torch.as_tensor(agent, device=device),
      torch.as_tensor(goal, device=device),
      torch.tensor((label - y_mean) / y_std, dtype=torch.float32, device=device),
    )

  def inputs(agent, goal):
    return torch.cat([table[agent], table[goal]], dim=1)

  train_t, val_t = to_tensors(*train), to_tensors(*val)

  model = nn.Sequential(
    nn.Linear(2 * encodings.shape[1], 256), nn.ReLU(),
    nn.Linear(256, 256), nn.ReLU(),
    nn.Linear(256, 1),
  ).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  best_loss, best_state = float("inf"), None
  for _ in range(epochs):
    model.train()
    for batch in torch.randperm(len(train_t[0]), device=device).split(batch_size):
      optimizer.zero_grad()
      prediction = model(inputs(train_t[0][batch], train_t[1][batch])).squeeze(1)
      nn.functional.mse_loss(prediction, train_t[2][batch]).backward()
      optimizer.step()

    model.eval()
    with torch.no_grad():
      val_loss = nn.functional.mse_loss(model(inputs(val_t[0], val_t[1])).squeeze(1), val_t[2]).item()
    if val_loss < best_loss:
      best_loss = val_loss
      best_state = {k: v.clone() for k, v in model.state_dict().items()}

  model.load_state_dict(best_state)
  model.eval()

  def predict(agent, goal, chunk=65536):
    agent_t, goal_t = torch.as_tensor(agent, device=device), torch.as_tensor(goal, device=device)
    with torch.no_grad():
      outputs = [model(inputs(agent_t[i:i + chunk], goal_t[i:i + chunk])).squeeze(1) for i in range(0, len(agent_t), chunk)]
    return torch.cat(outputs).cpu().numpy() * y_std + y_mean

  return predict

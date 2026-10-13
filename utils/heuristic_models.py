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


def train_mlp(x_train, y_train, x_val, y_val, device, seed, epochs=30, batch_size=1024, lr=1e-3):
  """
  Trains a small MLP and keeps the epoch with the lowest validation loss.
  Returns a function mapping inputs (numpy) to predictions (numpy, in label units).
  """
  torch.manual_seed(seed)
  x_mean, x_std = x_train.mean(0), x_train.std(0) + 1e-8
  y_mean, y_std = y_train.mean(), y_train.std()

  def prepare_x(x):
    return torch.tensor((x - x_mean) / x_std, dtype=torch.float32, device=device)

  def prepare_y(y):
    return torch.tensor((y - y_mean) / y_std, dtype=torch.float32, device=device)

  xt, yt, xv, yv = prepare_x(x_train), prepare_y(y_train), prepare_x(x_val), prepare_y(y_val)

  model = nn.Sequential(
    nn.Linear(xt.shape[1], 256), nn.ReLU(),
    nn.Linear(256, 256), nn.ReLU(),
    nn.Linear(256, 1),
  ).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  best_loss, best_state = float("inf"), None
  for _ in range(epochs):
    model.train()
    for batch in torch.randperm(len(xt), device=device).split(batch_size):
      optimizer.zero_grad()
      loss = nn.functional.mse_loss(model(xt[batch]).squeeze(1), yt[batch])
      loss.backward()
      optimizer.step()

    model.eval()
    with torch.no_grad():
      val_loss = nn.functional.mse_loss(model(xv).squeeze(1), yv).item()
    if val_loss < best_loss:
      best_loss = val_loss
      best_state = {k: v.clone() for k, v in model.state_dict().items()}

  model.load_state_dict(best_state)
  model.eval()

  def predict(x):
    with torch.no_grad():
      return model(prepare_x(x)).squeeze(1).cpu().numpy() * y_std + y_mean

  return predict

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from utils.true_distance.models.mlp.model import TorchMLP


def train_torch_mlp(
  X_train,
  y_train,
  X_val,
  y_val,
  hidden_dims=(128, 64),
  dropout=0.1,
  lr=1e-3,
  weight_decay=1e-4,
  batch_size=64,
  max_epochs=200,
  patience=10,
  device="cpu",
  seed=42,
):
  """
  Train a TorchMLP with early stopping on validation MAE.
  """
  torch.manual_seed(seed)

  # convert numpy arrays to pytorch tensors 
  X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
  y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
  X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
  y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

  loader = DataLoader(
    TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
  )

  model = TorchMLP(
    input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout
  ).to(device)
  optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
  loss_fn = nn.L1Loss()  # MAE loss

  best_val_mae = float("inf")
  best_state = None # saves model weights 
  epochs_without_improvement = 0

  for _epoch in range(max_epochs):
    model.train() # train mode
    for X_batch, y_batch in loader: # random batch samples from loader
      optimizer.zero_grad() # all grads to 0 
      y_pred = model(X_batch) # forward pass
      loss = loss_fn(y_pred, y_batch)
      loss.backward() # backward pass
      optimizer.step() 

    model.eval() # eval mode
    with torch.no_grad(): # no gradient calc for this step
      val_mae = loss_fn(model(X_val_t), y_val_t).item() 

    # is current iteration better then record 
    if val_mae < best_val_mae:
      best_val_mae = val_mae
      best_state = {k: v.clone() for k, v in model.state_dict().items()}
      epochs_without_improvement = 0
    else:
      epochs_without_improvement += 1
      if epochs_without_improvement >= patience:
        break

  model.load_state_dict(best_state) # dict of weights 
  return model, best_val_mae


def predict_torch_mlp(model, X, device="cpu"):
  model.eval()
  with torch.no_grad():
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_pred = model(X_t).cpu().numpy()
  return y_pred

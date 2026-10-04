from torch import nn


class TorchMLP(nn.Module):
  """
  Basic MLP configuration with hidden_dims hidden layers 
  One hidden layer consists of 4 Layer 
  """

  def __init__(self, input_dim, hidden_dims, dropout=0.1):
    super().__init__()
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
      layers.append(nn.Linear(prev_dim, hidden_dim))
      layers.append(nn.BatchNorm1d(hidden_dim))
      layers.append(nn.ReLU())
      layers.append(nn.Dropout(dropout))
      prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, 1))
    self.net = nn.Sequential(*layers)

  def forward(self, x):
    return self.net(x).squeeze(-1)

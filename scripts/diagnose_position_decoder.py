"""
Diagnosis script to check if JEPA encodings actually contain position information.
It trains a simple MLP: Encoding (192) -> Position (x, y).
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import utils.batchpredict_generate_jepa_heuristic_dataset as gen

def train_position_decoder(X, y, epochs=100, lr=1e-3, batch_size=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.from_numpy(X).float().to(device)
    y = torch.from_numpy(y).float().to(device)
    
    # Simple MLP: 192 -> 128 -> 64 -> 2
    model = torch.nn.Sequential(
        torch.nn.Linear(192, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2)
    ).to(device)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            
    return model

def main():
    # 1. Generate a small diagnostic dataset
    print("Generating diagnostic dataset (100k samples)...")
    df = gen.batchpredict_generate_jepa_heuristic_dataset(num_samples=100_000, batch_size=256)
    
    # 2. Split into Train/Test
    # We use agent_pos_enc and coords_a for the diagnosis
    X = np.stack(df["agent_pos_enc"].values).astype(np.float32)
    y = np.stack(df["coords_a"].values).astype(np.float32)
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Training decoder on {len(X_train)} samples...")
    model = train_position_decoder(X_train, y_train)
    
    # 3. Evaluate
    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).float().to(next(model.parameters()).device)
        y_test_t = torch.from_numpy(y_test).float().to(next(model.parameters()).device)
        preds = model(X_test_t).cpu().numpy()
    
    # Calculate Euclidean distance error (in pixels)
    errors = np.linalg.norm(preds - y_test, axis=1)
    mae = np.mean(errors)
    median_err = np.median(errors)
    max_err = np.max(errors)
    
    print("\n" + "="*30)
    print("DIAGNOSIS RESULTS")
    print("="*30)
    print(f"Mean Position Error:    {mae:.2f} pixels")
    print(f"Median Position Error: {median_err:.2f} pixels")
    print(f"Max Position Error:    {max_err:.2f} pixels")
    print("="*30)
    
    # Contextual check: How large is the room?
    # TwoRoomEnv usually has an image size of 224x224.
    print(f"\nNote: Room size is approx 224x224.")
    if mae < 10:
        print("\nCONCLUSION: The encodings contain highly precise position info. \nTHE POSITIONS-DECODER APPROACH WILL LIKELY WORK.")
    elif mae < 30:
        print("\nCONCLUSION: The encodings contain rough position info. \nSOME improvement possible, but expect limits.")
    else:
        print("CONCLUSION: The encodings barely contain position info. \nTHE MLP is already hitting the ceiling.")

if __name__ == "__main__":
    main()

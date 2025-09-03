"""
Supervised Training Script
--------------------------
- Trains LSTM (or other models) on historical data
- Logs metrics
- Saves best model
"""

import os
import torch
import torch.optim as optim
import torch.nn as nn
import logging
from ai.models.lstm import LSTMBrain
from ai.feature_engineer import add_indicators
from backtesting.metrics import compute_metrics
import pandas as pd


def train_model(train_loader, input_dim, epochs=10, lr=0.001, save_path="models/lstm_brain.pt"):
    logger = logging.getLogger("SupervisedTrainer")
    os.makedirs("models", exist_ok=True)

    model = LSTMBrain(input_dim=input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")

    for epoch in range(epochs):
        total_loss = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs.squeeze(), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{epochs} → Loss={avg_loss:.6f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            logger.info(f"💾 Best model saved @ {save_path}")

    return model


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Example dataset
    df = pd.read_csv("data/historical.csv", parse_dates=["time"])
    df = add_indicators(df)

    # TODO: convert df into DataLoader (X, y)

    # Example: dummy loader (replace with real)
    from torch.utils.data import DataLoader, TensorDataset
    import torch

    X = torch.randn(100, 30, 5)  # (samples, timesteps, features)
    y = torch.randn(100)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = train_model(loader, input_dim=5, epochs=10)
    print("✅ Supervised training complete")


if __name__ == "__main__":
    main()

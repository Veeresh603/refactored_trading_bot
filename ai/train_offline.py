"""
Offline Training Script
-----------------------
- Prepares dataset from historical market data
- Trains LSTM / Transformer / Ensemble offline
- Saves trained models for later inference
"""

import os
import pandas as pd
import torch
import logging
from torch.utils.data import DataLoader, TensorDataset

from ai.feature_engineer import add_indicators
from ai.models.lstm import LSTMBrain
from ai.models.transformer import TransformerModel
from ai.models.rl_agent import MLPAgent
from backtesting.metrics import compute_metrics


def prepare_offline_data(filepath="data/nifty50.csv", window_size=30):
    """
    Load historical data and prepare features/labels.
    """
    df = pd.read_csv(filepath, parse_dates=["time"])
    df = add_indicators(df)

    X, y = [], []
    for i in range(len(df) - window_size - 1):
        window = df.iloc[i:i+window_size]
        target = df.iloc[i+window_size]["close"]
        X.append(window.drop(columns=["time"]).values)
        y.append(target)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=32, shuffle=True), X.shape[-1]


def train_offline(filepath="data/nifty50.csv", epochs=10, save_dir="models"):
    """
    Train offline supervised models (LSTM + Transformer).
    """
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("OfflineTrainer")

    loader, input_dim = prepare_offline_data(filepath)

    # --- Train LSTM ---
    lstm = LSTMBrain(input_dim=input_dim)
    lstm_opt = torch.optim.Adam(lstm.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    best_loss = float("inf")

    for ep in range(epochs):
        total_loss = 0
        for X, y in loader:
            lstm_opt.zero_grad()
            out = lstm(X)
            loss = criterion(out.squeeze(), y)
            loss.backward()
            lstm_opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        logger.info(f"[LSTM] Epoch {ep+1}/{epochs} → Loss={avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(lstm.state_dict(), os.path.join(save_dir, "lstm_offline.pt"))
            logger.info("💾 Saved best LSTM model")

    # --- Train Transformer ---
    transformer = TransformerModel(input_dim=input_dim)
    tr_opt = torch.optim.Adam(transformer.parameters(), lr=0.001)
    best_loss = float("inf")

    for ep in range(epochs):
        total_loss = 0
        for X, y in loader:
            tr_opt.zero_grad()
            out = transformer(X)
            loss = criterion(out.squeeze(), y)
            loss.backward()
            tr_opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        logger.info(f"[Transformer] Epoch {ep+1}/{epochs} → Loss={avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(transformer.state_dict(), os.path.join(save_dir, "transformer_offline.pt"))
            logger.info("💾 Saved best Transformer model")

    logger.info("✅ Offline training complete")


def main():
    train_offline()


if __name__ == "__main__":
    main()

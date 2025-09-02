"""
Supervised training script for ML models
"""

import torch
import torch.optim as optim
import torch.nn as nn
from ai.models.lstm import LSTMBrain
from ai.feature_engineer import add_indicators


def train_model(train_loader, input_dim, epochs=10, lr=0.001):
    model = LSTMBrain(input_dim=input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for X, y in train_loader:
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return model

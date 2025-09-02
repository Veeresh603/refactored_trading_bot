import torch
import torch.nn as nn


class LSTMBrain(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=1, num_layers=2):
        super(LSTMBrain, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # last timestep
        return out

"""
Simple Transformer model for trading
"""
import torch
import torch.nn as nn


class TransformerBrain(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=1, num_heads=4, num_layers=2):
        super(TransformerBrain, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc_in(x)
        out = self.transformer(x)
        out = self.fc_out(out[:, -1, :])
        return out

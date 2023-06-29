import torch.nn as nn

from .layers import LayerNorm


class Layer(nn.Module):
    def __init__(self, channels, kernel_size, dropout):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.act = nn.ReLU()
        self.norm = LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class DurationPredictor(nn.Module):
    def __init__(self, channels, kernel_size, dropout, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            Layer(channels, kernel_size, dropout)
            for _ in range(num_layers)
        ])
        self.out_layer = nn.Conv1d(channels, 1, 1)
        
    def forward(self, x, mask):
        x = x.detach()
        for layer in self.layers:
            x = layer(x * mask)
        x = self.out_layer(x * mask)
        return x * mask

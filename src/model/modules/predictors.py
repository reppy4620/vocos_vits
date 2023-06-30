import torch.nn as nn

from .layers import ConvNeXtLayer


class DurationPredictor(nn.Module):
    def __init__(self, channels, h_channels, dropout, num_layers):
        super().__init__()
        scale = 1. / num_layers
        self.layers = nn.ModuleList([
            ConvNeXtLayer(channels, h_channels, scale)
            for _ in range(num_layers)
        ])
        self.out_layer = nn.Conv1d(channels, 1, 1)
        
    def forward(self, x, mask):
        x = x.detach()
        for layer in self.layers:
            x = layer(x, mask)
        x = self.out_layer(x) * mask
        return x

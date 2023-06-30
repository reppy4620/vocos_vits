import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import LayerNorm


class Layer(nn.Module):
    def __init__(self, channels, h_channels, scale, dropout=0.):
        super().__init__()
        self.dw_conv = nn.Conv1d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = LayerNorm(channels)
        self.pw_conv1 = nn.Conv1d(channels, h_channels, 1)
        self.pw_conv2 = nn.Conv1d(h_channels, channels, 1)
        self.scale = nn.Parameter(torch.full(size=(1, channels, 1), fill_value=scale), requires_grad=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        res = x
        x = self.dw_conv(x)
        x = self.norm(x)
        x = self.pw_conv1(x * mask)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.pw_conv2(x * mask)
        x = self.scale * x
        x = res + x
        return x * mask


class DurationPredictor(nn.Module):
    def __init__(self, channels, h_channels, dropout, num_layers):
        super().__init__()
        scale = 1. / num_layers
        self.layers = nn.ModuleList([
            Layer(channels, h_channels, scale, dropout)
            for _ in range(num_layers)
        ])
        self.out_layer = nn.Conv1d(channels, 1, 1)
        
    def forward(self, x, mask):
        x = x.detach()
        for layer in self.layers:
            x = layer(x, mask)
        x = self.out_layer(x) * mask
        return x

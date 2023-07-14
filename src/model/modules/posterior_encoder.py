import torch
import torch.nn as nn

from .layers import WaveNetLayer


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        kernel_size,
        dilation_rate,
        num_layers,
    ):
        super().__init__()
        self.out_channels = out_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WaveNetLayer(hidden_channels, kernel_size, dilation_rate, num_layers)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, mask):
        x = self.pre(x) * mask
        x = self.enc(x, mask)
        x = self.proj(x) * mask
        m, logs = x.split([self.out_channels] * 2, dim=1)
        z = (m + torch.exp(logs) * torch.randn_like(logs)) * mask
        return z, m, logs

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()

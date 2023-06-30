import torch
import torch.nn as nn
import torch.nn.functional as F

from torchaudio.transforms import InverseSpectrogram

from .layers import LayerNorm, ConvNeXtLayer
    

class Vocoder(nn.Module):
    def __init__(self, in_channels, channels, h_channels, out_channels, num_layers, istft_config):
        super().__init__()
        self.pad = nn.ReflectionPad1d([1, 0])
        self.in_conv = nn.Conv1d(in_channels, channels, kernel_size=7, padding=3)
        self.norm_pre = LayerNorm(channels)
        scale = 1 / num_layers
        self.layers = nn.ModuleList(
            [
                ConvNeXtLayer(channels, h_channels, scale)
                for _ in range(num_layers)
            ]
        )
        self.norm_post = LayerNorm(channels)
        self.out_conv = nn.Conv1d(channels, out_channels, kernel_size=1)
        self.istft = InverseSpectrogram(**istft_config)

    def forward(self, x):
        x = self.pad(x)
        x = self.in_conv(x)
        x = self.norm_pre(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_post(x)
        x = self.out_conv(x)
        mag, phase = x.chunk(2, dim=1)
        mag = mag.exp().clamp_max(max=1e2)
        s = mag * (phase.cos() + 1j * phase.sin())
        o = self.istft(s).unsqueeze(1)
        return o

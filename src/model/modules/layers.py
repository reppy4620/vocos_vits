import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import weight_norm, remove_weight_norm


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(1, channels, 1))
        self.beta = torch.nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor):
        mean = torch.mean(x, dim=1, keepdim=True)
        variance = torch.mean((x - mean)**2, dim=1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)
        x = x * self.gamma + self.beta
        return x


class WaveNetLayer(nn.Module):
    def __init__(self, channels, kernel_size, dilation_rate, num_layers, dropout=0):
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for i in range(num_layers):
            dilation = dilation_rate ** i
            self.in_layers.append(
                weight_norm(
                    nn.Conv1d(
                        channels, 
                        2 * channels,
                        kernel_size,
                        padding=kernel_size//2,
                        dilation=dilation
                    )
                )
            )
            # No split at the end module.
            if i == num_layers - 1:
                res_skip_channels = channels
            else:
                res_skip_channels = channels * 2
            
            self.res_skip_layers.append(
                weight_norm(
                    nn.Conv1d(channels, res_skip_channels, kernel_size=1)
                )
            )

    def forward(self, x, mask):
        o = torch.zeros_like(x)
        for i, (in_layer, skip_layer) in enumerate(zip(self.in_layers, self.res_skip_layers)):
            x_in = in_layer(x)
            x1, x2 = x_in.split([self.channels]*2, dim=1)
            acts = x1.tanh() * x2.sigmoid()
            acts = self.dropout(acts)

            x_acts = skip_layer(acts)
            if i == self.num_layers - 1:
                o = o + x_acts
            else:
                x1, x2 = x_acts.split([self.channels]*2, dim=1)
                x = (x + x1) * mask
                o = o + x2
        return o * mask

    def remove_weight_norm(self):
        for layer in self.in_layers:
            remove_weight_norm(layer)
        for layer in self.res_skip_layers:
            remove_weight_norm(layer)
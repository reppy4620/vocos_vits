import torch
import torch.nn as nn

from .layers import WaveNetLayer


class Flip(nn.Module):
    def forward(self, x, mask):
        x = torch.flip(x, [1])
        return x

    def reverse(self, x, mask):
        return self(x, mask)
    

class ResidualCouplingLayer(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        dilation_rate,
        n_layers,
        dropout=0
    ):
        super().__init__()
        self.half_channels = channels // 2

        self.pre = nn.Conv1d(channels // 2, channels, 1)
        self.wn = WaveNetLayer(channels, kernel_size, dilation_rate, n_layers, dropout=dropout)
        self.post = nn.Conv1d(channels, channels // 2, 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def _stats(self, x, mask):
        x0, x1 = x.split([self.half_channels]*2, dim=1)
        h = self.pre(x0) * mask
        h = self.wn(h, mask)
        stats = self.post(h) * mask

        m = stats
        logs = torch.zeros_like(m)
        return x0, x1, m, logs

    def forward(self, x, mask):
        x0, x1, m, logs = self._stats(x, mask)
        x1 = m + x1 * torch.exp(logs) * mask
        x = torch.cat([x0, x1], dim=1)
        return x

    def reverse(self, x, mask):
        x0, x1, m, logs = self._stats(x, mask)
        x1 = (x1 - m) * torch.exp(-logs) * mask
        x = torch.cat([x0, x1], dim=1)
        return x

    def remove_weight_norm(self):
        self.wn.remove_weight_norm()


class Flow(nn.Module):
    def __init__(self, channels, kernel_size, dilation_rate, n_layers, n_flows=4):
        super().__init__()

        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows += [
                ResidualCouplingLayer(
                    channels,
                    kernel_size,
                    dilation_rate,
                    n_layers
                ),
                Flip()
            ]
    
    def forward(self, x, mask):
        for flow in self.flows:
            x = flow(x, mask)
        return x

    def reverse(self, x, mask):
        for flow in reversed(self.flows):
            x = flow.reverse(x, mask)
        return x
    
    def remove_weight_norm(self):
        for flow in self.flows:
            flow.remove_weight_norm()

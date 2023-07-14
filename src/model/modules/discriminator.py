import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


LRELU_SLOPE = 0.1


class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        1024, 1024, (kernel_size, 1), 1, padding=(kernel_size // 2, 0)
                    )
                ),
            ]
        )
        self.conv_post = weight_norm(
            nn.Conv2d(1024, 1, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, periods=[2, 3, 5, 7, 11]):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(p) for p in periods])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorR(nn.Module):
    def __init__(self, resolution, channels=64, in_channels=1):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv2d(
                        in_channels,
                        channels,
                        kernel_size=(7, 5),
                        stride=(2, 2),
                        padding=(3, 2),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=(5, 3),
                        stride=(2, 1),
                        padding=(2, 1),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=(5, 3),
                        stride=(2, 2),
                        padding=(2, 1),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        channels, channels, kernel_size=3, stride=(2, 1), padding=1
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        channels, channels, kernel_size=3, stride=(2, 2), padding=1
                    )
                ),
            ]
        )
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        fmap = []
        x = self.spectrogram(x.squeeze(1))
        x = x.unsqueeze(1)
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        n_fft, hop_length, win_length = self.resolution
        magnitude_spectrogram = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=None,
            center=True,
            return_complex=True,
        ).abs()
        return magnitude_spectrogram


class MultiResolutionDiscriminator(nn.Module):
    def __init__(
        self, resolutions=((1024, 256, 1024), (2048, 512, 2048), (512, 128, 512))
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(resolution=r) for r in resolutions]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(x=y)
            y_d_g, fmap_g = d(x=y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator()
        self.mrd = MultiResolutionDiscriminator()

    def forward(self, y, y_hat):
        o1 = self.mpd(y, y_hat)
        o2 = self.mrd(y, y_hat)
        return [l1 + l2 for l1, l2 in zip(o1, o2)]

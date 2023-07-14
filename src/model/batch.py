import torch
from typing import List, NamedTuple


class Batch(NamedTuple):
    bnames: List[str]  # file stem (e.g. hoge.txt => hoge)
    phoneme: torch.Tensor  # phoneme id (see text.py)
    duration: torch.Tensor  # duration (extracted from fullcontext)
    x_lengths: torch.Tensor  # length of each phoneme id list
    wav: torch.Tensor  # waveform
    sample_lengths: torch.Tensor  # length of each waveform
    spec: torch.Tensor  # linear spectrogram
    frame_lengths: torch.Tensor  # length of each frame level feature (e.g. spectrogram, cf0, vuv)

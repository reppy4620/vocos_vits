from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from .prior_encoder import PhonemeEncoder
from .posterior_encoder import PosteriorEncoder
from .predictors import DurationPredictor
from .flow import Flow
from .vocoder import Vocoder

from ..batch import Batch
from ..utils import sequence_mask, generate_path, rand_slice_segments


@dataclass
class VITSOutput:
    wav: torch.Tensor
    duration: torch.Tensor
    m_p: torch.Tensor
    logs_p: torch.Tensor
    z: torch.Tensor
    z_p: torch.Tensor
    m_q: torch.Tensor
    logs_q: torch.Tensor
    x_mask: torch.Tensor
    frame_mask: torch.Tensor
    idx_slice: List[int]


class VITS(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.segment_size = params.segment_size
        self.hop_length = params.hop_length
        self.sample_segment_size = self.segment_size * self.hop_length

        self.phoneme_encoder    = PhonemeEncoder(**params.phoneme_encoder)
        self.duration_predictor = DurationPredictor(**params.duration_predictor)
        self.flow               = Flow(**params.flow)
        self.posterior_encoder  = PosteriorEncoder(**params.posterior_encoder)
        self.vocoder            = Vocoder(**params.vocoder)

    def forward(self, batch: Batch) -> VITSOutput:
        (
            _,
            x,
            duration,
            x_lengths,
            _,
            _,
            spec,
            frame_lengths
        ) = batch

        x_mask = sequence_mask(x_lengths, x.shape[-1]).unsqueeze(1).to(x.dtype)
        x, m_p, logs_p = self.phoneme_encoder(x, x_mask)
        duratino_pred = self.duration_predictor(x, x_mask)

        frame_mask = sequence_mask(frame_lengths, spec.shape[-1]).unsqueeze(1).to(spec.dtype)
        path_mask = x_mask.unsqueeze(-1) * frame_mask.unsqueeze(2)
        attn_path = generate_path(duration.squeeze(1), path_mask.squeeze(1))
        m_p = m_p @ attn_path
        logs_p = logs_p @ attn_path

        z, m_q, logs_q = self.posterior_encoder(spec, frame_mask)
        z_p = self.flow(z, frame_mask)

        z_slice, idx_slice = rand_slice_segments(z, frame_lengths, segment_size=self.segment_size)
        o = self.vocoder(z_slice)

        return VITSOutput(
            wav=o,
            duration=duratino_pred,
            m_p=m_p,
            logs_p=logs_p,
            z=z,
            z_p=z_p,
            m_q=m_q,
            logs_q=logs_q,
            x_mask=x_mask,
            frame_mask=frame_mask,
            idx_slice=idx_slice
        )
    
    def infer(self, x: torch.Tensor, noise_scale: float = 0.667) -> torch.Tensor:
        # x : [1, P]
        # P: Phoneme level length

        x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=x.device)
        x_mask = sequence_mask(x_lengths).unsqueeze(1).to(x.dtype)

        x, m_p, logs_p = self.phoneme_encoder(x, x_mask)
        log_duration = self.duration_predictor(x, x_mask)
        duration = torch.ceil(torch.exp(log_duration)).long()

        frame_lengths = torch.tensor([duration.sum()], dtype=torch.long, device=x.device)
        frame_mask = sequence_mask(frame_lengths).unsqueeze(1).to(x.dtype)
        path_mask = x_mask.unsqueeze(-1) * frame_mask.unsqueeze(2)
        attn_path = generate_path(duration.squeeze(1), path_mask.squeeze(1))
        m_p = m_p @ attn_path
        logs_p = logs_p @ attn_path

        z_p = m_p + torch.exp(logs_p) * torch.randn_like(logs_p) * noise_scale
        z = self.flow.reverse(z_p, frame_mask)
        
        o = self.vocoder(z)
        return o

    def remove_weight_norm(self):
        self.flow.remove_weight_norm()
        self.posterior_encoder.remove_weight_norm()
        self.vocoder.remove_weight_norm()

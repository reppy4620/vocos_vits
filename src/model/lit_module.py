import torch
import torch.optim as optim
import torch.nn.functional as F

from typing import Union, List
from lightning import LightningModule

from text import text_to_sequence
from audio import SpectrogramTransform
from .modules.vits import VITS, VITSOutput
from .modules.discriminator import Discriminator
from .utils import slice_segments, to_log_scale
from .batch import Batch
from .loss import (
    discriminator_loss, 
    generator_loss, 
    kl_loss,
    feature_loss, 
    masked_mse_loss
)


class VITSModule(LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.loss_coef = params.loss_coef
        self.segment_size = params.segment_size
        self.sample_segment_size = params.sample_segment_size
        self.hop_length = params.hop_length

        self.automatic_optimization = False

        self.net_g = VITS(params.vits)
        self.net_d = Discriminator()

        self.spec_tfm = SpectrogramTransform(**params.audio)
    
    def forward(self, x: Union[torch.Tensor, str, List[str]], noise_scale: float = 0.667):
        if isinstance(x, torch.Tensor):
            if len(x.shape) == 1:
                x = x.unsqueeze(0).to(self.device)
        elif isinstance(x, str):
            x = torch.LongTensor(text_to_sequence(x.split())).unsqueeze(0).to(self.device)
        elif isinstance(x, list):
            x = torch.LongTensor(text_to_sequence(x)).unsqueeze(0).to(self.device)
        return self.net_g.infer(x, noise_scale=noise_scale).cpu().squeeze(1)
    
    def infer_gt(self, x):
        return self.net_g.infer_gt(x).cpu().squeeze(1)
    
    def _handle_batch(self, batch, train=True):
        batch = Batch(*batch)
        optimizer_g, optimizer_d = self.optimizers()
        
        o: VITSOutput = self.net_g(batch)

        y = slice_segments(batch.wav, o.idx_slice * self.hop_length, self.sample_segment_size)
        y_hat = o.wav

        mel = self.spec_tfm.spec_to_mel(batch.spec)
        y_mel = slice_segments(mel, o.idx_slice, self.segment_size)
        y_hat_mel = self.spec_tfm.to_mel(y_hat.squeeze(1))
        
        d_real, d_fake, _, _ = self.net_d(y, y_hat.detach())
        loss_d = discriminator_loss(d_real, d_fake)
        if train:
            optimizer_d.zero_grad()
            self.manual_backward(loss_d)
            optimizer_d.step()

        _, d_fake, fmap_real, fmap_fake = self.net_d(y, y_hat)
        loss_gen = generator_loss(d_fake)
        loss_dur = masked_mse_loss(o.duration, to_log_scale(batch.duration), mask=o.x_mask)
        loss_mel = self.loss_coef.mel * F.l1_loss(y_hat_mel, y_mel)
        loss_kl  = kl_loss(o.z_p, o.logs_q, o.m_p, o.logs_p, o.frame_mask)
        loss_fm  = self.loss_coef.feature_matching * feature_loss(fmap_real, fmap_fake)
        loss_g = (
            loss_gen +
            loss_dur +
            loss_mel +
            loss_kl +
            loss_fm
        )
        if train:
            optimizer_g.zero_grad()
            self.manual_backward(loss_g)
            optimizer_g.step()

        loss_dict = {
            'disc': loss_d,
            'gen': loss_gen,
            'dur': loss_dur,
            'mel': loss_mel,
            'kl': loss_kl,
            'fm': loss_fm
        }
        
        self.log_dict(loss_dict, prog_bar=True)
    
    def training_step(self, batch):
        self._handle_batch(batch, train=True)

    def on_train_epoch_end(self):
        scheduler_g, scheduler_d = self.lr_schedulers()
        scheduler_g.step()
        scheduler_d.step()

    def validation_step(self, batch, batch_idx):
        self._handle_batch(batch, train=False)

    def configure_optimizers(self):
        optimizer_g = optim.AdamW(self.net_g.parameters(), **self.params.optimizer_g)
        optimizer_d = optim.AdamW(self.net_d.parameters(), **self.params.optimizer_d)
        scheduler_g = optim.lr_scheduler.ExponentialLR(optimizer_g, **self.params.scheduler_g)
        scheduler_d = optim.lr_scheduler.ExponentialLR(optimizer_d, **self.params.scheduler_d)
        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]

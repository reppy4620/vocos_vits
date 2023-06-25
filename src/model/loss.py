import torch 
from torch.nn import functional as F


def feature_loss(fmap_real, fmap_fake):
    loss = 0
    for d_real, d_fake in zip(fmap_real, fmap_fake):
        for o_real, o_fake in zip(d_real, d_fake):
            o_real = o_real.detach()
            loss += torch.mean(torch.abs(o_real - o_fake))
    return loss


def discriminator_loss(disc_real, disc_fake):
    loss = 0
    for d_real, d_fake in zip(disc_real, disc_fake):
        real_loss = torch.mean((1 - d_real) ** 2)
        fake_loss = torch.mean(d_fake ** 2)
        loss += (real_loss + fake_loss)
    return loss


def generator_loss(disc_outputs):
    loss = 0
    for d_fake in disc_outputs:
        d_fake = d_fake.float()
        l = torch.mean((1 - d_fake) ** 2)
        loss += l
    return loss


# if you suspect this implementation, please browse this page 
# "https://github.com/jaywalnut310/vits/issues/6"
def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2. * logs_p)
    kl = torch.sum(kl * z_mask)
    loss = kl / torch.sum(z_mask)
    return loss


def masked_mse_loss(pred, target, mask):
    return F.mse_loss(pred, target, reduction='sum') / mask.sum()


def masked_bce_loss(pred, target, mask):
    return F.binary_cross_entropy(pred, target, reduction='sum') / mask.sum()

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import LayerNorm


# Windowed Relative Positional Encoding is applied
class MultiHeadAttention(nn.Module):
    def __init__(self, channels, n_heads, dropout, window_size=4):
        super().__init__()
        assert channels % n_heads == 0

        self.inter_channels = channels // n_heads
        self.n_heads = n_heads
        self.window_size = window_size
        self.scale = math.sqrt(self.inter_channels)

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(dropout)

        rel_stddev = self.inter_channels**-0.5
        self.emb_rel_k = nn.Parameter(
            torch.randn(1, window_size * 2 + 1, self.inter_channels) * rel_stddev
        )
        self.emb_rel_v = nn.Parameter(
            torch.randn(1, window_size * 2 + 1, self.inter_channels) * rel_stddev
        )

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, mask):
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)

        B, C, T = q.size()
        H = self.n_heads
        query = q.view(B, H, self.inter_channels, T).transpose(2, 3)
        key = k.view(B, H, self.inter_channels, T).transpose(2, 3)
        value = v.view(B, H, self.inter_channels, T).transpose(2, 3)

        score = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        pad_length = max(0, T - (self.window_size + 1))
        start = max(0, (self.window_size + 1) - T)
        end = start + 2 * T - 1

        pad_rel_emb = F.pad(self.emb_rel_k, [0, 0, pad_length, pad_length, 0, 0])
        k_emb = pad_rel_emb[:, start:end]

        rel_logits = torch.matmul(query, k_emb.unsqueeze(0).transpose(-2, -1))
        rel_logits = F.pad(rel_logits, [0, 1])
        rel_logits = rel_logits.view([B, H, 2 * T * T])
        rel_logits = F.pad(rel_logits, [0, T - 1])
        score_loc = rel_logits.view([B, H, T + 1, 2 * T - 1])[:, :, :T, T - 1 :]  # noqa
        score_loc = score_loc / self.scale

        score = score + score_loc
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e4)
        p_attn = F.softmax(score, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)

        p_attn = F.pad(p_attn, [0, T - 1])
        p_attn = p_attn.view([B, H, T * (2 * T - 1)])
        p_attn = F.pad(p_attn, [T, 0])
        relative_weights = p_attn.view([B, H, T, 2 * T])[:, :, :, 1:]

        pad_rel_emb = F.pad(self.emb_rel_v, [0, 0, pad_length, pad_length, 0, 0])
        v_emb = pad_rel_emb[:, start:end]

        output = output + torch.matmul(relative_weights, v_emb.unsqueeze(0))

        x = output.transpose(2, 3).contiguous().view(B, C, T)

        x = self.conv_o(x)
        return x


class FFN(nn.Module):
    def __init__(self, channels, kernel_size, dropout, scale=4):
        super(FFN, self).__init__()
        self.conv_1 = torch.nn.Conv1d(
            channels, channels * scale, kernel_size, padding=kernel_size // 2
        )
        self.conv_2 = torch.nn.Conv1d(
            channels * scale, channels, kernel_size, padding=kernel_size // 2
        )
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = F.relu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


class AttentionLayer(nn.Module):
    def __init__(self, channels, num_head, dropout, window_size):
        super().__init__()
        self.attention_layer = MultiHeadAttention(
            channels, num_head, dropout, window_size
        )
        self.norm = LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        y = self.attention_layer(x, attn_mask)
        y = self.dropout(y)
        x = self.norm(x + y)
        return x


class FFNLayer(nn.Module):
    def __init__(self, channels, kernel_size, dropout, scale=4):
        super().__init__()
        self.ffn = FFN(channels, kernel_size, dropout, scale)
        self.norm = LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        y = self.ffn(x, mask)
        y = self.dropout(y)
        x = self.norm(x + y)
        return x * mask


class EncoderLayer(nn.Module):
    def __init__(self, channels, num_head, kernel_size, dropout, window_size):
        super().__init__()
        self.attention = AttentionLayer(channels, num_head, dropout, window_size)
        self.ffn = FFNLayer(channels, kernel_size=kernel_size, dropout=dropout)

    def forward(self, x, mask, attn_mask):
        x = self.attention(x, attn_mask)
        x = self.ffn(x, mask)
        return x


class PhonemeEncoder(nn.Module):
    def __init__(
        self,
        num_vocab,
        channels,
        num_head,
        num_layers,
        kernel_size,
        dropout,
        window_size,
    ):
        super().__init__()
        self.channels = channels
        self.emb = nn.Embedding(num_vocab, channels)
        torch.nn.init.normal_(self.emb.weight, 0.0, channels**-0.5)
        self.scale = math.sqrt(channels)

        self.layers = nn.ModuleList(
            [
                EncoderLayer(channels, num_head, kernel_size, dropout, window_size)
                for _ in range(num_layers)
            ]
        )
        self.postnet = nn.Conv1d(channels, channels * 2, 1)

    def forward(self, x, mask):
        x = self.emb(x) * self.scale
        x = torch.transpose(x, 1, -1)
        attn_mask = mask.unsqueeze(2) * mask.unsqueeze(-1)
        x = x * mask
        for layer in self.layers:
            x = layer(x, mask, attn_mask)
        o = self.postnet(x) * mask
        m, logs = o.split([self.channels] * 2, dim=1)
        return x, m, logs

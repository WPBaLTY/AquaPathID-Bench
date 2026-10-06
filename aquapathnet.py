"""AquaPathNet model definition (lightweight dilated 1D CNN on raw DNA).

Input : float32 (B, 5, L) one-hot of A,C,G,T,N
Output: logits (B, n_classes)

Design goals: sequence-length-agnostic (global pooling), small enough to run
per-read in real time on a laptop/Raspberry-Pi-class CPU, no k-mer preprocessing
(model sees raw bases, so it can exploit longer context than fixed-k features).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_mid: int, c_out: int, d: int):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_mid, 7, padding=3 * d, dilation=d)
        self.bn1 = nn.BatchNorm1d(c_mid)
        self.conv2 = nn.Conv1d(c_mid, c_out, 5, padding=2 * d, dilation=d)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.act = nn.GELU()
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return self.pool(x)


class AquaPathNet(nn.Module):
    def __init__(self, n_classes: int, width: float = 1.0, dropout: float = 0.2):
        super().__init__()
        c = [max(8, int(round(w))) for w in (48 * width, 64 * width,
                                             96 * width, 128 * width)]
        self.stem = nn.Sequential(
            nn.Conv1d(5, c[0], 9, padding=4), nn.BatchNorm1d(c[0]), nn.GELU())
        self.blocks = nn.ModuleList([
            ConvBlock(c[0], c[0], c[1], d=1),
            ConvBlock(c[1], c[1], c[2], d=2),
            ConvBlock(c[2], c[2], c[3], d=4),
            ConvBlock(c[3], c[3], c[3], d=8),
        ])
        self.head = nn.Sequential(
            nn.Linear(2 * c[3], 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, n_classes))

    def forward(self, x):
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        gap = x.mean(dim=2)
        gmp = x.amax(dim=2)
        return self.head(torch.cat([gap, gmp], dim=1))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


KMER5_DIM = 5 ** 5  # 3125


class AquaPathNetV2(nn.Module):
    """Dual-branch model: local dilated convs on raw bases + global 5-mer
    spectrum MLP. The conv branch carries motif/context evidence; the k-mer
    branch carries genome-wide composition evidence that a linear k-mer model
    already exploits (see baselines)."""

    def __init__(self, n_classes: int, width: float = 1.0, dropout: float = 0.2):
        super().__init__()
        self.conv = AquaPathNet(n_classes, width=width, dropout=dropout)
        del self.conv.head  # unused; v2 has its own head (keeps param count honest)
        c = max(8, int(round(128 * width)))
        self.stem_dim = 2 * c  # GAP+GMP of last conv block
        self.kmer = nn.Sequential(
            nn.LayerNorm(KMER5_DIM),
            nn.Linear(KMER5_DIM, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(self.stem_dim + 128, 256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, n_classes))

    def conv_features(self, x):
        m = self.conv
        x = m.stem(x)
        for blk in m.blocks:
            x = blk(x)
        return torch.cat([x.mean(dim=2), x.amax(dim=2)], dim=1)

    def forward(self, x, kmer):
        h = torch.cat([self.conv_features(x), self.kmer(kmer)], dim=1)
        return self.head(h)

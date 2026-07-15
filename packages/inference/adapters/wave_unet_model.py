"""Minimal Wave-U-Net style 1D U-Net for guitar solo/rhythm demix."""

from __future__ import annotations

import torch
from torch import nn


class GuitarDemixWaveUNet(nn.Module):
    """
    Time-domain U-Net: mono guitar in → solo + rhythm out.

    Checkpoint may be a full module pickle or a state_dict compatible with this layout.
    """

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, c, kernel_size=15, padding=7),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(c, c * 2, kernel_size=15, stride=2, padding=7),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv1d(c * 2, c * 2, kernel_size=15, padding=7),
            nn.ReLU(inplace=True),
        )
        self.dec_solo = nn.ConvTranspose1d(
            c * 2, 1, kernel_size=15, stride=2, padding=7, output_padding=1
        )
        self.dec_rhythm = nn.ConvTranspose1d(
            c * 2, 1, kernel_size=15, stride=2, padding=7, output_padding=1
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc1(x)
        h = self.enc2(h)
        h = self.bottleneck(h)
        return self.dec_solo(h), self.dec_rhythm(h)

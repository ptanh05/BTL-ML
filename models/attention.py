"""Coordinate Attention module."""

from __future__ import annotations

import torch
from torch import nn


class CoordAtt(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 32) -> None:
        super().__init__()
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace=True)
        self.conv_h = nn.Conv2d(mip, in_channels, 1)
        self.conv_w = nn.Conv2d(mip, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        _, _, h, w = x.shape
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w.permute(0, 1, 3, 2)))
        return identity * a_h * a_w


if __name__ == "__main__":
    for c, h in ((128, 76), (256, 38), (512, 19)):
        x = torch.randn(2, c, h, h)
        assert CoordAtt(c)(x).shape == x.shape
    print("attention.py: PASS")

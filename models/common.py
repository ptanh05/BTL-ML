"""Shared convolutional blocks used by every YOLOv4 variant."""

from __future__ import annotations

import torch
from torch import nn


class ConvBnAct(nn.Module):
    """Convolution followed by batch normalization and Hardswish."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1,
                 p: int | None = None, groups: int = 1) -> None:
        super().__init__()
        padding = k // 2 if p is None else p
        self.conv = nn.Conv2d(c1, c2, k, s, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DSConv(nn.Module):
    """Depthwise-separable 3x3 convolution used in Models 2-6."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 3, stride, 1,
            groups=in_channels, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.Hardswish(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.Hardswish(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.depthwise(x)))
        return self.act2(self.bn2(self.pointwise(x)))


def conv3x3(c1: int, c2: int, stride: int = 1, dsc: bool = True) -> nn.Module:
    """Factory for the standard or depthwise-separable 3x3 convolution."""
    return DSConv(c1, c2, stride) if dsc else ConvBnAct(c1, c2, k=3, s=stride)


class SPP(nn.Module):
    """Spatial pyramid pooling that preserves the input resolution."""

    def __init__(self, kernels: tuple[int, ...] = (5, 9, 13)) -> None:
        super().__init__()
        self.pools = nn.ModuleList(
            [nn.MaxPool2d(k, stride=1, padding=k // 2) for k in kernels]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, *(pool(x) for pool in self.pools)], dim=1)


if __name__ == "__main__":
    x = torch.randn(2, 64, 76, 76)
    y = DSConv(64, 128, stride=2)(x)
    assert y.shape == (2, 128, 38, 38)
    print("common.py: PASS")

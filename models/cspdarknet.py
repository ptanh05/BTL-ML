"""
================================================================================
MODULE: models/cspdarknet.py - BACKBONE CSPDARKNET53 (BASELINE YOLOV4)
================================================================================
MỤC ĐÍCH:
    - Cung cấp mạng Backbone CSPDarknet53 nguyên bản cho Model-1 (Baseline) và Model-2.
    - Trích xuất 3 tầng đặc trưng đa tỉ lệ C3, C4, C5 với số kênh tiêu chuẩn của YOLOv4:
      + Tầng C3: Kích thước 76x76, 256 kênh.
      + Tầng C4: Kích thước 38x38, 512 kênh.
      + Tầng C5: Kích thước 19x19, 1024 kênh.

PHÂN CÔNG TRÁCH NHIỆM (RACI):
    - Người thực thi (Responsible): Sơn (Technical Mentor)
    - Người chịu trách nhiệm (Accountable): T.A (Project Lead)
================================================================================
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from typing import Tuple


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            kernel_size // 2, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, channels, hidden_channels=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = channels
        self.block = nn.Sequential(
            BasicConv(channels, hidden_channels, 1),
            BasicConv(hidden_channels, channels, 3)
        )

    def forward(self, x):
        return x + self.block(x)


class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, first=False):
        super().__init__()
        self.downsample = BasicConv(in_channels, out_channels, 3, stride=2)
        if first:
            self.split_conv0 = BasicConv(out_channels, out_channels, 1)
            self.split_conv1 = BasicConv(out_channels, out_channels, 1)
            self.blocks_conv = nn.Sequential(
                *[ResBlock(out_channels, hidden_channels=out_channels // 2) for _ in range(num_blocks)],
                BasicConv(out_channels, out_channels, 1)
            )
            self.concat_conv = BasicConv(out_channels * 2, out_channels, 1)
        else:
            self.split_conv0 = BasicConv(out_channels, out_channels // 2, 1)
            self.split_conv1 = BasicConv(out_channels, out_channels // 2, 1)
            self.blocks_conv = nn.Sequential(
                *[ResBlock(out_channels // 2) for _ in range(num_blocks)],
                BasicConv(out_channels // 2, out_channels // 2, 1)
            )
            self.concat_conv = BasicConv(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample(x)
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)
        return self.concat_conv(torch.cat([x1, x0], dim=1))


class CSPDarknet53(nn.Module):
    """
    Backbone CSPDarknet53 tiêu chuẩn của YOLOv4 gốc:
    Trích xuất 3 tầng C3 (256 kênh), C4 (512 kênh), C5 (1024 kênh).
    """
    def __init__(self, pretrained_path: str = None):
        super().__init__()
        self.stem = BasicConv(3, 32, 3)
        self.stage1 = CSPBlock(32, 64, num_blocks=1, first=True)     # 304x304
        self.stage2 = CSPBlock(64, 128, num_blocks=2)                # 152x152
        self.stage3 = CSPBlock(128, 256, num_blocks=8)               # C3: 76x76
        self.stage4 = CSPBlock(256, 512, num_blocks=8)               # C4: 38x38
        self.stage5 = CSPBlock(512, 1024, num_blocks=4)              # C5: 19x19

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        c3 = self.stage3(x)   # (B, 256, 76, 76)
        c4 = self.stage4(c3)  # (B, 512, 38, 38)
        c5 = self.stage5(c4)  # (B, 1024, 19, 19)
        return c3, c4, c5


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print(">> [Unit Test] Running models/cspdarknet.py...")
    model = CSPDarknet53()
    x = torch.randn(1, 3, 608, 608)
    c3, c4, c5 = model(x)

    assert c3.shape == (1, 256, 76, 76), f"Lỗi shape C3: {c3.shape}"
    assert c4.shape == (1, 512, 38, 38), f"Lỗi shape C4: {c4.shape}"
    assert c5.shape == (1, 1024, 19, 19), f"Lỗi shape C5: {c5.shape}"

    num_params = sum(p.numel() for p in model.parameters())
    print(f">> Shape C3: {c3.shape} | C4: {c4.shape} | C5: {c5.shape}")
    print(f">> Số tham số CSPDarknet53: {num_params / 1e6:.2f}M params")

    loss = c3.mean() + c4.mean() + c5.mean()
    loss.backward()
    print(">> [Unit Test] Backward Gradient Test: PASS!")
    print(">> Unit Test models/cspdarknet.py: ALL PASS! [DoD M2]")

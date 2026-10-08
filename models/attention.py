"""
================================================================================
MODULE: models/attention.py - CƠ CHẾ CHÚ Ý TỌA ĐỘ (COORDINATE ATTENTION - CA)
================================================================================
MỤC ĐÍCH:
    - Hiện thực hóa cơ chế Coordinate Attention (CA) theo Mục 3.2 và Hình 3 của bài báo Sensors 2023.
    - Khắc phục nhược điểm của SE-Net (chỉ nắm bắt thông tin kênh, làm mất vị trí không gian):
      CA phân rã không gian 2D thành hai phép gom cụm 1D dọc theo trục X (ngang) và trục Y (dọc),
      giúp mô hình định vị chính xác vị trí mũ bảo hộ trên người công nhân.
    - Được nhúng vào 3 đầu ra của Backbone (C3, C4, C5) trong Model-4, Model-5, Model-6.

PHÂN CÔNG TRÁCH NHIỆM (RACI):
    - Người thực thi (Responsible): Tiến (Coordinate Attention Developer)
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


class CoordAtt(nn.Module):
    """
    Coordinate Attention Module (CA):
    Gom cụm không gian theo hai trục X và Y độc lập, bảo toàn vị trí chính xác của mục tiêu.
    Kích thước đầu ra hoàn toàn trùng khớp kích thước đầu vào (Spatial Invariance).
    """
    def __init__(self, in_channels: int, reduction: int = 32):
        super(CoordAtt, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # Pooling dọc (trục Y) -> (B, C, H, 1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # Pooling ngang (trục X) -> (B, C, 1, W)

        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace=True)

        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        n, c, h, w = x.size()

        # 1. Coordinate Information Embedding
        x_h = self.pool_h(x)                       # (N, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (N, C, 1, W) -> (N, C, W, 1)

        # 2. Concat & Non-linear transform
        y = torch.cat([x_h, x_w], dim=2)           # (N, C, H+W, 1)
        y = self.act(self.bn1(self.conv1(y)))      # (N, mip, H+W, 1)

        # 3. Split & Sigmoid Attention
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)             # Khôi phục (N, mip, 1, W)

        a_h = torch.sigmoid(self.conv_h(x_h))      # (N, C, H, 1)
        a_w = torch.sigmoid(self.conv_w(x_w))      # (N, C, 1, W)

        # 4. Residual Multiplication
        return identity * a_h * a_w


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print(">> [Unit Test] Running models/attention.py...")
    ca_128 = CoordAtt(in_channels=128)
    dummy_c3 = torch.randn(2, 128, 76, 76)
    out = ca_128(dummy_c3)
    assert out.shape == dummy_c3.shape, f"Lỗi shape CoordAtt: {out.shape} != {dummy_c3.shape}"
    print(">> [Unit Test] Spatial Invariance Test: PASS! (Shape 100% khớp)")

    loss = out.mean()
    loss.backward()
    print(">> [Unit Test] Backward Gradient Test: PASS!")
    print(">> Unit Test models/attention.py: ALL PASS! [DoD M2]")

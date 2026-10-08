"""
================================================================================
MODULE: models/pp_lcnet.py - BACKBONE MẠNG NHẸ PP-LCNET x1.0
================================================================================
MỤC ĐÍCH:
    - Xây dựng mạng Backbone nhẹ PP-LCNet phiên bản x1.0 theo bài báo gốc (Baidu / PaddleClas).
    - Cắt bỏ hoàn toàn tầng Global Average Pooling (GAP) và Fully Connected (FC) theo Mục 3.1 bài báo.
    - Trích xuất 3 tầng đặc trưng đa tỉ lệ (Feature Maps) ở các mức độ phân giải:
      + Tầng C3: Kích thước 76x76, 128 kênh (Stride 8) - Nhận diện đối tượng nhỏ.
      + Tầng C4: Kích thước 38x38, 256 kênh (Stride 16) - Nhận diện đối tượng vừa.
      + Tầng C5: Kích thước 19x19, 512 kênh (Stride 32) - Nhận diện đối tượng lớn.

PHÂN CÔNG TRÁCH NHIỆM (RACI):
    - Người thực thi (Responsible): Hưng (Backbone Developer)
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


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation Module: Tái cân chỉnh trọng số giữa các kênh.
    AdaptiveAvgPool2d -> Conv 1x1 giảm kênh (r=4) -> ReLU -> Conv 1x1 tăng kênh -> Hardsigmoid.
    """
    def __init__(self, c: int, reduction: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, c // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(c // reduction, c, kernel_size=1)
        self.gate = nn.Hardsigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(self.fc2(self.relu(self.fc1(self.pool(x)))))


class DepthSepConv(nn.Module):
    """
    Khối cơ bản của PP-LCNet: DWConv(k x k) -> [SEModule tùy chọn] -> PWConv(1 x 1).
    Lưu ý: Không dùng residual shortcut theo đúng thiết kế nguyên bản của PP-LCNet.
    """
    def __init__(self, c1: int, c2: int, k: int = 3, stride: int = 1, use_se: bool = False):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=k, stride=stride, padding=k // 2, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.Hardswish(inplace=True)
        )
        self.se = SEModule(c1) if use_se else nn.Identity()
        self.pw = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.Hardswish(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.se(self.dw(x)))


# Cấu hình chuẩn PP-LCNet x1.0: (kernel, c_in, c_out, stride, use_se)
CFG = {
    "stage2": [(3, 16, 32, 1, False)],                                       # 304x304x32
    "stage3": [(3, 32, 64, 2, False), (3, 64, 64, 1, False)],                # 152x152x64
    "stage4": [(3, 64, 128, 2, False), (3, 128, 128, 1, False)],             # C3: 76x76x128
    "stage5": [(3, 128, 256, 2, False)] + [(5, 256, 256, 1, False)] * 5,     # C4: 38x38x256
    "stage6": [(5, 256, 512, 2, True), (5, 512, 512, 1, True)],              # C5: 19x19x512 (5x5 + SE)
}


def _make_stage(cfg):
    return nn.Sequential(*[DepthSepConv(c1, c2, k, s, se) for k, c1, c2, s, se in cfg])


class PPLCNet(nn.Module):
    """
    Backbone PP-LCNet x1.0 tối ưu cho di động/biên, trích xuất 3 mức đặc trưng C3, C4, C5.
    Cắt bỏ hoàn toàn GAP và FC layers theo đúng Mục 3.1 của bài báo.
    """
    def __init__(self, pretrained_path: str = None):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True)
        )
        self.stage2 = _make_stage(CFG["stage2"])
        self.stage3 = _make_stage(CFG["stage3"])
        self.stage4 = _make_stage(CFG["stage4"])
        self.stage5 = _make_stage(CFG["stage5"])
        self.stage6 = _make_stage(CFG["stage6"])

        if pretrained_path:
            self.load_pretrained(pretrained_path)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)                  # (B, 16, 304, 304)
        x = self.stage3(self.stage2(x))   # (B, 64, 152, 152)
        c3 = self.stage4(x)               # (B, 128, 76, 76)
        c4 = self.stage5(c3)              # (B, 256, 38, 38)
        c5 = self.stage6(c4)              # (B, 512, 19, 19)
        return c3, c4, c5

    def load_pretrained(self, path: str) -> None:
        state_dict = torch.load(path, map_location="cpu")
        self.load_state_dict(state_dict, strict=False)
        print(f">> Đã nạp thành công trọng số PP-LCNet từ {path}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print(">> [Unit Test] Running models/pp_lcnet.py...")
    model = PPLCNet()
    dummy_input = torch.randn(2, 3, 608, 608)
    c3, c4, c5 = model(dummy_input)

    assert c3.shape == (2, 128, 76, 76), f"Lỗi shape C3: {c3.shape}"
    assert c4.shape == (2, 256, 38, 38), f"Lỗi shape C4: {c4.shape}"
    assert c5.shape == (2, 512, 19, 19), f"Lỗi shape C5: {c5.shape}"

    num_params = sum(p.numel() for p in model.parameters())
    print(f">> Shape C3: {c3.shape} | C4: {c4.shape} | C5: {c5.shape}")
    print(f">> Số tham số Backbone PP-LCNet: {num_params / 1e6:.2f}M params")

    loss = c3.mean() + c4.mean() + c5.mean()
    loss.backward()
    print(">> [Unit Test] Backward Gradient Test: PASS!")
    print(">> Unit Test models/pp_lcnet.py: ALL PASS! [DoD M2]")

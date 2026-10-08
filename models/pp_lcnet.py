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
    - Tham vấn (Consulted): Sơn (Technical Mentor)
    - Nhận bàn giao (Informed): Tiến (Tiến nhận C3, C4, C5 để đưa qua Coordinate Attention)

GIAO DIỆN & ĐẶC TẢ TENSOR:
    1. Input:
       - Tensor ảnh `x`: Kích thước (B, 3, 608, 608), kiểu torch.float32.
    2. Output:
       - Tuple gồm 3 tensor: `(C3, C4, C5)`
         + C3: (B, 128, 76, 76)
         + C4: (B, 256, 38, 38)
         + C5: (B, 512, 19, 19)

CẤU TRÚC CHI TIẾT CÁC TẦNG THEO PP-LCNET x1.0:
    - Stem: Conv 3x3, stride=2, padding=1 (3 -> 16 kênh) -> (B, 16, 304, 304).
    - Stage 2: 1 khối DepthSepConv k=3, stride=1 (16 -> 32 kênh).
    - Stage 3: 2 khối DepthSepConv k=3, stride=2 rồi stride=1 (32 -> 64 kênh) -> 152x152.
    - Stage 4: 2 khối DepthSepConv k=3, stride=2 rồi stride=1 (64 -> 128 kênh) -> Ra C3 (76x76x128).
    - Stage 5: 1 khối k=3 stride=2 + 5 khối k=5 stride=1 (128 -> 256 kênh) -> Ra C4 (38x38x256).
    - Stage 6: 2 khối k=5 stride=2 rồi stride=1 tích hợp SEModule (256 -> 512 kênh) -> Ra C5 (19x19x512).

TIÊU CHÍ NGHIỆM THU (DoD M2):
    [ ] Khi truyền dummy tensor `torch.randn(2, 3, 608, 608)`, shape đầu ra C3, C4, C5 phải đúng 100%.
    [ ] Số lượng tham số của Backbone đạt xấp xỉ ~1.88M - 2.0M tham số (rất nhẹ so với CSPDarknet53 ~27M).
    [ ] Tích hợp cơ chế load pretrained weights (chuyển đổi từ PaddleClas nếu có) thông qua hàm `load_pretrained()`.

BẪY LỖI KỸ THUẬT CẦN TRÁNH:
    - Bẫy 1: Sai lệch số kênh ở C3, C4, C5 sẽ làm gãy interface chuyển tiếp sang Coordinate Attention của Tiến và PB Module của T.A.
    - Bẫy 2: Hình 1 và 2 trong bài báo Sensors 2023 vẽ thu gọn số khối lặp; Hưng PHẢI tuân thủ theo chuẩn PP-LCNet x1.0 để khớp chuẩn 128/256/512 kênh.
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
        # TODO: Hưng khởi tạo AdaptiveAvgPool2d(1), Conv2d 1x1, ReLU, Conv2d 1x1, Hardsigmoid
        raise NotImplementedError("Cần được Hưng cài đặt SEModule.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Cần được Hưng cài đặt forward của SEModule.")


class DepthSepConv(nn.Module):
    """
    Khối cơ bản của PP-LCNet: DWConv(k x k) -> [SEModule tùy chọn] -> PWConv(1 x 1).
    Lưu ý: Không dùng residual shortcut theo đúng thiết kế nguyên bản của PP-LCNet.
    """
    def __init__(self, c1: int, c2: int, k: int = 3, stride: int = 1, use_se: bool = False):
        super().__init__()
        # TODO: Hưng khởi tạo DWConv, BatchNorm, Hardswish, SEModule (nếu use_se), PWConv, BatchNorm, Hardswish
        raise NotImplementedError("Cần được Hưng cài đặt DepthSepConv.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Cần được Hưng cài đặt forward của DepthSepConv.")


class PPLCNet(nn.Module):
    """
    Backbone PP-LCNet x1.0 tối ưu cho di động/biên, trích xuất 3 mức đặc trưng C3, C4, C5.
    Cắt bỏ hoàn toàn GAP và FC layers theo đúng Mục 3.1 của bài báo.
    """
    def __init__(self, pretrained_path: str = None):
        super().__init__()
        # TODO: Hưng khởi tạo stem, stage2, stage3, stage4, stage5, stage6 theo đặc tả trên
        raise NotImplementedError("Cần được Hưng cài đặt PPLCNet.")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Đầu vào: (B, 3, 608, 608)
        Đầu ra: Tuple(C3, C4, C5) tương ứng (B, 128, 76, 76), (B, 256, 38, 38), (B, 512, 19, 19)
        """
        raise NotImplementedError("Cần được Hưng cài đặt forward của PPLCNet.")

    def load_pretrained(self, path: str) -> None:
        """
        Nạp trọng số ImageNet tiền huấn luyện của PP-LCNet.
        """
        raise NotImplementedError("Cần được Hưng cài đặt load_pretrained.")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(">> [TODO] models/pp_lcnet.py: Hưng cần hoàn thiện PPLCNet để vượt qua Unit Test.")

"""
================================================================================
MODULE: models/cspdarknet.py - BACKBONE CSPDARKNET53 (BASELINE YOLOV4)
================================================================================
MỤC ĐÍCH:
    - Cung cấp mạng Backbone CSPDarknet53 nguyên bản cho Model-1 (Baseline) và Model-2.
    - Sơn tích hợp từ repo PyTorch YOLOv4 uy tín (như repo Tianxiaomo hoặc WongKinYiu).
    - Trích xuất 3 tầng đặc trưng đa tỉ lệ C3, C4, C5 với số kênh tiêu chuẩn của YOLOv4:
      + Tầng C3: Kích thước 76x76, 256 kênh.
      + Tầng C4: Kích thước 38x38, 512 kênh.
      + Tầng C5: Kích thước 19x19, 1024 kênh.

PHÂN CÔNG TRÁCH NHIỆM (RACI):
    - Người thực thi (Responsible): Sơn (Technical Mentor)
    - Người chịu trách nhiệm (Accountable): T.A (Project Lead)
    - Tham vấn (Consulted): Hưng (so sánh hiệu năng tham số với PP-LCNet)
    - Nhận bàn giao (Informed): Toàn đội (dùng để train Model-1 và Model-2)

GIAO DIỆN & ĐẶC TẢ TENSOR:
    1. Input:
       - Tensor ảnh `x`: Kích thước (B, 3, 608, 608), kiểu torch.float32.
    2. Output:
       - Tuple gồm 3 tensor: `(C3, C4, C5)`
         + C3: (B, 256, 76, 76)
         + C4: (B, 512, 38, 38)
         + C5: (B, 1024, 19, 19)

HƯỚNG DẪN CÀI ĐẶT:
    - Tận dụng cấu trúc Cross Stage Partial Network (CSPNet) với hàm kích hoạt Mish.
    - Cắt bỏ phần pooling và phân loại 1000 lớp ImageNet, chỉ giữ lại phần trích xuất feature maps.
    - Hỗ trợ tải trọng số pretrained qua hàm `load_pretrained(weights_path)`.

TIÊU CHÍ NGHIỆM THU (DoD M2):
    [ ] Truyền dummy tensor `(2, 3, 608, 608)` nhận đúng 3 tensor: (2, 256, 76, 76), (2, 512, 38, 38), (2, 1024, 19, 19).
    [ ] Số lượng tham số khoảng ~27.6M tham số.
    [ ] Kiểm tra khả năng đóng băng tham số (Freeze backbone) phục vụ giai đoạn huấn luyện 1.

BẪY LỖI KỸ THUẬT CẦN TRÁNH:
    - Bẫy 1: Số kênh của CSPDarknet53 (256, 512, 1024) gấp đôi PP-LCNet (128, 256, 512). Cần cấu hình kênh ở Neck cho phù hợp từng model!
    - Bẫy 2: CSPDarknet53 tốn nhiều VRAM GPU hơn nhiều so với PP-LCNet -> Bắt buộc giám sát OOM khi train trên Colab T4.
================================================================================
"""

import torch
import torch.nn as nn
from typing import Tuple


class CSPDarknet53(nn.Module):
    """
    Backbone CSPDarknet53 tiêu chuẩn của YOLOv4 gốc.
    """
    def __init__(self, pretrained_path: str = None):
        super().__init__()
        # HƯỚNG DẪN: Sơn tích hợp các tầng Conv, ResBlock, CSPBlocks tại đây
        pass

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Đầu vào: (B, 3, 608, 608)
        Đầu ra: Tuple(C3, C4, C5) tương ứng (B, 256, 76, 76), (B, 512, 38, 38), (B, 1024, 19, 19)
        """
        raise NotImplementedError("Cần được Sơn cài đặt forward của CSPDarknet53.")

    def load_pretrained(self, weights_path: str) -> None:
        """
        Nạp trọng số YOLOv4 / ImageNet pretrained.
        """
        raise NotImplementedError("Cần được Sơn cài đặt load_pretrained.")


if __name__ == "__main__":
    print("[TODO] models/cspdarknet.py: Chạy unit test kiểm tra shape C3, C4, C5.")

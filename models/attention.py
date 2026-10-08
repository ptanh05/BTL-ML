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
    - Tham vấn (Consulted): Sơn (Technical Mentor)
    - Nhận bàn giao (Informed): T.A (T.A nhúng 3 khối CA vào trước PB Module trong yolo.py)

GIAO DIỆN & ĐẶC TẢ TENSOR:
    1. Input:
       - Tensor `x`: (B, C, H, W)
    2. Output:
       - Tensor `out`: (B, C, H, W) - KÍCH THƯỚC ĐẦU RA PHẢI HOÀN TOÀN TRÙNG KHỚP ĐẦU VÀO!

CÔNG THỨC & THUẬT TOÁN (Hình 3 bài báo):
    1. Coordinate Information Embedding:
       - Pooling theo chiều cao Y: `pool_h = AdaptiveAvgPool2d((None, 1))` -> tensor shape (B, C, H, 1)
       - Pooling theo chiều rộng X: `pool_w = AdaptiveAvgPool2d((1, None))` -> permute sang (B, C, W, 1)
    2. Concatenation & Non-linear Transform:
       - Nối 2 tensor lại: `cat([x_h, x_w], dim=2)` -> shape (B, C, H+W, 1)
       - Nén số kênh: Conv2d 1x1 (C -> mip = max(8, C // reduction)) với reduction=32
       - BatchNorm2d(mip) -> Hardswish()
    3. Split & Attention Generation:
       - Tách lại theo dim=2 thành `x_h` (B, mip, H, 1) và `x_w` (B, mip, W, 1)
       - Khôi phục trục `x_w`: permute sang (B, mip, 1, W)
       - Tái lập số kênh: `conv_h` 1x1 (mip -> C) và `conv_w` 1x1 (mip -> C)
       - Tạo trọng số chú ý: `a_h = sigmoid(conv_h(x_h))` và `a_w = sigmoid(conv_w(x_w))`
    4. Residual Weighting:
       - `out = identity * a_h * a_w`

TIÊU CHÍ NGHIỆM THU (DoD M2):
    [ ] Kiểm tra tính bất biến không gian (Spatial Invariance): `out.shape == dummy_input.shape`.
    [ ] Thử nghiệm với cả 3 kích thước:
        - C3: (2, 128, 76, 76)
        - C4: (2, 256, 38, 38)
        - C5: (2, 512, 19, 19)
    [ ] Khối `if __name__ == '__main__':` chạy unit test thành công và in thông báo PASS.

BẪY LỖI KỸ THUẬT CẦN TRÁNH:
    - Bẫy 1: Sai chiều khi permute trục W -> dẫn tới phép nhân ma trận broadcasting sai kích thước.
    - Bẫy 2: Thiếu giá trị chặn `mip = max(8, in_channels // reduction)`: Khi kênh nhỏ có thể bị nén về 0 hoặc quá bé gây tiêu biến đặc trưng.
================================================================================
"""

import torch
import torch.nn as nn


class CoordAtt(nn.Module):
    """
    Coordinate Attention Module (CA): Chú ý không gian theo hai trục X và Y độc lập.
    """
    def __init__(self, in_channels: int, reduction: int = 32):
        super(CoordAtt, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        # HƯỚNG DẪN: Tiến khai báo pool_h, pool_w, conv1, bn1, act, conv_h, conv_w theo đặc tả trên
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Đầu vào: x (B, C, H, W)
        Đầu ra: out (B, C, H, W)
        """
        raise NotImplementedError("Cần được Tiến cài đặt forward của CoordAtt.")


if __name__ == "__main__":
    print("[TODO] models/attention.py: Chạy unit test kiểm tra tính bất biến không gian của CoordAtt.")

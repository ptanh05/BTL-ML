"""
================================================================================
MODULE: models/pb_module.py - PB MODULE (SPP + PANET + BIFPN)
================================================================================
MỤC ĐÍCH:
    - Xây dựng mạng dung hợp đặc trưng đa tỉ lệ PB Module (Mục 3.3, Hình 1 & Hình 4 bài báo).
    - Kết hợp sức mạnh của 3 kiến trúc:
      1. SPP (Spatial Pyramid Pooling): Mở rộng trường tiếp nhận (Receptive Field) trên tầng C5.
      2. PANet (Path Aggregation Network): Dẫn truyền thông tin đặc trưng 2 chiều top-down và bottom-up.
      3. BiFPN (Bidirectional Feature Pyramid Network): Hợp nhất có trọng số (Weighted Feature Fusion)
         và bổ sung cạnh dư cùng tầng (same-level residual connection) ở tầng giữa.
    - Toàn bộ phép tích chập 3x3 trong Neck đều sử dụng DSConv (dsc=True) để tối ưu dung lượng và tốc độ.
    - Sử dụng trong Model-5 và Model-6. Với Model-1 đến Model-4, chỉ sử dụng PANet tiêu chuẩn.

PHÂN CÔNG TRÁCH NHIỆM (RACI):
    - Người thực thi & Chịu trách nhiệm (R & A): T.A (Project Lead)
    - Tham vấn (Consulted): Sơn (Technical Mentor)
    - Nhận bàn giao (Informed): Tú (Tú nhận 3 tensor đầu ra để đưa vào YOLOHead)

GIAO DIỆN & ĐẶC TẢ TENSOR:
    1. Input:
       - 3 feature maps (p3, p4, p5) nhận từ Attention hoặc Backbone:
         + p3: (B, 128, 76, 76)
         + p4: (B, 256, 38, 38)
         + p5: (B, 512, 19, 19)
    2. Output:
       - 3 feature maps dung hợp cao cấp bàn giao cho Head:
         + o3: (B, bi_ch, 76, 76)  (mặc định bi_ch = 128)
         + o4: (B, bi_ch, 38, 38)
         + o5: (B, bi_ch, 19, 19)

CÔNG THỨC & THUẬT TOÁN:
    1. Khối SPP:
       - MaxPool2d với kernel k=(5, 9, 13), stride=1, padding=k//2.
       - Ghép nối concat tensor gốc + 3 kết quả pooling -> số kênh tăng gấp 4 lần.
    2. Hợp nhất BiFPN có trọng số (Fast Normalized Fusion):
       - O = sum(w_i * I_i) / (sum(w_i) + eps) với w_i >= 0 là trọng số học được (Parameter).
    3. Cạnh dư cùng tầng (Residual Shortcut):
       - Nối trực tiếp đầu vào x4 sang đầu ra o4 ở tầng giữa để giữ lại thông tin ngữ nghĩa gốc.

TIÊU CHÍ NGHIỆM THU (DoD M2):
    [ ] Chạy unit test độc lập: `python models/pb_module.py` PASS thành công.
    [ ] Đầu ra của PANet và PBModule đúng kích thước phân giải không gian (76x76, 38x38, 19x19).
    [ ] Tính toán đạo hàm `loss.backward()` trơn tru, không phát sinh lỗi gradient hay NaN.

BẪY LỖI KỸ THUẬT CẦN TRÁNH:
    - Bẫy 1: Số kênh của BiFPN (bi_ch) ảnh hưởng trực tiếp đến kích thước file model:
      Cần tinh chỉnh số kênh để Model-5 và Model-6 đạt đúng dung lượng mục tiêu ~41.88 MB.
    - Bẫy 2: Trọng số học được w của BiFPN phải đi qua hàm ReLU `w = torch.relu(self.w)` để đảm bảo w >= 0 và tránh chia cho 0 với hằng số `eps = 1e-4`.
================================================================================
"""

import torch
import torch.nn as nn
from typing import Tuple, List
from models.common import ConvBnAct, conv3x3


class SPP(nn.Module):
    """
    Spatial Pyramid Pooling: Max-pooling đa tỉ lệ (5, 9, 13) mở rộng receptive field.
    """
    def __init__(self, pool_sizes: Tuple[int, ...] = (5, 9, 13)):
        super().__init__()
        # HƯỚNG DẪN: T.A khởi tạo nn.ModuleList chứa các lớp MaxPool2d
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Đầu vào: x (B, C, H, W)
        Đầu ra: cat([x, pool1, pool2, pool3], dim=1) -> (B, 4C, H, W)
        """
        raise NotImplementedError("Cần được T.A cài đặt SPP.")


class WeightedAdd(nn.Module):
    """
    Fast Normalized Fusion của BiFPN:
    out = sum(w_i * x_i) / (sum(w_i) + eps) với w_i >= 0 học được.
    """
    def __init__(self, n: int, eps: float = 1e-4):
        super().__init__()
        # HƯỚNG DẪN: T.A khai báo self.w = nn.Parameter(torch.ones(n))
        pass

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError("Cần được T.A cài đặt WeightedAdd.")


class PANet(nn.Module):
    """
    Path Aggregation Network (PANet) tiêu chuẩn có SPP ở tầng C5:
    Dùng cho Model-1, Model-2, Model-3, Model-4.
    """
    def __init__(self, ch: Tuple[int, int, int] = (128, 256, 512), dsc: bool = True):
        super().__init__()
        # HƯỚNG DẪN: T.A khởi tạo các nhánh Top-down và Bottom-up
        pass

    def forward(
        self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Cần được T.A cài đặt PANet.")


class BiFPN(nn.Module):
    """
    Bidirectional Feature Pyramid Network (BiFPN 3 tầng):
    Hợp nhất có trọng số và bổ sung cạnh dư cùng tầng (same-level shortcut).
    """
    def __init__(self, in_ch: Tuple[int, int, int], ch: int = 128, dsc: bool = True):
        super().__init__()
        # HƯỚNG DẪN: T.A khởi tạo các node tổng hợp có trọng số WeightedAdd và conv3x3
        pass

    def forward(
        self, x3: torch.Tensor, x4: torch.Tensor, x5: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Cần được T.A cài đặt BiFPN.")


class PBModule(nn.Module):
    """
    PB Module hoàn chỉnh: PANet (kèm SPP) -> BiFPN.
    Dùng cho Model-5 và Model-6.
    """
    def __init__(
        self,
        ch: Tuple[int, int, int] = (128, 256, 512),
        bi_ch: int = 128,
        dsc: bool = True
    ):
        super().__init__()
        # HƯỚNG DẪN: T.A kết nối PANet và BiFPN tuần tự
        pass

    def forward(
        self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Cần được T.A cài đặt PBModule.")


if __name__ == "__main__":
    print("[TODO] models/pb_module.py: Chạy unit test kiểm tra forward/backward PANet và PBModule.")

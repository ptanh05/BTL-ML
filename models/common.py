"""
================================================================================
MODULE: models/common.py - CÁC KHỐI TÍCH CHẬP CƠ BẢN & DSCONV
================================================================================
MỤC ĐÍCH:
    - Cung cấp các khối tích chập dùng chung cho toàn bộ mạng YOLOv4 và các biến thể.
    - Lập trình khối Tích chập sâu tách biệt (Depthwise Separable Convolution - DSConv):
      Thay thế Conv 3x3 truyền thống nhằm giảm mạnh số lượng tham số và phép tính FLOPs (giảm ~8-9 lần).
    - Cung cấp hàm chuyển đổi `conv3x3(c1, c2, stride=1, dsc=True)`:
      + Nếu `dsc=True`: Trả về `DSConv` (Dùng cho Model-2, Model-3, Model-4, Model-5, Model-6).
      + Nếu `dsc=False`: Trả về `ConvBnAct` tiêu chuẩn (Dùng cho Model-1 Baseline).

PHÂN CÔNG TRÁCH NHIỆM (RACI):
    - Người thực thi (Responsible): Hưng (Backbone & DSC Developer)
    - Người chịu trách nhiệm (Accountable): T.A (Project Lead)
    - Tham vấn (Consulted): Sơn (Technical Mentor)
    - Nhận bàn giao (Informed): Toàn đội (Tiến, T.A, Tú dùng DSConv trong Neck và Head)

GIAO DIỆN & ĐẶC TẢ TENSOR:
    1. Input:
       - Tensor `x`: (B, C_in, H, W)
    2. Output:
       - Tensor `out`: (B, C_out, H // stride, W // stride)

CÔNG THỨC & THUẬT TOÁN DSCONV:
    DSConv gồm 2 bước tách biệt tuần tự:
    1. Depthwise Convolution (DWConv):
       - Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
       - BatchNorm2d(in_channels)
       - Activation: Hardswish()
    2. Pointwise Convolution (PWConv):
       - Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
       - BatchNorm2d(out_channels)
       - Activation: Hardswish()

TIÊU CHÍ NGHIỆM THU (DoD M2):
    [ ] Chạy unit test độc lập: `python models/common.py` không phát sinh lỗi.
    [ ] Shape đầu ra kiểm tra với dummy tensor (2, 64, 76, 76) đưa qua DSConv(64, 128, stride=2) phải ra đúng (2, 128, 38, 38).
    [ ] Khối `ConvBnAct` chạy mượt mà với cả conv 1x1 và conv 3x3 chuẩn.
    [ ] Backward pass tính gradient hữu hạn, không có NaN/Inf.

BẪY LỖI KỸ THUẬT CẦN TRÁNH:
    - Bẫy 1: Quên tham số `groups=in_channels` trong Depthwise Conv khiến nó biến thành Standard Conv!
    - Bẫy 2: Dùng ReLU thay vì Hardswish: Bài báo chỉ định hàm kích hoạt Hardswish cho kiến trúc cải tiến.
================================================================================
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn


class ConvBnAct(nn.Module):
    """
    Khối tích chập tiêu chuẩn: Conv2d(k x k) -> BatchNorm2d -> Hardswish.
    Dùng cho conv 1x1 và conv 3x3 truyền thống (Model-1 Baseline).
    """
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1):
        super().__init__()
        # TODO: Hưng khởi tạo Conv2d (padding=k//2, bias=False), BatchNorm2d, Hardswish
        raise NotImplementedError("Cần được Hưng (Backbone Developer) cài đặt ConvBnAct.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Cần được Hưng cài đặt forward của ConvBnAct.")


class DSConv(nn.Module):
    """
    Depthwise Separable Convolution (DSConv):
    Gồm Depthwise Conv 3x3 (lọc từng kênh) + Pointwise Conv 1x1 (kết hợp các kênh).
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # TODO: Hưng khởi tạo Depthwise Conv (groups=in_channels) và Pointwise Conv (1x1)
        raise NotImplementedError("Cần được Hưng (Backbone Developer) cài đặt DSConv.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Cần được Hưng cài đặt forward của DSConv.")


def conv3x3(c1: int, c2: int, stride: int = 1, dsc: bool = True) -> nn.Module:
    """
    Hàm nhà máy (Factory) chọn lựa giữa DSConv và Standard Conv 3x3.
    - dsc=True  -> DSConv (Model-2..6)
    - dsc=False -> ConvBnAct 3x3 chuẩn (Model-1)
    """
    # TODO: Hưng trả về DSConv nếu dsc else ConvBnAct k=3
    raise NotImplementedError("Cần được Hưng cài đặt hàm conv3x3.")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(">> [TODO] models/common.py: Hưng cần hoàn thiện ConvBnAct, DSConv và conv3x3 để vượt qua Unit Test.")

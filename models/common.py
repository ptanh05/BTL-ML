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
================================================================================
"""

import torch
import torch.nn as nn


class ConvBnAct(nn.Module):
    """
    Khối tích chập tiêu chuẩn: Conv2d(k x k) -> BatchNorm2d -> Hardswish.
    Dùng cho conv 1x1 và conv 3x3 truyền thống (Model-1 Baseline).
    """
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DSConv(nn.Module):
    """
    Depthwise Separable Convolution (DSConv):
    Gồm Depthwise Conv 3x3 (lọc từng kênh) + Pointwise Conv 1x1 (kết hợp các kênh).
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # 1. Depthwise Convolution: lọc đặc trưng từng kênh độc lập
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3,
            stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.Hardswish(inplace=True)

        # 2. Pointwise Convolution: kết hợp thông tin giữa các kênh
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1,
            stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.Hardswish(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.depthwise(x)))
        x = self.act2(self.bn2(self.pointwise(x)))
        return x


def conv3x3(c1: int, c2: int, stride: int = 1, dsc: bool = True) -> nn.Module:
    """
    Hàm nhà máy (Factory) chọn lựa giữa DSConv và Standard Conv 3x3.
    - dsc=True  -> DSConv (Model-2..6)
    - dsc=False -> ConvBnAct 3x3 chuẩn (Model-1)
    """
    return DSConv(c1, c2, stride=stride) if dsc else ConvBnAct(c1, c2, k=3, s=stride)


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print(">> [Unit Test] Running models/common.py...")
    x = torch.randn(2, 64, 76, 76)

    # Test ConvBnAct 1x1
    c1x1 = ConvBnAct(64, 128, k=1)
    out1 = c1x1(x)
    assert out1.shape == (2, 128, 76, 76), f"Lỗi shape ConvBnAct 1x1: {out1.shape}"

    # Test DSConv stride=1
    dsc_s1 = DSConv(64, 128, stride=1)
    out_dsc1 = dsc_s1(x)
    assert out_dsc1.shape == (2, 128, 76, 76), f"Lỗi shape DSConv s=1: {out_dsc1.shape}"

    # Test DSConv stride=2
    dsc_s2 = DSConv(64, 128, stride=2)
    out_dsc2 = dsc_s2(x)
    assert out_dsc2.shape == (2, 128, 38, 38), f"Lỗi shape DSConv s=2: {out_dsc2.shape}"

    # Test backward pass
    loss = out_dsc2.mean()
    loss.backward()

    print(">> Unit Test models/common.py: ALL PASS! [DoD M2]")

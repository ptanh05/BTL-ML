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
================================================================================
"""

import sys
from pathlib import Path

# Đảm bảo đường dẫn gốc luôn có trong sys.path khi chạy file trực tiếp
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from typing import Tuple, List
from models.common import ConvBnAct, conv3x3


def make_n_conv(c_in: int, c: int, n: int, dsc: bool = True) -> nn.Sequential:
    """
    Tạo chuỗi n tầng tích chập xen kẽ 1x1 và 3x3 (n = 3 hoặc 5).
    Tương ứng với các khối 'DSConv x3' và 'DSConv x5' trong Hình 1 bài báo.
    Đầu ra luôn có c kênh.
    """
    layers = []
    for i in range(n):
        if i % 2 == 0:
            layers.append(ConvBnAct(c_in if i == 0 else 2 * c, c, k=1))
        else:
            layers.append(conv3x3(c, 2 * c, dsc=dsc))
    return nn.Sequential(*layers)


class SPP(nn.Module):
    """
    Spatial Pyramid Pooling (SPP):
    Max-pooling đa tỉ lệ với kích thước kernel (5, 9, 13) mở rộng trường tiếp nhận (Receptive Field).
    Đầu vào: (B, C, H, W) -> Đầu ra: (B, 4C, H, W).
    """
    def __init__(self, pool_sizes: Tuple[int, ...] = (5, 9, 13)):
        super().__init__()
        self.pools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
            for k in pool_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x] + [p(x) for p in self.pools], dim=1)


class PANet(nn.Module):
    """
    Path Aggregation Network (PANet) tiêu chuẩn có khối SPP ở tầng C5:
    Dùng cho Model-1, Model-2, Model-3, Model-4 (và là bước 1 trong PBModule).
    Đầu vào: C3, C4, C5 (kênh c3, c4, c5).
    Đầu ra: o3, o4, o5 (kênh c3//2, c4//2, c5//2).
    """
    def __init__(self, ch: Tuple[int, int, int] = (128, 256, 512), dsc: bool = True):
        super().__init__()
        c3, c4, c5 = ch
        h3, h4, h5 = c3 // 2, c4 // 2, c5 // 2  # ví dụ: 64, 128, 256

        # Nhánh C5: DSConv x3 -> SPP -> Concat + DSConv x3
        self.c5_pre = make_n_conv(c5, h5, 3, dsc=dsc)
        self.spp = SPP(pool_sizes=(5, 9, 13))
        self.c5_post = make_n_conv(h5 * 4, h5, 3, dsc=dsc)

        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        # Nhánh Top-down (Từ sâu về nông)
        self.td5_red = ConvBnAct(h5, h4, k=1)
        self.c4_lat = ConvBnAct(c4, h4, k=1)
        self.td4 = make_n_conv(2 * h4, h4, 5, dsc=dsc)

        self.td4_red = ConvBnAct(h4, h3, k=1)
        self.c3_lat = ConvBnAct(c3, h3, k=1)
        self.td3 = make_n_conv(2 * h3, h3, 5, dsc=dsc)

        # Nhánh Bottom-up (Từ nông lên sâu)
        self.down3 = conv3x3(h3, h4, stride=2, dsc=dsc)
        self.bu4 = make_n_conv(2 * h4, h4, 5, dsc=dsc)

        self.down4 = conv3x3(h4, h5, stride=2, dsc=dsc)
        self.bu5 = make_n_conv(2 * h5, h5, 5, dsc=dsc)

        self.out_channels = (h3, h4, h5)

    def forward(
        self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # C5 qua SPP
        p5_td = self.c5_post(self.spp(self.c5_pre(p5)))

        # Top-down fusion
        p4_cat = torch.cat([self.c4_lat(p4), self.up(self.td5_red(p5_td))], dim=1)
        p4_td = self.td4(p4_cat)

        p3_cat = torch.cat([self.c3_lat(p3), self.up(self.td4_red(p4_td))], dim=1)
        o3 = self.td3(p3_cat)

        # Bottom-up fusion
        o4_cat = torch.cat([self.down3(o3), p4_td], dim=1)
        o4 = self.bu4(o4_cat)

        o5_cat = torch.cat([self.down4(o4), p5_td], dim=1)
        o5 = self.bu5(o5_cat)

        return o3, o4, o5


class WeightedAdd(nn.Module):
    """
    Fast Normalized Fusion của BiFPN (Công thức chuẩn EfficientDet / Sensors 2023):
    out = sum(w_i * x_i) / (sum(w_i) + eps)
    với w_i >= 0 là trọng số học được (Parameter), khởi tạo bằng 1.0.
    """
    def __init__(self, n: int, eps: float = 1e-4):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n, dtype=torch.float32))
        self.eps = eps

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        w = torch.relu(self.w)
        weights_normalized = w / (w.sum() + self.eps)
        return sum(wi * xi for wi, xi in zip(weights_normalized, xs))


class BiFPN(nn.Module):
    """
    BiFPN 3 tầng cải tiến:
    - Loại bỏ các nút chỉ có 1 cạnh vào (tiết kiệm tính toán).
    - Hợp nhất đặc trưng có trọng số thông qua WeightedAdd.
    - Bổ sung cạnh dư cùng tầng (same-level residual connection) ở tầng giữa (x4 -> o4).
    """
    def __init__(self, in_ch: Tuple[int, int, int], ch: int = 128, dsc: bool = True):
        super().__init__()
        self.proj = nn.ModuleList([ConvBnAct(c, ch, k=1) for c in in_ch])
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

        # Trọng số hợp nhất cho từng nút
        self.w_td4 = WeightedAdd(2)
        self.w_o3 = WeightedAdd(2)
        self.w_o4 = WeightedAdd(3)  # Nhận 3 đầu vào: x4 (residual), td4, down(o3)
        self.w_o5 = WeightedAdd(2)

        # Các tầng tích chập tinh chỉnh đặc trưng
        self.c_td4 = conv3x3(ch, ch, dsc=dsc)
        self.c_o3 = conv3x3(ch, ch, dsc=dsc)
        self.c_o4 = conv3x3(ch, ch, dsc=dsc)
        self.c_o5 = conv3x3(ch, ch, dsc=dsc)

        self.out_channels = (ch, ch, ch)

    def forward(
        self, x3: torch.Tensor, x4: torch.Tensor, x5: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Chiếu về cùng số kênh ch
        x3, x4, x5 = [m(x) for m, x in zip(self.proj, (x3, x4, x5))]

        # Top-down pathway
        td4 = self.c_td4(self.w_td4([x4, self.up(x5)]))
        o3 = self.c_o3(self.w_o3([x3, self.up(td4)]))

        # Bottom-up pathway kèm CẠNH DƯ CÙNG TẦNG x4
        o4 = self.c_o4(self.w_o4([x4, td4, self.down(o3)]))
        o5 = self.c_o5(self.w_o5([x5, self.down(o4)]))

        return o3, o4, o5


class PBModule(nn.Module):
    """
    PB Module hoàn chỉnh: PANet (kèm SPP) -> BiFPN.
    Dùng cho Model-5 và Model-6 theo đúng Hình 1 & Hình 4 của bài báo MDPI Sensors 2023.
    """
    def __init__(
        self,
        ch: Tuple[int, int, int] = (128, 256, 512),
        bi_ch: int = 128,
        dsc: bool = True
    ):
        super().__init__()
        self.panet = PANet(ch, dsc=dsc)
        self.bifpn = BiFPN(self.panet.out_channels, ch=bi_ch, dsc=dsc)
        self.out_channels = (bi_ch, bi_ch, bi_ch)

    def forward(
        self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.bifpn(*self.panet(p3, p4, p5))


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print(">> [Unit Test] Running models/pb_module.py...")
    p3 = torch.randn(2, 128, 76, 76)
    p4 = torch.randn(2, 256, 38, 38)
    p5 = torch.randn(2, 512, 19, 19)

    # 1. Test PANet đơn lẻ (dùng cho Model 1..4)
    panet = PANet(ch=(128, 256, 512), dsc=True)
    o_panet = panet(p3, p4, p5)
    assert [tuple(t.shape) for t in o_panet] == [
        (2, 64, 76, 76),
        (2, 128, 38, 38),
        (2, 256, 19, 19),
    ], f"Lỗi shape PANet: {[t.shape for t in o_panet]}"
    print(">> [Unit Test] PANet Shape Test: PASS!")

    # 2. Test PBModule (dùng cho Model 5..6)
    pb = PBModule(ch=(128, 256, 512), bi_ch=128, dsc=True)
    o_pb = pb(p3, p4, p5)
    assert [tuple(t.shape) for t in o_pb] == [
        (2, 128, 76, 76),
        (2, 128, 38, 38),
        (2, 128, 19, 19),
    ], f"Lỗi shape PBModule: {[t.shape for t in o_pb]}"
    print(">> [Unit Test] PBModule Shape Test: PASS!")

    # 3. Test Backward Gradient
    loss = sum(t.mean() for t in o_pb)
    loss.backward()
    print(">> [Unit Test] Backward Gradient Test: PASS!")
    print(">> Unit Test models/pb_module.py: ALL PASS! [DoD M2]")

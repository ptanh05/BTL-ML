"""PANet, SPP and weighted BiFPN necks used by the ablation models."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

try:
    from .common import ConvBnAct, SPP, conv3x3
except ImportError:  # direct execution: python models/pb_module.py
    from common import ConvBnAct, SPP, conv3x3


def _resize(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=ref.shape[-2:], mode="nearest")


def make_n_conv(c_in: int, c_out: int, n: int, dsc: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(n):
        layers.append(conv3x3(c_in if i == 0 else c_out, c_out, dsc=dsc))
    return nn.Sequential(*layers)


class PANet(nn.Module):
    """SPP + top-down and bottom-up path aggregation."""

    def __init__(self, ch: tuple[int, int, int] = (128, 256, 512), dsc: bool = True) -> None:
        super().__init__()
        c3, c4, c5 = ch
        h3, h4, h5 = c3 // 2, c4 // 2, c5 // 2
        self.c5_pre = make_n_conv(c5, h5, 3, dsc)
        self.spp = SPP()
        self.c5_post = make_n_conv(h5 * 4, h5, 3, dsc)
        self.p4_lat = ConvBnAct(c4, h4, 1)
        self.p5_red = ConvBnAct(h5, h4, 1)
        self.p4_td = make_n_conv(h4 * 2, h4, 3, dsc)
        self.p3_lat = ConvBnAct(c3, h3, 1)
        self.p4_red = ConvBnAct(h4, h3, 1)
        self.p3_td = make_n_conv(h3 * 2, h3, 3, dsc)
        self.down3 = conv3x3(h3, h4, 2, dsc)
        self.p4_bu = make_n_conv(h4 * 2, h4, 3, dsc)
        self.down4 = conv3x3(h4, h5, 2, dsc)
        self.p5_bu = make_n_conv(h5 * 2, h5, 3, dsc)
        self.out_channels = (h3, h4, h5)

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor):
        p5_td = self.c5_post(self.spp(self.c5_pre(p5)))
        p4_td = self.p4_td(torch.cat([self.p4_lat(p4), _resize(self.p5_red(p5_td), p4)], dim=1))
        o3 = self.p3_td(torch.cat([self.p3_lat(p3), _resize(self.p4_red(p4_td), p3)], dim=1))
        o4 = self.p4_bu(torch.cat([self.down3(o3), p4_td], dim=1))
        o5 = self.p5_bu(torch.cat([self.down4(o4), p5_td], dim=1))
        return o3, o4, o5


class WeightedAdd(nn.Module):
    def __init__(self, n: int, eps: float = 1e-4) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n))
        self.eps = eps

    def forward(self, xs: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + self.eps)
        return sum(weight * x for weight, x in zip(weights, xs))


class BiFPN(nn.Module):
    def __init__(self, in_ch: tuple[int, int, int], ch: int = 128, dsc: bool = True) -> None:
        super().__init__()
        self.proj = nn.ModuleList([ConvBnAct(c, ch, 1) for c in in_ch])
        self.w_td4, self.w_o3 = WeightedAdd(2), WeightedAdd(2)
        self.w_o4, self.w_o5 = WeightedAdd(3), WeightedAdd(2)
        self.c_td4 = conv3x3(ch, ch, dsc=dsc)
        self.c_o3 = conv3x3(ch, ch, dsc=dsc)
        self.c_o4 = conv3x3(ch, ch, dsc=dsc)
        self.c_o5 = conv3x3(ch, ch, dsc=dsc)
        self.down = nn.MaxPool2d(2, 2)
        self.out_channels = (ch, ch, ch)

    def forward(self, x3: torch.Tensor, x4: torch.Tensor, x5: torch.Tensor):
        x3, x4, x5 = [m(x) for m, x in zip(self.proj, (x3, x4, x5))]
        td4 = self.c_td4(self.w_td4([x4, _resize(x5, x4)]))
        o3 = self.c_o3(self.w_o3([x3, _resize(td4, x3)]))
        down_o3 = F.max_pool2d(o3, kernel_size=2, stride=2)
        o4 = self.c_o4(self.w_o4([x4, td4, _resize(down_o3, x4)]))
        down_o4 = F.max_pool2d(o4, kernel_size=2, stride=2)
        o5 = self.c_o5(self.w_o5([x5, _resize(down_o4, x5)]))
        return o3, o4, o5


class PBModule(nn.Module):
    """PB neck = PANet followed by a three-level weighted BiFPN."""

    def __init__(self, ch: tuple[int, int, int] = (128, 256, 512),
                 bi_ch: int = 128, dsc: bool = True) -> None:
        super().__init__()
        self.panet = PANet(ch, dsc=dsc)
        self.bifpn = BiFPN(self.panet.out_channels, ch=bi_ch, dsc=dsc)
        self.out_channels = self.bifpn.out_channels

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor):
        return self.bifpn(*self.panet(p3, p4, p5))


if __name__ == "__main__":
    p3 = torch.randn(2, 128, 76, 76)
    p4 = torch.randn(2, 256, 38, 38)
    p5 = torch.randn(2, 512, 19, 19)
    out = PBModule() (p3, p4, p5)
    assert [tuple(t.shape) for t in out] == [
        (2, 128, 76, 76), (2, 128, 38, 38), (2, 128, 19, 19)
    ]
    print("pb_module.py: PASS")

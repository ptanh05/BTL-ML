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
        # YOLOv4 alternates 1x1 reduction and 3x3 expansion.
        in_ch = c_in if i == 0 else (c_out * 2 if i % 2 == 0 else c_out)
        if i % 2 == 0:
            layers.append(ConvBnAct(in_ch, c_out, 1, act="leaky"))
        else:
            layers.append(conv3x3(in_ch, c_out * 2, dsc=dsc))
    return nn.Sequential(*layers)


class PANet(nn.Module):
    """SPP + top-down and bottom-up path aggregation."""

    def __init__(self, ch: tuple[int, int, int] = (128, 256, 512), dsc: bool = True,
                 neck_channels: tuple[int, int, int] | None = None) -> None:
        super().__init__()
        widths = tuple(neck_channels or (256, 512, 1024))
        if len(widths) != 3 or any(c < 2 or c % 2 for c in widths):
            raise ValueError("neck_channels must contain three positive even widths")
        c3, c4, c5 = widths
        h3, h4, h5 = c3 // 2, c4 // 2, c5 // 2
        # Backbone channel counts are independent from PANet internal widths.
        # Changing backbone does not silently halve the whole neck/head.
        self.c5_pre = make_n_conv(ch[2], h5, 3, dsc)
        self.spp = SPP()
        self.c5_post = make_n_conv(h5 * 4, h5, 3, dsc)
        self.p4_lat = ConvBnAct(ch[1], h4, 1, act="leaky")
        self.p5_red = ConvBnAct(h5, h4, 1, act="leaky")
        self.p4_td = make_n_conv(h4 * 2, h4, 5, dsc)
        self.p3_lat = ConvBnAct(ch[0], h3, 1, act="leaky")
        self.p4_red = ConvBnAct(h4, h3, 1, act="leaky")
        self.p3_td = make_n_conv(h3 * 2, h3, 5, dsc)
        self.down3 = conv3x3(h3, h4, 2, dsc)
        self.p4_bu = make_n_conv(h4 * 2, h4, 5, dsc)
        self.down4 = conv3x3(h4, h5, 2, dsc)
        self.p5_bu = make_n_conv(h5 * 2, h5, 5, dsc)
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
    """Figure 4: fine-to-coarse first, then coarse-to-fine with middle residual."""

    def __init__(self, in_ch: tuple[int, int, int], ch: int | None = None, dsc: bool = True) -> None:
        super().__init__()
        channels = (ch, ch, ch) if ch is not None else in_ch
        self.proj = nn.ModuleList([
            ConvBnAct(src, dst, 1, act="leaky") if src != dst else nn.Identity()
            for src, dst in zip(in_ch, channels)
        ])
        c3, c4, c5 = channels
        self.down3 = conv3x3(c3, c4, 2, dsc)
        self.down4 = conv3x3(c4, c5, 2, dsc)
        self.up5 = ConvBnAct(c5, c4, 1, act="leaky")
        self.up4 = ConvBnAct(c4, c3, 1, act="leaky")
        self.w_mid4, self.w_o3 = WeightedAdd(2), WeightedAdd(2)
        self.w_o4, self.w_o5 = WeightedAdd(3), WeightedAdd(2)
        self.c_mid4 = conv3x3(c4, c4, dsc=dsc)
        self.c_o3 = conv3x3(c3, c3, dsc=dsc)
        self.c_o4 = conv3x3(c4, c4, dsc=dsc)
        self.c_o5 = conv3x3(c5, c5, dsc=dsc)
        self.out_channels = channels

    def forward(self, x3: torch.Tensor, x4: torch.Tensor, x5: torch.Tensor):
        x3, x4, x5 = [m(x) for m, x in zip(self.proj, (x3, x4, x5))]
        mid4 = self.c_mid4(self.w_mid4([x4, _resize(self.down3(x3), x4)]))
        o5 = self.c_o5(self.w_o5([x5, _resize(self.down4(mid4), x5)]))
        o4 = self.c_o4(self.w_o4([x4, mid4, _resize(self.up5(o5), x4)]))
        o3 = self.c_o3(self.w_o3([x3, _resize(self.up4(o4), x3)]))
        return o3, o4, o5


class PBModule(nn.Module):
    """PB neck = PANet followed by a three-level weighted BiFPN."""

    def __init__(self, ch: tuple[int, int, int] = (128, 256, 512),
                 bi_ch: int | None = None, dsc: bool = True,
                 neck_channels: tuple[int, int, int] | None = None) -> None:
        super().__init__()
        self.panet = PANet(ch, dsc=dsc, neck_channels=neck_channels)
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
        (2, 128, 76, 76), (2, 256, 38, 38), (2, 512, 19, 19)
    ]
    print("pb_module.py: PASS")

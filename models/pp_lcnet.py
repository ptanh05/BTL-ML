"""PP-LCNet x1.0-style lightweight backbone."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch import nn

try:
    from .common import ConvBnAct
except ImportError:  # direct execution: python models/pp_lcnet.py
    from common import ConvBnAct


class SEModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Hardsigmoid(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class DepthSepConv(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3, stride: int = 1,
                 use_se: bool = False) -> None:
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, k, stride, k // 2, groups=c1, bias=False)
        self.dw_bn = nn.BatchNorm2d(c1)
        self.dw_act = nn.Hardswish(inplace=True)
        self.se = SEModule(c1) if use_se else nn.Identity()
        self.pw = nn.Conv2d(c1, c2, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(c2)
        self.pw_act = nn.Hardswish(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_act(self.dw_bn(self.dw(x)))
        x = self.se(x)
        return self.pw_act(self.pw_bn(self.pw(x)))


class PPLCNet(nn.Module):
    """Returns C3=(128,76,76), C4=(256,38,38), C5=(512,19,19)."""

    out_channels = (128, 256, 512)

    def __init__(self, pretrained_path: str | Path | None = None) -> None:
        super().__init__()
        self.stem = ConvBnAct(3, 16, k=3, s=2)
        self.stage2 = nn.Sequential(DepthSepConv(16, 32, 3, 1))
        self.stage3 = nn.Sequential(
            DepthSepConv(32, 64, 3, 2), DepthSepConv(64, 64, 3, 1)
        )
        self.stage4 = nn.Sequential(
            DepthSepConv(64, 128, 3, 2), DepthSepConv(128, 128, 3, 1)
        )
        self.stage5 = nn.Sequential(
            DepthSepConv(128, 256, 3, 2),
            *(DepthSepConv(256, 256, 5, 1) for _ in range(5)),
        )
        self.stage6 = nn.Sequential(
            DepthSepConv(256, 512, 5, 2, use_se=True),
            DepthSepConv(512, 512, 5, 1, use_se=True),
        )
        if pretrained_path:
            self.load_pretrained(pretrained_path)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stage2(self.stem(x))
        x = self.stage3(x)
        c3 = self.stage4(x)
        c4 = self.stage5(c3)
        c5 = self.stage6(c4)
        return c3, c4, c5

    def load_pretrained(self, path: str | Path) -> tuple[list[str], list[str]]:
        payload = torch.load(path, map_location="cpu")
        state = payload.get("state_dict", payload.get("model", payload)) if isinstance(payload, dict) else payload
        missing, unexpected = self.load_state_dict(state, strict=False)
        return list(missing), list(unexpected)


if __name__ == "__main__":
    model = PPLCNet().eval()
    with torch.no_grad():
        c3, c4, c5 = model(torch.randn(2, 3, 608, 608))
    assert (c3.shape, c4.shape, c5.shape) == (
        (2, 128, 76, 76), (2, 256, 38, 38), (2, 512, 19, 19)
    )
    print("pp_lcnet.py: PASS")

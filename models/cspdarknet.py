"""Compact CSPDarknet53-compatible feature extractor for baseline models."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch import nn

try:
    from .common import ConvBnAct
except ImportError:  # direct execution: python models/cspdarknet.py
    from common import ConvBnAct


class Bottleneck(nn.Module):
    def __init__(self, channels: int, shortcut: bool = True) -> None:
        super().__init__()
        hidden = max(channels // 2, 1)
        self.cv1 = ConvBnAct(channels, hidden, 1)
        self.cv2 = ConvBnAct(hidden, channels, 3)
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.shortcut else y


class CSPStage(nn.Module):
    def __init__(self, c1: int, c2: int, blocks: int) -> None:
        super().__init__()
        self.down = ConvBnAct(c1, c2, 3, 2) if c1 != c2 else nn.Identity()
        hidden = max(c2 // 2, 1)
        self.left = ConvBnAct(c2, hidden, 1)
        self.right = ConvBnAct(c2, hidden, 1)
        self.blocks = nn.Sequential(*(Bottleneck(hidden) for _ in range(blocks)))
        self.fuse = ConvBnAct(hidden * 2, c2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return self.fuse(torch.cat([self.blocks(self.left(x)), self.right(x)], dim=1))


class CSPDarknet53(nn.Module):
    out_channels = (256, 512, 1024)

    def __init__(self, pretrained_path: str | Path | None = None) -> None:
        super().__init__()
        self.stem = ConvBnAct(3, 32, 3, 2)
        self.stage1 = CSPStage(32, 64, 1)
        self.stage2 = CSPStage(64, 128, 1)
        # Keep the first detection feature at stride 8 (76x76), then project
        # the CSP features to the YOLOv4 C3 width.
        self.stage3 = nn.Sequential(CSPStage(128, 128, 2), ConvBnAct(128, 256, 1))
        self.stage4 = CSPStage(256, 512, 2)
        self.stage5 = CSPStage(512, 1024, 2)
        if pretrained_path:
            self.load_pretrained(pretrained_path)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stage2(self.stage1(self.stem(x)))
        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return c3, c4, c5

    def load_pretrained(self, path: str | Path) -> tuple[list[str], list[str]]:
        payload = torch.load(path, map_location="cpu")
        state = payload.get("state_dict", payload.get("model", payload)) if isinstance(payload, dict) else payload
        missing, unexpected = self.load_state_dict(state, strict=False)
        return list(missing), list(unexpected)


if __name__ == "__main__":
    model = CSPDarknet53().eval()
    with torch.no_grad():
        outputs = model(torch.randn(1, 3, 608, 608))
    assert [tuple(t.shape) for t in outputs] == [
        (1, 256, 76, 76), (1, 512, 38, 38), (1, 1024, 19, 19)
    ]
    print("cspdarknet.py: PASS")

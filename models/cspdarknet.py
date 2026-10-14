"""YOLOv4 CSPDarknet53 backbone: Mish and CSP depths (1, 2, 8, 8, 4)."""

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
    def __init__(self, channels: int, shortcut: bool = True,
                 hidden_channels: int | None = None) -> None:
        super().__init__()
        hidden = hidden_channels if hidden_channels is not None else channels
        self.cv1 = ConvBnAct(channels, hidden, 1, act="mish")
        self.cv2 = ConvBnAct(hidden, channels, 3, act="mish")
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.shortcut else y


class CSPStage(nn.Module):
    def __init__(self, c1: int, c2: int, blocks: int, first: bool = False) -> None:
        super().__init__()
        self.down = ConvBnAct(c1, c2, 3, 2, act="mish")
        hidden = c2 if first else c2 // 2
        self.left = ConvBnAct(c2, hidden, 1, act="mish")
        self.right = ConvBnAct(c2, hidden, 1, act="mish")
        self.blocks = nn.Sequential(*(
            Bottleneck(hidden, hidden_channels=c2 // 2 if first else hidden)
            for _ in range(blocks)
        ))
        self.post = ConvBnAct(hidden, hidden, 1, act="mish")
        self.fuse = ConvBnAct(hidden * 2, c2, 1, act="mish")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return self.fuse(torch.cat([self.post(self.blocks(self.left(x))), self.right(x)], dim=1))


class CSPDarknet53(nn.Module):
    out_channels = (256, 512, 1024)

    def __init__(self, pretrained_path: str | Path | None = None) -> None:
        super().__init__()
        self.stem = ConvBnAct(3, 32, 3, 1, act="mish")
        self.stage1 = CSPStage(32, 64, 1, first=True)
        self.stage2 = CSPStage(64, 128, 2)
        self.stage3 = CSPStage(128, 256, 8)
        self.stage4 = CSPStage(256, 512, 8)
        self.stage5 = CSPStage(512, 1024, 4)
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
        if any(key.startswith("backbone.") for key in state):
            state = {key.removeprefix("backbone."): value for key, value in state.items()
                     if key.startswith("backbone.")}
        missing, unexpected = self.load_state_dict(state, strict=True)
        return list(missing), list(unexpected)


if __name__ == "__main__":
    model = CSPDarknet53().eval()
    with torch.no_grad():
        outputs = model(torch.randn(1, 3, 608, 608))
    assert [tuple(t.shape) for t in outputs] == [
        (1, 256, 76, 76), (1, 512, 38, 38), (1, 1024, 19, 19)
    ]
    print("cspdarknet.py: PASS")

"""Factory and unified model class for all six ablation configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

try:
    from .attention import CoordAtt
    from .cspdarknet import CSPDarknet53
    from .head import YOLOHead
    from .pb_module import PANet, PBModule
    from .pp_lcnet import PPLCNet
except ImportError:  # direct execution: python models/yolo.py
    from attention import CoordAtt
    from cspdarknet import CSPDarknet53
    from head import YOLOHead
    from pb_module import PANet, PBModule
    from pp_lcnet import PPLCNet


MODEL_CFGS = {
    1: {"backbone": "cspdarknet53", "dsc": False, "ca": False, "neck": "panet", "loss": "ciou"},
    2: {"backbone": "cspdarknet53", "dsc": True, "ca": False, "neck": "panet", "loss": "ciou"},
    3: {"backbone": "pplcnet", "dsc": True, "ca": False, "neck": "panet", "loss": "ciou"},
    4: {"backbone": "pplcnet", "dsc": True, "ca": True, "neck": "panet", "loss": "ciou"},
    5: {"backbone": "pplcnet", "dsc": True, "ca": True, "neck": "pb", "loss": "ciou"},
    6: {"backbone": "pplcnet", "dsc": True, "ca": True, "neck": "pb", "loss": "siou"},
}

TARGET_MB = {1: 243.92, 2: 136.13, 3: 38.75, 4: 38.99, 5: 41.88, 6: 41.88}


class YOLOv4Variant(nn.Module):
    def __init__(self, cfg: dict[str, Any], num_classes: int = 2) -> None:
        super().__init__()
        arch = cfg.get("architecture", cfg)
        self.cfg = arch
        self.num_classes = num_classes
        if arch.get("backbone", "pplcnet") == "pplcnet":
            self.backbone = PPLCNet(layout=arch.get("pplcnet_layout", "paper"),
                                    se_blocks=int(arch.get("pplcnet_se_blocks", 2)))
            channels = (128, 256, 512)
        elif arch.get("backbone") == "cspdarknet53":
            self.backbone = CSPDarknet53()
            channels = (256, 512, 1024)
        else:
            raise ValueError(f"Unsupported backbone: {arch.get('backbone')}")
        self.use_ca = bool(arch.get("ca", False))
        neck_channels = tuple(arch.get("neck_channels", (256, 512, 1024)))
        self.ca = nn.ModuleList([
            CoordAtt(c, reduction=int(arch.get("ca_reduction", 32))) for c in channels
        ]) if self.use_ca else nn.ModuleList()
        dsc = bool(arch.get("dsc", True))
        if arch.get("neck", "panet") == "pb":
            self.neck = PBModule(
                channels, bi_ch=(int(arch["bifpn_channels"]) if arch.get("bifpn_channels") is not None else None), dsc=dsc,
                neck_channels=neck_channels,
            )
        elif arch.get("neck", "panet") == "panet":
            self.neck = PANet(channels, dsc=dsc, neck_channels=neck_channels)
        else:
            raise ValueError(f"Unsupported neck: {arch.get('neck')}")
        self.head = YOLOHead(self.neck.out_channels, num_classes, 3, dsc=dsc)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = self.backbone(x)
        if self.use_ca:
            features = tuple(att(feature) for att, feature in zip(self.ca, features))
        return self.head(*self.neck(*features))

    def get_model_size_mb(self) -> float:
        return sum(p.numel() for p in self.parameters()) * 4 / (1024 ** 2)


def build_model(config_path_or_dict: str | Path | dict[str, Any], num_classes: int = 2) -> YOLOv4Variant:
    if isinstance(config_path_or_dict, (str, Path)):
        with open(config_path_or_dict, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    else:
        config = config_path_or_dict
    return YOLOv4Variant(config, num_classes=num_classes)


if __name__ == "__main__":
    for model_id, cfg in MODEL_CFGS.items():
        model = YOLOv4Variant(cfg).eval()
        with torch.no_grad():
            outputs = model(torch.randn(1, 3, 608, 608))
        assert [tuple(o.shape) for o in outputs] == [
            (1, 21, 76, 76), (1, 21, 38, 38), (1, 21, 19, 19)
        ]
        print(f"model-{model_id}: {model.get_model_size_mb():.2f} MB - PASS")

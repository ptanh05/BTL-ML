"""
================================================================================
MODULE: models/yolo.py - LẮP RÁP KIẾN TRÚC THỐNG NHẤT CHO CẢ 6 BIẾN THỂ YOLOV4
================================================================================
MỤC ĐÍCH:
    - Cung cấp lớp mô hình thống nhất `YOLOv4Variant(nn.Module)` có khả năng khởi tạo
      bất kỳ mô hình nào trong 6 biến thể thực nghiệm (Model-1 đến Model-6) thông qua file cấu hình.
    - Kết nối mượt mà 4 khối kiến trúc:
        Backbone (CSPDarknet53 / PP-LCNet)
        -> [Coordinate Attention tùy chọn]
        -> Neck (PANet / PB Module)
        -> Detection Head (YOLOHead)
    - Cung cấp hàm đo đạc và kiểm tra kích thước file trọng số (Model Size in MB),
      đối chiếu trực tiếp với Bảng 1 của bài báo.

PHÂN CÔNG TRÁCH NHIỆM (RACI):
    - Người thực thi & Chịu trách nhiệm (R & A): T.A (Project Lead) & Sơn (Technical Mentor)
================================================================================
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import yaml
from typing import Dict, Any, List, Union

from models.pp_lcnet import PPLCNet
from models.cspdarknet import CSPDarknet53
from models.attention import CoordAtt
from models.pb_module import PANet, PBModule
from models.head import YOLOHead


# Cấu hình chuẩn của 6 mô hình phục vụ kiểm thử (Ablation Study)
MODEL_CFGS = {
    1: dict(backbone="cspdarknet53", dsc=False, ca=False, neck="panet", loss="ciou"),
    2: dict(backbone="cspdarknet53", dsc=True,  ca=False, neck="panet", loss="ciou"),
    3: dict(backbone="pplcnet",      dsc=True,  ca=False, neck="panet", loss="ciou"),
    4: dict(backbone="pplcnet",      dsc=True,  ca=True,  neck="panet", loss="ciou"),
    5: dict(backbone="pplcnet",      dsc=True,  ca=True,  neck="pb",    loss="ciou"),
    6: dict(backbone="pplcnet",      dsc=True,  ca=True,  neck="pb",    loss="siou"),
}

TARGET_MB = {
    1: 243.92,
    2: 136.13,
    3: 38.75,
    4: 38.99,
    5: 41.88,
    6: 41.88
}


class YOLOv4Variant(nn.Module):
    """
    Kiến trúc YOLOv4 tổng quát: Linh hoạt tạo Model-1 đến Model-6 qua dict cấu hình.
    """
    def __init__(self, cfg: Dict[str, Any], num_classes: int = 2):
        super().__init__()
        # Hỗ trợ đọc cả dạng cfg phẳng hoặc lồng trong architecture:
        arch_cfg = cfg.get("architecture", cfg)
        self.cfg = arch_cfg
        self.num_classes = num_classes

        # 1. Khởi tạo Backbone
        if arch_cfg.get("backbone") == "pplcnet":
            self.backbone = PPLCNet()
            ch = (128, 256, 512)
        else:
            self.backbone = CSPDarknet53()
            ch = (256, 512, 1024)

        # 2. Khởi tạo Coordinate Attention (CA) nếu bật
        self.use_ca = arch_cfg.get("ca", False)
        if self.use_ca:
            reduction = arch_cfg.get("ca_reduction", 32)
            self.ca = nn.ModuleList([CoordAtt(c, reduction=reduction) for c in ch])
        else:
            self.ca = None

        # 3. Khởi tạo Neck (PANet hoặc PBModule)
        use_dsc = arch_cfg.get("dsc", True)
        if arch_cfg.get("neck") == "pb":
            bi_ch = arch_cfg.get("bifpn_channels", 128)
            self.neck = PBModule(ch=ch, bi_ch=bi_ch, dsc=use_dsc)
        else:
            self.neck = PANet(ch=ch, dsc=use_dsc)

        # 4. Khởi tạo YOLOHead
        self.head = YOLOHead(
            in_channels=self.neck.out_channels,
            num_classes=self.num_classes,
            num_anchors=3,
            dsc=use_dsc
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward Pass:
        x -> Backbone -> [CA] -> Neck -> Head -> 3 predicted feature maps
        """
        feats = self.backbone(x)
        if self.ca is not None:
            feats = tuple(att(f) for att, f in zip(self.ca, feats))
        neck_feats = self.neck(*feats)
        return self.head(*neck_feats)

    def get_model_size_mb(self) -> float:
        """
        Tính kích thước file trọng số FP32 (MB) tương đương file .pth chứa state_dict.
        """
        return sum(p.numel() for p in self.parameters()) * 4 / (1024 * 1024)


def build_model(config_path_or_dict: Union[str, Path, dict], num_classes: int = 2) -> YOLOv4Variant:
    """
    Factory tạo mô hình từ file cấu hình YAML hoặc dict.
    """
    if isinstance(config_path_or_dict, (str, Path)):
        with open(config_path_or_dict, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = config_path_or_dict
    return YOLOv4Variant(cfg, num_classes=num_classes)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print(">> [Unit Test] Running models/yolo.py cho toàn bộ 6 Model...")
    x = torch.randn(1, 3, 608, 608)

    for k in range(1, 7):
        cfg = MODEL_CFGS[k]
        model = YOLOv4Variant(cfg)
        outs = model(x)
        assert [tuple(o.shape) for o in outs] == [
            (1, 21, 76, 76),
            (1, 21, 38, 38),
            (1, 21, 19, 19)
        ], f"Lỗi shape đầu ra Model-{k}: {[o.shape for o in outs]}"

        # Test backward
        loss = sum(o.mean() for o in outs)
        loss.backward()

        mb = model.get_model_size_mb()
        print(f">> Model-{k} [Backbone: {cfg['backbone']}, DSC: {cfg['dsc']}, CA: {cfg['ca']}, Neck: {cfg['neck']}]: {mb:.2f} MB (Target: {TARGET_MB[k]} MB) - PASS!")

    print(">> Unit Test models/yolo.py: ALL 6 MODELS PASS! [DoD M3]")

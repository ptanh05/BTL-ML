"""
================================================================================
MODULE: models/head.py - YOLO DETECTION HEAD & GIẢI MÃ BOUNDING BOX
================================================================================
MỤC ĐÍCH:
    - Xây dựng tầng đầu ra phát hiện mục tiêu (YOLOHead) trên 3 tỉ lệ đặc trưng (76x76, 38x38, 19x19).
    - Cấu hình số kênh đầu ra: `num_anchors * (5 + num_classes)`.
      Với 2 lớp (`hat` và `person`) và 3 anchors/tầng:
      Số kênh đầu ra = 3 * (4 tọa độ + 1 objectness + 2 class) = 21 kênh.
    - Hỗ trợ cờ `dsc`: Sử dụng DSConv (Model-2..6) hoặc Conv 3x3 chuẩn (Model-1).

PHÂN CÔNG TRÁCH NHIỆM (RACI):
    - Người thực thi (Responsible): Tú (Loss & Head Developer)
    - Người chịu trách nhiệm (Accountable): T.A (Project Lead)
================================================================================
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from typing import List, Tuple
from models.common import conv3x3


class YOLOHead(nn.Module):
    """
    Detection Head cho 3 mức tỉ lệ đặc trưng:
    Mỗi nhánh gồm: DSConv (hoặc Conv 3x3) -> Conv 1x1 ra 21 kênh.
    """
    def __init__(
        self,
        in_channels: Tuple[int, int, int],
        num_classes: int = 2,
        num_anchors: int = 3,
        dsc: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.out_channels = num_anchors * (5 + num_classes)  # = 21

        self.heads = nn.ModuleList([
            nn.Sequential(
                conv3x3(c, 2 * c, stride=1, dsc=dsc),
                nn.Conv2d(2 * c, self.out_channels, kernel_size=1)
            )
            for c in in_channels
        ])

    def forward(self, *feats: torch.Tensor) -> List[torch.Tensor]:
        """
        Nhận 3 feature maps (từ Neck), trả về 3 tensor dự đoán có kênh = 21.
        """
        return [head(f) for head, f in zip(self.heads, feats)]


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print(">> [Unit Test] Running models/head.py...")
    # Thử nghiệm với đầu ra của PBModule (128, 128, 128)
    head = YOLOHead(in_channels=(128, 128, 128), num_classes=2, num_anchors=3, dsc=True)

    f3 = torch.randn(2, 128, 76, 76)
    f4 = torch.randn(2, 128, 38, 38)
    f5 = torch.randn(2, 128, 19, 19)

    outs = head(f3, f4, f5)
    assert [tuple(o.shape) for o in outs] == [
        (2, 21, 76, 76),
        (2, 21, 38, 38),
        (2, 21, 19, 19)
    ], f"Lỗi shape YOLOHead: {[o.shape for o in outs]}"
    print(">> [Unit Test] YOLOHead Output Shapes (21 channels): PASS!")

    loss = sum(o.mean() for o in outs)
    loss.backward()
    print(">> [Unit Test] Backward Gradient Test: PASS!")
    print(">> Unit Test models/head.py: ALL PASS! [DoD M2]")

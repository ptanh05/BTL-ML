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
        # TODO: Tú khởi tạo nn.ModuleList chứa các nhánh conv3x3 và conv 1x1
        raise NotImplementedError("Cần được Tú (Loss & Head Developer) cài đặt YOLOHead.")

    def forward(self, *feats: torch.Tensor) -> List[torch.Tensor]:
        """
        Nhận 3 feature maps (từ Neck), trả về 3 tensor dự đoán có kênh = 21.
        """
        raise NotImplementedError("Cần được Tú cài đặt forward của YOLOHead.")


def decode_box_predictions(
    predictions: List[torch.Tensor],
    anchors: torch.Tensor,
    strides: List[int] = [8, 16, 32]
) -> torch.Tensor:
    """
    Giải mã dự đoán thô từ head thành bounding boxes chuẩn hóa [cx, cy, w, h] hoặc [x1, y1, x2, y2].
    """
    raise NotImplementedError("Cần được Tú cài đặt decode_box_predictions.")


def non_max_suppression(
    prediction: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45
) -> List[torch.Tensor]:
    """
    Thực hiện NMS loại bỏ các hộp bao trùng lặp dựa trên IoU threshold.
    """
    raise NotImplementedError("Cần được Tú cài đặt non_max_suppression.")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(">> [TODO] models/head.py: Tú cần hoàn thiện YOLOHead và hàm decode để vượt qua Unit Test.")

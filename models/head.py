"""
================================================================================
MODULE: models/head.py - YOLO DETECTION HEAD & GIẢI MÃ BOUNDING BOX
================================================================================
MỤC ĐÍCH:
    - Xây dựng tầng đầu ra phát hiện mục tiêu (YOLOHead) trên 3 tỉ lệ đặc trưng (76x76, 38x38, 19x19).
    - Cấu hình số kênh đầu ra: `num_anchors * (5 + num_classes)`.
      Với 2 lớp (`hat` và `person`) và 3 anchors/tầng:
      Số kênh đầu ra = 3 * (4 tọa độ + 1 objectness + 2 class) = 21 kênh.
    - Cung cấp hàm giải mã dự đoán (Box Decoding) từ giá trị offset không gian lưới sang tọa độ thực tế:
      tâm (cx, cy), kích thước (w, h), độ tin cậy (conf), xác suất lớp (class prob).
    - Hỗ trợ Non-Maximum Suppression (NMS) để loại bỏ các hộp bao trùng lặp khi suy luận (Inference).

PHÂN CÔNG TRÁCH NHIỆM (RACI):
    - Người thực thi (Responsible): Tú (Loss & Head Developer)
    - Người chịu trách nhiệm (Accountable): T.A (Project Lead)
    - Tham vấn (Consulted): Sơn (Technical Mentor)
    - Nhận bàn giao (Informed): Toàn đội (dùng trong train.py, eval.py, detect.py)

GIAO DIỆN & ĐẶC TẢ TENSOR:
    1. Input:
       - 3 feature maps từ Neck: feats = [f3, f4, f5]
         + f3: (B, c3, 76, 76)
         + f4: (B, c4, 38, 38)
         + f5: (B, c5, 19, 19)
    2. Output:
       - Khi Huấn luyện (Train): Trả về danh sách 3 tensor thô (B, 21, 76, 76), (B, 21, 38, 38), (B, 21, 19, 19)
         để truyền vào hàm Loss.
       - Khi Suy luận (Inference): Trả về tensor đã giải mã: (B, N_boxes, 7) gồm [x1, y1, x2, y2, conf, score, class_id].

CÔNG THỨC GIẢI MÃ TỌA ĐỘ THEO YOLOV4:
    - Tâm dự đoán:
      + bx = (sigmoid(tx) + grid_x) * stride
      + by = (sigmoid(ty) + grid_y) * stride
    - Kích thước dự đoán:
      + bw = anchor_w * exp(tw.clamp(max=10.0))
      + bh = anchor_h * exp(th.clamp(max=10.0))
    - Confidence & Class probabilities:
      + objectness = sigmoid(to)
      + class_prob = sigmoid(t_cls)

TIÊU CHÍ NGHIỆM THU (DoD M2):
    [ ] Truyền 3 feature maps qua YOLOHead: nhận về chính xác 3 tensor có số kênh là 21.
    [ ] Hàm giải mã chuyển đổi mượt mà giữa tensor thô và bounding box chuẩn.
    [ ] Chạy unit test độc lập kiểm tra giải mã và NMS thành công.

BẪY LỖI KỸ THUẬT CẦN TRÁNH:
    - Bẫy 1: Giá trị tw, th quá lớn khi exp() sẽ gây ra hiện tượng tràn số (Overflow Inf) -> Bắt buộc dùng `clamp(max=10.0)` trước hàm `torch.exp()`.
    - Bẫy 2: Lệch thứ tự grid_x và grid_y khi tạo meshgrid. Cần lưu ý `torch.meshgrid(indexing='ij')` so với `'xy'`.
================================================================================
"""

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
        # HƯỚNG DẪN: Tú khởi tạo nn.ModuleList chứa các nhánh tích chập
        pass

    def forward(self, *feats: torch.Tensor) -> List[torch.Tensor]:
        """
        Nhận 3 feature maps, trả về 3 tensor dự đoán có kênh = 21.
        """
        raise NotImplementedError("Cần được Tú cài đặt YOLOHead forward.")


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
    print("[TODO] models/head.py: Chạy unit test kiểm tra YOLOHead và hàm decode.")

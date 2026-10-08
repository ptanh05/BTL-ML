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
    - Người thực thi (Responsible): T.A (Project Lead) & Sơn (Technical Mentor)
    - Người chịu trách nhiệm (Accountable): T.A (Project Lead)
    - Tham vấn (Consulted): Hưng, Tiến, Tú
    - Nhận bàn giao (Informed): Toàn bộ 5 thành viên để huấn luyện trên Colab cá nhân

GIAO DIỆN & ĐẶC TẢ TENSOR:
    1. Input:
       - Tensor ảnh `x`: (B, 3, 608, 608)
    2. Output:
       - Khi Huấn luyện: Trả về danh sách 3 tensor thô từ head: (B, 21, 76, 76), (B, 21, 38, 38), (B, 21, 19, 19).
       - Khi Đánh giá / Suy luận: Trả về tensor bounding boxes đã giải mã.

BẢNG ĐỐI CHIẾU DUNG LƯỢNG MỤC TIÊU (Bảng 1 bài báo):
    - Model-1: 243.92 MB (CSPDarknet53 + Conv chuẩn + PANet + CIoU)
    - Model-2: 136.13 MB (CSPDarknet53 + DSConv + PANet + CIoU)
    - Model-3:  38.75 MB (PP-LCNet + DSConv + PANet + CIoU)
    - Model-4:  38.99 MB (PP-LCNet + CA + DSConv + PANet + CIoU)
    - Model-5:  41.88 MB (PP-LCNet + CA + DSConv + PB Module + CIoU)
    - Model-6:  41.88 MB (PP-LCNet + CA + DSConv + PB Module + SIoU)

TIÊU CHÍ NGHIỆM THU (DoD M3):
    [ ] Chạy `python models/yolo.py`: Cả 6 cấu hình đều khởi tạo thành công không lỗi cú pháp.
    [ ] Forward pass: Tensor (1, 3, 608, 608) đi qua toàn bộ mạng mượt mà.
    [ ] Backward pass: `loss.backward()` tính được gradient ổn định cho mọi tham số.
    [ ] Dung lượng mô hình FP32 (.pth) xấp xỉ dung lượng mục tiêu trong Bảng 1 (dung sai cho phép: ±10%).

BẪY LỖI KỸ THUẬT CẦN TRÁNH:
    - Bẫy 1: Sai lệch số kênh giữa Backbone và Neck (CSPDarknet là 256/512/1024, còn PP-LCNet là 128/256/512).
    - Bẫy 2: Không đồng bộ cờ `dsc` giữa Neck và Head (khiến Model-1 vô tình dùng DSConv hoặc Model-2 dùng Standard Conv).
================================================================================
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List
from pathlib import Path


class YOLOv4Variant(nn.Module):
    """
    Kiến trúc YOLOv4 tổng quát: Linh hoạt tạo Model-1 đến Model-6 qua dict cấu hình.
    """
    def __init__(self, cfg: Dict[str, Any], num_classes: int = 2):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes

        # HƯỚNG DẪN: T.A & Sơn khởi tạo 4 khối:
        # 1. Backbone: PPLCNet() nếu cfg['backbone'] == 'pplcnet' else CSPDarknet53()
        # 2. Attention: nn.ModuleList([CoordAtt(c) for c in ch]) nếu cfg['ca'] else None
        # 3. Neck: PBModule(ch, dsc=cfg['dsc']) nếu cfg['neck'] == 'pb' else PANet(ch, dsc=cfg['dsc'])
        # 4. Head: YOLOHead(out_channels, num_classes=2, dsc=cfg['dsc'])
        pass

    def forward(self, x: torch.Tensor) -> List[torch.Tensor] | torch.Tensor:
        """
        Dòng chảy dữ liệu (Forward Pass):
        x -> Backbone -> [CA] -> Neck -> Head -> predictions
        """
        raise NotImplementedError("Cần được T.A & Sơn cài đặt forward của YOLOv4Variant.")

    def get_model_size_mb(self) -> float:
        """
        Tính kích thước file trọng số state_dict ở định dạng FP32 (MB).
        Công thức: sum(param.numel() for param in model.parameters()) * 4 / (1024 * 1024)
        """
        raise NotImplementedError("Cần được T.A cài đặt get_model_size_mb.")


def build_model(config_path_or_dict: str | Path | dict) -> YOLOv4Variant:
    """
    Hàm Factory đọc cấu hình từ file YAML hoặc dictionary và tạo đối tượng YOLOv4Variant.
    """
    raise NotImplementedError("Cần được T.A cài đặt build_model.")


if __name__ == "__main__":
    print("[TODO] models/yolo.py: Chạy unit test kiểm tra forward pass và tính Model Size cho cả 6 mô hình.")

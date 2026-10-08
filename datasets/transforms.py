"""
================================================================================
MODULE: datasets/transforms.py - TIỀN XỬ LÝ & TĂNG CƯỜNG DỮ LIỆU (AUGMENTATION)
================================================================================
MỤC ĐÍCH:
    - Cung cấp pipeline tiền xử lý ảnh và nhãn cho mô hình YOLOv4 với kích thước cố định 608x608.
    - Hỗ trợ Letterbox Resize (giữ nguyên tỷ lệ khung hình, bù viền xám 114) để tránh méo ảnh.
    - Cung cấp các phép biến đổi tăng cường dữ liệu khi huấn luyện:
      + Lật ngang ảnh ngẫu nhiên (Horizontal Flip với xác suất 50%).
      + Biến đổi không gian màu HSV (Hue, Saturation, Value).
      + Chuẩn hóa giá trị điểm ảnh về khoảng [0.0, 1.0].
    - Pipeline riêng biệt cho Validation & Test: Chỉ thực hiện Letterbox resize + Normalize.

PHÂN CÔNG TRÁCH NHIỆM (RACI):
    - Người thực thi (Responsible): Sơn (Technical Mentor)
    - Người chịu trách nhiệm (Accountable): T.A (Project Lead)
    - Nhận bàn giao (Informed): Toàn đội (sử dụng chung transform cho cả 6 mô hình)

GIAO DIỆN & ĐẶC TẢ DỮ LIỆU:
    1. Input:
       - image: Ảnh gốc dạng numpy array (H, W, 3) kênh RGB hoặc PIL Image.
       - bboxes: Ma trận bounding box dạng numpy array (N, 4) [x1, y1, x2, y2]
         hoặc [cx, cy, w, h] kèm nhãn lớp classes (N,).
    2. Output:
       - image_tensor: torch.Tensor có kích thước cố định (3, 608, 608), kiểu torch.float32, giá trị [0.0, 1.0].
       - target_boxes: Bounding boxes đã được scale và offset tọa độ theo đúng tỷ lệ Letterbox.

HƯỚNG DẪN CÀI ĐẶT:
    1. `letterbox_image(image, target_size=(608, 608), pad_color=(114, 114, 114))`:
       - Tính tỷ lệ scale: r = min(target_h / h, target_w / w).
       - Resize ảnh theo r, tính padding dw, dh để bù viền cân đối 2 bên.
       - Cập nhật lại tọa độ bounding box tương ứng: x' = x * r + dw, y' = y * r + dh.
    2. `get_train_transforms(img_size=608)`:
       - Trả về đối tượng Albumentations Compose hoặc PyTorch transform cho tập Train.
    3. `get_val_transforms(img_size=608)`:
       - Trả về transform chuẩn cho tập Val và Test (chỉ letterbox + normalize, không flip/jitter).

TIÊU CHÍ NGHIỆM THU (DoD M1):
    [ ] Ảnh đầu ra luôn đúng kích thước (3, 608, 608).
    [ ] Tọa độ bounding box sau biến đổi không bị tràn ra ngoài kích thước 608x608.
    [ ] Bbox sau biến đổi vẫn bao chính xác mục tiêu trên ảnh (kiểm tra bằng unit test vẽ bbox).

BẪY LỖI KỸ THUẬT CẦN TRÁNH:
    - Bẫy 1: Không scale tọa độ bounding box khi resize ảnh -> Model học sai nhãn hoàn toàn!
    - Bẫy 2: OpenCV đọc ảnh mặc định là BGR, PyTorch và Albumentations mong đợi RGB. Phải chuyển đổi màu `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`.
================================================================================
"""

import numpy as np
import torch
from typing import Tuple, Any


def letterbox(
    image: np.ndarray,
    target_size: Tuple[int, int] = (608, 608),
    pad_color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize ảnh kèm bù viền (letterboxing) để bảo toàn tỷ lệ khung hình (aspect ratio).
    
    Args:
        image: Ảnh gốc RGB (H, W, 3).
        target_size: (target_height, target_width), mặc định (608, 608).
        pad_color: Giá trị RGB của phần bù viền, mặc định xám (114, 114, 114).
        
    Returns:
        padded_image: Ảnh sau letterbox (target_h, target_w, 3).
        ratio: Tỷ lệ co giãn r.
        (pad_w, pad_h): Giá trị bù viền theo chiều ngang và dọc.
    """
    # HƯỚNG DẪN: Sơn triển khai thuật toán tính ratio và cv2.copyMakeBorder tại đây
    raise NotImplementedError("Chức năng letterbox cần được Sơn (Mentor) hoàn thiện.")


def get_train_transforms(img_size: int = 608) -> Any:
    """
    Khởi tạo pipeline tăng cường dữ liệu cho tập huấn luyện (Train).
    """
    # HƯỚNG DẪN: Khởi tạo albumentations.Compose với HorizontalFlip, ColorJitter, Normalize...
    raise NotImplementedError("Chức năng get_train_transforms cần được Sơn hoàn thiện.")


def get_val_transforms(img_size: int = 608) -> Any:
    """
    Khởi tạo pipeline tiền xử lý cho tập kiểm định (Val) và đánh giá (Test).
    """
    # HƯỚNG DẪN: Chỉ chuẩn hóa Normalize và đưa về tensor, không thêm nhiễu/lật
    raise NotImplementedError("Chức năng get_val_transforms cần được Sơn hoàn thiện.")


if __name__ == "__main__":
    print("[TODO] datasets/transforms.py: Chạy unit test kiểm tra letterbox và bounding box coordinate shift.")

"""
================================================================================
MODULE: utils/metrics.py - ĐÁNH GIÁ CHỈ SỐ: MAP@0.5, PRECISION, RECALL, F1 & FPS
================================================================================
MỤC ĐÍCH:
    - Cung cấp toàn bộ công thức và hàm đo lường chỉ số định lượng theo bài báo MDPI Sensors 2023.
    - Tính toán độ chính xác phát hiện cho 2 lớp mục tiêu:
      + AP Hat (Average Precision lớp mũ bảo hộ tại IoU=0.5).
      + AP Person (Average Precision lớp người không đội mũ tại IoU=0.5).
      + mAP@0.5 = (AP Hat + AP Person) / 2 (Công thức 11 bài báo).
    - Tính toán Precision (P), Recall (R), F1-Score tại ngưỡng confidence tối ưu (Bảng 2 bài báo).
    - Đo đạc tốc độ suy luận FPS (Frames Per Second) và dung lượng file trọng số (MB).

PHÂN CÔNG TRÁCH NHIỆM (RACI):
    - Người thực thi (Responsible): Tú & Sơn (Loss & Metrics Developers)
    - Người chịu trách nhiệm (Accountable): T.A (Project Lead)
    - Nhận bàn giao (Informed): Toàn đội (sử dụng trong eval.py để xuất kết quả nghiệm thu)

GIAO DIỆN & ĐẶC TẢ DỮ LIỆU:
    1. Input:
       - Dự đoán đã qua NMS: Danh sách detections [x1, y1, x2, y2, conf, class_id].
       - Nhãn thật (Ground Truths): [class_id, x1, y1, x2, y2].
       - Ngưỡng IoU đánh giá: iou_threshold = 0.5.
    2. Output:
       - Dictionary chứa đầy đủ chỉ số:
         {
           'ap_hat': float,      # Ví dụ: 0.9434
           'ap_person': float,   # Ví dụ: 0.9163
           'map_50': float,      # Ví dụ: 0.9298
           'precision': float,
           'recall': float,
           'f1': float,
           'fps': float,
           'model_size_mb': float
         }

BẢNG CHỈ TIÊU NGHIỆM THU ĐỐI CHIẾU:
    - Bảng 1 (Ablation Study):
      Model-1: AP Hat 93.15% | AP Person 91.77% | mAP 92.46% | 243.92 MB
      Model-2: AP Hat 92.71% | AP Person 91.00% | mAP 91.86% | 136.13 MB
      Model-3: AP Hat 88.85% | AP Person 89.38% | mAP 89.34% |  38.75 MB
      Model-4: AP Hat 89.36% | AP Person 92.82% | mAP 90.09% |  38.99 MB
      Model-5: AP Hat 90.54% | AP Person 92.03% | mAP 91.29% |  41.88 MB
      Model-6: AP Hat 94.34% | AP Person 91.63% | mAP 92.98% |  41.88 MB
      (*) Chấp nhận dung sai ±1.0% mAP do sai lệch nội tại của bài báo ở Model-3/4.
    - Bảng 2: So sánh tốc độ:
      Model-6 / Model-1: Tỉ lệ FPS bài báo là ~1.87x (43.24 vs 23.02 FPS).

TIÊU CHÍ NGHIỆM THU (DoD M5):
    [ ] Script `eval.py` tính toán chỉ số tự động từ test set và trả kết quả dưới dạng bảng DataFrame / Markdown.
    [ ] Đo FPS trên cùng một thiết bị GPU (Colab T4), thực hiện warmup tối thiểu 50 ảnh trước khi bấm giờ.

BẪY LỖI KỸ THUẬT CẦN TRÁNH:
    - Bẫy 1: Không đồng bộ kích thước ảnh khi đo FPS: Đo FPS bắt buộc dùng đúng ảnh kích thước (1, 3, 608, 608).
    - Bẫy 2: Đo FPS bao gồm cả thời gian đọc ghi ổ cứng: Chỉ bấm giờ quá trình suy luận `model(x)` trên GPU kèm `torch.cuda.synchronize()`.
================================================================================
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Tính diện tích dưới đường cong Precision-Recall (AUC-PR) theo chuẩn 11 điểm hoặc all-points.
    """
    raise NotImplementedError("Cần được Tú cài đặt compute_ap.")


def calculate_map(
    detections: List[np.ndarray],
    ground_truths: List[np.ndarray],
    iou_thresh: float = 0.5,
    num_classes: int = 2
) -> Dict[str, float]:
    """
    Tính AP cho từng lớp (hat, person) và mAP@0.5.
    """
    raise NotImplementedError("Cần được Tú & Sơn cài đặt calculate_map.")


def compute_prf1(
    tp: int, fp: int, fn: int
) -> Tuple[float, float, float]:
    """
    Tính Precision, Recall và F1-Score từ True Positive, False Positive, False Negative.
    """
    raise NotImplementedError("Cần được Tú cài đặt compute_prf1.")


def measure_fps(
    model: torch.nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 608, 608),
    device: str = "cuda",
    warmup: int = 50,
    iterations: int = 200
) -> float:
    """
    Đo tốc độ suy luận FPS chính xác của mô hình trên GPU với torch.cuda.synchronize().
    """
    raise NotImplementedError("Cần được Tú & Sơn cài đặt measure_fps.")


if __name__ == "__main__":
    print("[TODO] utils/metrics.py: Chạy unit test kiểm tra tính toán AP và mAP.")

"""
================================================================================
MODULE: utils/plots.py - TRỰC QUAN HÓA KẾT QUẢ: LOSS CURVE, PR CURVE & DEMO ẢNH
================================================================================
MỤC ĐÍCH:
    - Xuất các biểu đồ trực quan hóa dữ liệu phục vụ báo cáo khoa học và slide thuyết trình:
      1. Đồ thị hàm mất mát (Loss Curves): Đối chiếu tốc độ hội tụ giữa CIoU (Model-5)
         và SIoU (Model-6) theo Hình 9 của bài báo.
      2. Đường cong Precision-Recall (PR Curve): Cho cả 2 nhãn hat và person theo Hình 8 của bài báo.
      3. Vẽ hộp bao nhận diện (Bounding Box Overlay): Vẽ nhãn dự đoán (màu xanh cho 'hat',
         màu đỏ cho 'person') kèm độ tin cậy confidence trên ảnh thực tế theo Hình 11 bài báo.

PHÂN CÔNG TRÁCH NHIỆM (RACI):
    - Người thực thi (Responsible): Tú (Training & Plots Developer)
    - Người chịu trách nhiệm (Accountable): T.A (Project Lead)
    - Nhận bàn giao (Informed): Toàn đội (sử dụng hình ảnh để đưa vào Báo cáo tổng kết M5)

GIAO DIỆN & ĐẶC TẢ ĐẦU RA:
    - `plot_loss_comparison(ciou_log_path, siou_log_path, output_png)`:
      Lưu biểu đồ so sánh Loss tại `results/plots/loss_ciou_vs_siou.png`.
    - `plot_pr_curve(recalls, precisions, class_names, output_png)`:
      Lưu biểu đồ PR Curve tại `results/plots/pr_curve.png`.
    - `visualize_detections(image_path, boxes, labels, scores, output_png)`:
      Lưu ảnh trực quan hóa bounding box tại `results/detections/`.

TIÊU CHÍ NGHIỆM THU (DoD M5):
    [ ] Đồ thị thể hiện rõ nét, font chữ dễ đọc, có chú giải Legend, nhãn trục X (Epochs), trục Y (Loss).
    [ ] Hộp bao vẽ chính xác, nhãn hiển thị rõ ràng trên cả ảnh ban ngày lẫn thiếu sáng.
================================================================================
"""

from pathlib import Path
from typing import List, Dict, Any
import numpy as np


def plot_loss_comparison(
    log_model5: str | Path,
    log_model6: str | Path,
    save_path: str | Path = "results/plots/loss_comparison.png"
) -> None:
    """
    Vẽ biểu đồ đối chiếu tốc độ giảm loss giữa Model-5 (CIoU) và Model-6 (SIoU) (Hình 9 bài báo).
    """
    raise NotImplementedError("Cần được Tú cài đặt plot_loss_comparison.")


def plot_pr_curve(
    pr_data: Dict[str, Any],
    save_path: str | Path = "results/plots/pr_curve.png"
) -> None:
    """
    Vẽ đường cong Precision - Recall cho 2 lớp hat và person (Hình 8 bài báo).
    """
    raise NotImplementedError("Cần được Tú cài đặt plot_pr_curve.")


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    classes: np.ndarray,
    scores: np.ndarray,
    save_path: str | Path = None
) -> np.ndarray:
    """
    Vẽ bounding box lên ảnh với màu xanh lá cho 'hat' (0) và màu đỏ cho 'person' (1).
    """
    raise NotImplementedError("Cần được Tú cài đặt draw_bounding_boxes.")


if __name__ == "__main__":
    print("[TODO] utils/plots.py: Chạy thử nghiệm vẽ biểu đồ mẫu.")

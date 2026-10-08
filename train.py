"""
================================================================================
SCRIPT: train.py - LUỒNG HUẤN LUYỆN 2 GIAI ĐOẠN (TWO-STAGE TRANSFER LEARNING)
================================================================================
MỤC ĐÍCH:
    - Thực thi quy trình huấn luyện 200 epochs theo đúng Mục 4.2 của bài báo MDPI Sensors 2023:
      + GIAI ĐOẠN 1 (50 Epochs): Đóng băng trọng số Backbone (Freeze Backbone)
        * Batch size: 16
        * Learning rate: 0.001 (Optimizer Adam)
        * Scheduler: CosineAnnealingLR (T_max=50)
        * Mục tiêu: Ổn định các tầng ngẫu nhiên của Neck và Head mà không làm hỏng đặc trưng tiền huấn luyện.
      + GIAI ĐOẠN 2 (150 Epochs): Mở băng toàn bộ mạng (Unfreeze All Parameters)
        * Batch size: 8 (giảm batch size để tránh tràn bộ nhớ GPU VRAM Colab khi lan truyền ngược toàn mạng)
        * Learning rate: 0.0001 (giảm 10 lần để tinh chỉnh trọng số nhẹ nhàng)
        * Scheduler: CosineAnnealingLR (T_max=150)
        * Mục tiêu: Tinh chỉnh liên kết giữa Backbone và Neck/Head đạt độ chính xác tối ưu.
    - Hỗ trợ Mixed Precision Training (`torch.cuda.amp.autocast()` và `GradScaler`)
      để tăng tốc độ đào tạo và tiết kiệm 50% bộ nhớ đồ họa.
    - Tự động lưu checkpoint `last.pth` và `best.pth` định kỳ 10 epoch.

PHÂN CÔNG TRÁCH NHIỆM (RACI):
    - Người thực thi (Responsible): Tú (Training Pipeline Developer)
    - Người chịu trách nhiệm (Accountable): T.A (Project Lead)
    - Tham vấn chuyên môn (Consulted): Sơn (Technical Mentor)
    - Nhận bàn giao & Huấn luyện song song (Informed):
      + Sơn: Huấn luyện Model-1 (Colab 1)
      + Hưng: Huấn luyện Model-2 & Model-3 (Colab 2)
      + Tiến: Huấn luyện Model-4 (Colab 3)
      + T.A: Huấn luyện Model-5 (Colab 4)
      + Tú: Huấn luyện Model-6 (Colab 5)

THAM SỐ DÒNG LỆNH (CLI ARGUMENTS):
    python train.py --config configs/model6.yaml --data configs/dataset.yaml --epochs 200 --resume path/to/last.pth

CẤU TRÚC LOGIC CHÍNH:
    1. Parse args & nạp file cấu hình model/dataset.
    2. Khởi tạo DataLoader (Train & Val).
    3. Xây dựng mô hình thông qua `build_model(cfg)` từ `models/yolo.py`.
    4. Nạp hàm mất mát `ComputeLoss(cfg)` từ `utils/loss.py`.
    5. Giai đoạn 1: Đóng băng `model.backbone.parameters()`, chạy 50 epoch với batch=16, lr=0.001.
    6. Giai đoạn 2: Mở băng toàn bộ, tái tạo optimizer với lr=0.0001, chạy 150 epoch với batch=8.
    7. Lưu `best.pth` dựa trên metric `mAP@0.5` trên tập Validation.

TIÊU CHÍ NGHIỆM THU (DoD M3 & M4):
    [ ] Chạy Dry Run trên 50 ảnh mẫu trong 2 epoch thành công mà không có lỗi lệch shape hay CUDA OOM.
    [ ] Quá trình Freeze và Unfreeze chuyển đổi mượt mà, optimizer cập nhật đúng các tham số.
    [ ] Huấn luyện đủ 200 epochs sinh ra file trọng số `best.pth`.

BẪY LỖI KỸ THUẬT CẦN TRÁNH:
    - Bẫy 1: Quên tạo lại optimizer sau khi mở băng backbone -> Làm backbone không được cập nhật gradient ở Giai đoạn 2!
    - Bẫy 2: CUDA OOM ở Giai đoạn 2 khi unfreeze: Khắc phục bằng cách hạ batch size từ 16 xuống 8 và bật `torch.cuda.amp`.
================================================================================
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình YOLOv4 2 giai đoạn")
    parser.add_argument("--config", type=str, default="configs/model6.yaml", help="Đường dẫn file cấu hình model")
    parser.add_argument("--data", type=str, default="configs/dataset.yaml", help="Đường dẫn file cấu hình dataset")
    parser.add_argument("--device", type=str, default="cuda", help="Thiết bị chạy (cuda hoặc cpu)")
    parser.add_argument("--resume", type=str, default=None, help="Đường dẫn checkpoint để resume")
    return parser.parse_args()


def train():
    args = parse_args()
    print(f"=== BẮT ĐẦU HUẤN LUYỆN: {args.config} ===")
    # HƯỚNG DẪN CHI TIẾT:
    # 1. Tú nạp cấu hình YAML từ args.config và args.data
    # 2. Xây dựng mô hình từ models/yolo.py
    # 3. Chạy Vòng lặp Giai đoạn 1 (Freeze Backbone, 50 Epochs, lr=0.001, batch=16)
    # 4. Chạy Vòng lặp Giai đoạn 2 (Unfreeze All, 150 Epochs, lr=0.0001, batch=8)
    # 5. Đánh giá sau mỗi epoch, ghi nhận mAP và lưu best.pth
    raise NotImplementedError("Script train.py cần được Tú (Training Developer) và T.A hoàn thiện.")


if __name__ == "__main__":
    train()

"""
================================================================================
SCRIPT: detect.py - SUY LUẬN NHẬN DIỆN THỰC TẾ & XUẤT ẢNH DEMO BÁO CÁO
================================================================================
MỤC ĐÍCH:
    - Chạy suy luận (Inference) trên ảnh thực tế hoặc video để kiểm tra khả năng nhận diện
      của mô hình trong điều kiện thực địa (công trường xây dựng, nhà máy).
    - Xuất các hình ảnh trực quan có vẽ bounding box (xanh lá cho 'hat', đỏ cho 'person')
      kèm điểm số tin cậy (Confidence Score) theo đúng Hình 11 trong bài báo.
    - Phục vụ trực tiếp cho tài liệu Báo cáo nghiệm thu và slide thuyết trình bảo vệ đề tài.

PHÂN CÔNG TRÁCH NHIỆM (RACI):
    - Người thực thi & Chịu trách nhiệm (R & A): T.A (Project Lead)
    - Nhận bàn giao (Informed): Toàn đội (lấy ảnh demo cho slide)

THAM SỐ DÒNG LỆNH (CLI ARGUMENTS):
    python detect.py --weights weights/model6_best.pth --config configs/model6.yaml --source data/test_samples/ --output results/detections/

CÁC BƯỚC THỰC HIỆN:
    1. Nạp mô hình (khuyến nghị Model-6) và trọng số `best.pth`. Chuyển sang `model.eval()`.
    2. Đọc ảnh từ `--source`, áp dụng `letterbox` đưa về kích thước 608x608.
    3. Thực hiện forward pass và giải mã bounding box qua `YOLOHead`.
    4. Áp dụng Non-Maximum Suppression (NMS) với `conf_thres=0.25` và `iou_thres=0.45`.
    5. Ánh xạ tọa độ bounding box ngược trở lại kích thước ảnh gốc.
    6. Dùng module `utils/plots.py` vẽ bounding box và nhãn lên ảnh.
    7. Lưu ảnh kết quả vào thư mục `--output`.

TIÊU CHÍ NGHIỆM THU (DoD M5):
    [ ] Nhận diện chính xác mũ bảo hộ trong các điều kiện: khoảng cách xa, thiếu sáng, bị che khuất một phần.
    [ ] Thời gian suy luận hiển thị kèm trên từng ảnh (Inference time in ms).
================================================================================
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Chạy nhận diện mũ bảo hộ trên ảnh hoặc video")
    parser.add_argument("--weights", type=str, default="weights/model6_best.pth", help="Đường dẫn file trọng số")
    parser.add_argument("--config", type=str, default="configs/model6.yaml", help="Đường dẫn file cấu hình model")
    parser.add_argument("--source", type=str, default="data/samples/", help="Đường dẫn file ảnh, video hoặc thư mục")
    parser.add_argument("--output", type=str, default="results/detections/", help="Thư mục lưu ảnh kết quả")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Ngưỡng tin cậy confidence")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="Ngưỡng NMS IoU")
    parser.add_argument("--device", type=str, default="cuda", help="Thiết bị chạy (cuda hoặc cpu)")
    return parser.parse_args()


def detect():
    args = parse_args()
    print(f"=== BẮT ĐẦU NHẬN DIỆN MŨ BẢO HỘ TRÊN: {args.source} ===")
    # HƯỚNG DẪN CHI TIẾT:
    # 1. T.A nạp mô hình từ models/yolo.py và trọng số args.weights
    # 2. Xử lý ảnh đầu vào bằng datasets/transforms.py
    # 3. Suy luận qua model và NMS qua models/head.py
    # 4. Vẽ bounding box qua utils/plots.py và lưu ảnh
    raise NotImplementedError("Script detect.py cần được T.A (Project Lead) hoàn thiện.")


if __name__ == "__main__":
    detect()

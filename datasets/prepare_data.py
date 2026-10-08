"""
================================================================================
SCRIPT: datasets/prepare_data.py - TIỀN XỬ LÝ & CHUẨN BỊ TẬP DỮ LIỆU SHWD
================================================================================
MỤC ĐÍCH:
    - Giải nén file nén dữ liệu `Dataset_Split-*.zip` sang thư mục cục bộ (SSD trên Colab).
    - Quét và kiểm định toàn bộ ảnh và nhãn: Đối chiếu với thống kê bài báo
      (7,004 ảnh SHWD đã làm sạch, 7,709 nhãn hat, 101,174 nhãn person).
    - Tạo cố định 3 file phân chia tập dữ liệu: `train.txt`, `val.txt`, `test.txt`
      với tỷ lệ khuyến nghị 80% - 10% - 10% và random_seed cố định (seed=42).

PHÂN CÔNG TRÁCH NHIỆM (RACI):
    - Người thực thi (Responsible): Sơn (Technical Mentor)
    - Người chịu trách nhiệm (Accountable): T.A (Project Lead)
    - Tham vấn (Consulted): Tú
    - Nhận bàn giao (Informed): Hưng, Tiến (sử dụng chung 1 bộ split duy nhất)

GIAO DIỆN & ĐẶC TẢ DỮ LIỆU:
    1. Input:
       - Đường dẫn file zip: `Dataset_Split-*.zip` hoặc thư mục dataset thô.
       - Tỉ lệ phân chia: train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
       - Random seed: int = 42
    2. Output:
       - Thư mục dữ liệu đã giải nén: `data/SHWD/images/` và `data/SHWD/labels/` (hoặc `/content/dataset/`)
       - File phân chia cố định:
         + `datasets/splits/train.txt` (danh sách đường dẫn tuyệt đối/tương đối tới ảnh train)
         + `datasets/splits/val.txt`   (danh sách đường dẫn tới ảnh validation)
         + `datasets/splits/test.txt`  (danh sách đường dẫn tới ảnh test - dùng cho eval.py)
       - Thống kê in ra màn hình hoặc ghi vào file log: số lượng ảnh, số lượng box mỗi nhãn.

QUY TRÌNH THỰC HIỆN CẦN LẬP TRÌNH:
    Bước 1: `extract_dataset(zip_path, target_dir)`:
            Giải nén an toàn, hiển thị thanh tiến trình tqdm, tránh giải nén lặp lại nếu đã tồn tại.
    Bước 2: `validate_and_count_labels(image_dir, label_dir)`:
            Đọc tất cả file nhãn (.xml hoặc .txt), kiểm tra tọa độ bounding box hợp lệ (xmin < xmax, ymin < ymax),
            đếm tổng số bbox nhãn 'hat' (0) và 'person' (1). Báo lỗi nếu số lượng lệch nghiêm trọng so với 7,004 ảnh.
    Bước 3: `create_fixed_splits(image_list, output_dir, seed=42, ratios=(0.8, 0.1, 0.1))`:
            Xáo trộn ngẫu nhiên với seed cố định, ghi đường dẫn vào train.txt, val.txt, test.txt theo chuẩn Linux path ('/').

TIÊU CHÍ NGHIỆM THU (DoD M1):
    [ ] 100% 5 thành viên sử dụng CHUNG 1 file chia train/val/test; tuyệt đối không đổi file split giữa chừng.
    [ ] File txt chứa đường dẫn hợp lệ, không chứa ảnh lỗi hoặc không tìm thấy nhãn tương ứng.

BẪY LỖI KỸ THUẬT CẦN TRÁNH:
    - Bẫy 1: Windows dùng dấu gạch ngược `\\`, Colab Linux dùng `/`. Cần chuẩn hóa tất cả đường dẫn sang posix (`/`).
    - Bẫy 2: Data leakage nếu không cố định random seed.
================================================================================
"""

from pathlib import Path
from typing import Tuple, Dict


def extract_dataset(zip_path: str | Path, target_dir: str | Path) -> None:
    """
    Giải nén file zip chứa tập dữ liệu SHWD vào thư mục đích (SSD trên Colab).
    
    Args:
        zip_path: Đường dẫn tới file nén .zip.
        target_dir: Thư mục đích giải nén.
    """
    # HƯỚNG DẪN: Sơn triển khai logic zipfile.ZipFile tại đây
    raise NotImplementedError("Chức năng extract_dataset cần được Sơn (Mentor) hoàn thiện.")


def validate_and_count_labels(data_dir: str | Path) -> Dict[str, int]:
    """
    Quét toàn bộ thư mục dữ liệu, kiểm tra tính toàn vẹn và đếm số lượng nhãn hat / person.
    
    Args:
        data_dir: Thư mục chứa ảnh và nhãn.
        
    Returns:
        dict: Thống kê {'num_images': int, 'num_hat': int, 'num_person': int}
    """
    # HƯỚNG DẪN: Đọc Pascal VOC XML hoặc YOLO txt, kiểm tra box nằm trong [0, width/height]
    raise NotImplementedError("Chức năng validate_and_count_labels cần được Sơn hoàn thiện.")


def create_fixed_splits(
    data_dir: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> None:
    """
    Chia ngẫu nhiên tập dữ liệu với seed cố định và xuất ra 3 file train.txt, val.txt, test.txt.
    
    Args:
        data_dir: Thư mục chứa dữ liệu đã kiểm định.
        output_dir: Thư mục lưu 3 file .txt.
        seed: Random seed cố định để đảm bảo tính tái lập (mặc định 42).
        ratios: Bộ tỉ lệ (train, val, test), tổng bằng 1.0.
    """
    # HƯỚNG DẪN: Sử dụng sklearn.model_selection.train_test_split hoặc numpy.random với seed
    raise NotImplementedError("Chức năng create_fixed_splits cần được Sơn hoàn thiện.")


if __name__ == "__main__":
    print("[TODO] datasets/prepare_data.py: Chạy script này tại Mốc M1 để chuẩn bị dữ liệu.")

"""
================================================================================
MODULE: datasets/shwd.py - PYTORCH DATASET & DATALOADER CHO SHWD
================================================================================
MỤC ĐÍCH:
    - Xây dựng lớp `SHWDDataset(torch.utils.data.Dataset)` để nạp ảnh và nhãn.
    - Hỗ trợ cả định dạng Pascal VOC (.xml) hoặc YOLO (.txt).
    - Cung cấp hàm `collate_fn` chuyên biệt để gom batch khi mỗi ảnh có số lượng bounding box khác nhau.
    - Cung cấp hàm `build_dataloader` tạo DataLoader tối ưu với đa luồng (num_workers), pin_memory trên GPU Colab.

PHÂN CÔNG TRÁCH NHIỆM (RACI):
    - Người thực thi (Responsible): Sơn (Technical Mentor)
    - Người chịu trách nhiệm (Accountable): T.A (Project Lead)
    - Tham vấn (Consulted): Tú (đảm bảo định dạng target khớp với hàm loss)
    - Nhận bàn giao (Informed): Toàn đội (Hưng, Tiến, Tú dùng để train và test)

GIAO DIỆN & ĐẶC TẢ DỮ LIỆU:
    1. Input:
       - `split_file`: Đường dẫn file danh sách ảnh (train.txt / val.txt / test.txt).
       - `img_size`: (608, 608).
       - `is_train`: bool (True khi huấn luyện để bật augmentation).
    2. Output:
       - Mỗi phần tử `dataset[i]` trả về:
         + `image`: torch.Tensor kích thước (3, 608, 608), dtype=torch.float32.
         + `targets`: torch.Tensor kích thước (N, 5) gồm [class_id, cx, cy, w, h] chuẩn hóa [0, 1].
       - Khi nạp qua `DataLoader(collate_fn=yolo_collate_fn)`:
         + `images_batch`: torch.Tensor (B, 3, 608, 608).
         + `targets_batch`: torch.Tensor (M, 6) gồm [image_idx_in_batch, class_id, cx, cy, w, h].

HƯỚNG DẪN CÀI ĐẶT:
    1. Lớp `SHWDDataset`:
       - `__init__`: Đọc danh sách đường dẫn ảnh từ file `.txt`. Tìm file nhãn tương ứng (cùng tên, khác đuôi .xml hoặc .txt).
       - `__len__`: Trả về tổng số ảnh trong tập.
       - `__getitem__(index)`:
         * Đọc ảnh bằng OpenCV hoặc PIL.
         * Đọc tọa độ và nhãn lớp từ file nhãn (hat = 0, person = 1).
         * Áp dụng `transforms` (Letterbox resize 608x608).
         * Trả về (image, targets).
    2. Hàm `yolo_collate_fn(batch)`:
       - Tách riêng danh sách images và targets.
       - Ghép images thành tensor (B, 3, 608, 608).
       - Thêm cột `batch_idx` vào targets và ghép lại thành tensor (M, 6).
    3. Hàm `build_dataloader(split_file, batch_size, is_train=True, num_workers=4)`:
       - Trả về đối tượng `torch.utils.data.DataLoader`.

TIÊU CHÍ NGHIỆM THU (DoD M1):
    [ ] Chạy thử nạp 1 batch: shape `images` phải là `(B, 3, 608, 608)`.
    [ ] Shape `targets` phải là `(M, 6)` với nhãn class chỉ gồm 0 hoặc 1.
    [ ] Tốc độ nạp dữ liệu mượt mà, không bị hiện tượng CPU bottleneck nghẽn GPU.

BẪY LỖI KỸ THUẬT CẦN TRÁNH:
    - Bẫy 1: Ảnh không có bounding box nào (ảnh âm tính): Phải xử lý trả về targets rỗng shape (0, 5) chứ không được crash!
    - Bẫy 2: Lỗi pin_memory khi bộ nhớ RAM Colab bị đầy: Đặt `pin_memory=True` chỉ khi GPU khả dụng.
================================================================================
"""

from pathlib import Path
from typing import List, Tuple, Any
import torch
from torch.utils.data import Dataset, DataLoader


class SHWDDataset(Dataset):
    """
    Dataset nạp ảnh và nhãn mũ bảo hộ Safety Helmet Wearing Dataset (SHWD).
    """
    def __init__(
        self,
        split_file: str | Path,
        img_size: int = 608,
        is_train: bool = True
    ):
        self.split_file = Path(split_file)
        self.img_size = img_size
        self.is_train = is_train
        # HƯỚNG DẪN: Sơn nạp danh sách đường dẫn ảnh từ split_file
        pass

    def __len__(self) -> int:
        raise NotImplementedError("Cần được Sơn cài đặt __len__.")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Trả về ảnh (3, 608, 608) và targets (N, 5) gồm [class, cx, cy, w, h].
        """
        raise NotImplementedError("Cần được Sơn cài đặt __getitem__.")


def yolo_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gom cụm các phần tử từ dataset thành batch, xử lý kích thước bounding box thay đổi động.
    
    Returns:
        images: (B, 3, 608, 608)
        targets: (M, 6) dạng [batch_idx, class_id, cx, cy, w, h]
    """
    raise NotImplementedError("Cần được Sơn cài đặt yolo_collate_fn.")


def build_dataloader(
    split_file: str | Path,
    batch_size: int = 16,
    img_size: int = 608,
    is_train: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Tạo DataLoader chuẩn hóa cho quá trình Train / Val / Test.
    """
    raise NotImplementedError("Cần được Sơn cài đặt build_dataloader.")


if __name__ == "__main__":
    print("[TODO] datasets/shwd.py: Chạy unit test nạp thử 1 batch từ DataLoader.")

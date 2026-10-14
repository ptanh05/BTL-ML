# HƯỚNG DẪN TRIỂN KHAI HUẤN LUYỆN TRÊN GOOGLE COLAB (DISTRIBUTED TRAINING)

Tài liệu này hướng dẫn 5 thành viên thiết lập môi trường và chạy huấn luyện 6 mô hình song song trên 5 tài khoản Google Colab theo kế hoạch **Mốc 4 (M4)**.

---

## 1. Phân chia tài nguyên & Tài khoản

| Thành viên | Tài khoản Colab | Mô hình đảm nhiệm | File cấu hình | Dung lượng mục tiêu |
| :--- | :--- | :--- | :--- | :--- |
| **Sơn** (Mentor) | Colab Account 1 | **Model-1** (Baseline YOLOv4) | `configs/model1.yaml` | 243.92 MB |
| **Hưng** | Colab Account 2 | **Model-2** & **Model-3** | `configs/model2.yaml`<br>`configs/model3.yaml` | 136.13 MB<br>38.75 MB |
| **Tiến** | Colab Account 3 | **Model-4** (Model-3 + CA) | `configs/model4.yaml` | 38.99 MB |
| **T.A** (Lead) | Colab Account 4 | **Model-5** (Model-4 + PB Module) | `configs/model5.yaml` | 41.88 MB |
| **Tú** | Colab Account 5 | **Model-6** (Mô hình cải tiến hoàn chỉnh) | `configs/model6.yaml` | 41.88 MB |

---

## 2. Quy trình 4 bước thực thi trên Colab

### Bước 1: Kết nối GPU & Mount Google Drive
```python
# Kiểm tra GPU nhận diện
!nvidia-smi

# Gắn kết Google Drive để tự động lưu checkpoint phòng ngừa timeout
from google.colab import drive
drive.mount('/content/drive')
```

### Bước 2: Clone Repository & Cài đặt Thư viện
```bash
!git clone https://github.com/ptanh05/BTL-ML.git
%cd BTL-ML
!pip install -r requirements.txt
```

### Bước 3: Tăng tốc I/O - Giải nén dữ liệu vào SSD cục bộ
> **Quan trọng**: Tuyệt đối không đọc trực tiếp 7,004 ảnh từ Google Drive vì sẽ nghẽn I/O dẫn đến GPU bị nhàn rỗi (GPU utilization < 20%).
Chạy cell chuẩn bị dữ liệu trong `train_colab.ipynb`. Cell này xử lý cả archive
có thư mục gốc `dataset/`, kiểm tra annotations, tạo split và sinh
`configs/dataset_colab.yaml` trỏ tới `/content/dataset`. Dùng file cấu hình này
cho cả train, resume, eval và detect để các lệnh đọc cùng một bộ dữ liệu.

### Bước 4: Khởi chạy Huấn luyện 2 Giai đoạn
```bash
# Ví dụ cho Tú huấn luyện Model-6:
!python train.py --config configs/model6.yaml --data configs/dataset_colab.yaml --backbone-weights /content/drive/MyDrive/BTL_ML/pretrained/pplcnet_paper.pth --output /content/drive/MyDrive/BTL_ML/checkpoints/model_6

# Trường hợp bị ngắt kết nối (Colab Timeout), chạy resume từ Drive:
!python train.py --config configs/model6.yaml --data configs/dataset_colab.yaml --resume /content/drive/MyDrive/BTL_ML/checkpoints/model_6/last.pth
```

---

## 3. Quy chuẩn sao lưu và bàn giao trọng số (DoD M4)
Chế độ `paper` yêu cầu pretrained PyTorch khớp backbone và SHWD đã sửa nhãn,
để chạy 50 epoch freeze rồi 150 epoch unfreeze. Dataset 7.581 ảnh đang có trong
repo không phải bản 7.004 ảnh của bài báo. Dùng `--dry-run` để kiểm tra thực thi,
hoặc `--from-scratch` để train thử trên dữ liệu hiện có; kết quả được ghi là
`smoke`/`development`. Khi resume chế độ development, giữ `--from-scratch`.
Khi resume và không truyền `--output`, checkpoint
mới được lưu cạnh checkpoint đang đọc trên Drive.

1. Tự động lưu checkpoint `last.pth` sau mỗi 10 epoch, khi validation loss cải thiện và ở epoch cuối vào thư mục `--output`.
2. Khi hoàn thành đủ 200 epochs, sao lưu file `model_6_best.pth` (hoặc ID model tương ứng) về thư mục Google Drive chung của nhóm:
   `MyDrive/BTL_ML_Shared_Weights/`
3. Bàn giao file trọng số cho **Tú & Sơn** để chạy script `eval.py` phục vụ nghiệm thu **Mốc 5 (M5)**.

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
```python
import zipfile, os

zip_path = '/content/drive/MyDrive/BTL_ML/Dataset_Split-20261005T022535Z-1-001.zip'
extract_path = '/content/dataset'

if not os.path.exists(extract_path):
    os.makedirs(extract_path, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(">> Đã giải nén xong dữ liệu vào SSD cục bộ /content/dataset!")
```

### Bước 4: Khởi chạy Huấn luyện 2 Giai đoạn
```bash
# Ví dụ cho Tú huấn luyện Model-6:
!python train.py --config configs/model6.yaml --data configs/dataset.yaml

# Trường hợp bị ngắt kết nối (Colab Timeout), chạy resume từ Drive:
!python train.py --config configs/model6.yaml --resume /content/drive/MyDrive/BTL_ML/checkpoints/model6_last.pth
```

---

## 3. Quy chuẩn sao lưu và bàn giao trọng số (DoD M4)
1. Tự động lưu checkpoint `last.pth` sau mỗi 10 epoch vào Google Drive chung.
2. Khi hoàn thành đủ 200 epochs, sao lưu file `modelX_best.pth` về thư mục Google Drive chung của nhóm:
   `MyDrive/BTL_ML_Shared_Weights/`
3. Bàn giao file trọng số cho **Tú & Sơn** để chạy script `eval.py` phục vụ nghiệm thu **Mốc 5 (M5)**.

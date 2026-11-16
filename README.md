# Vietnamese License Plate Pipeline

Chuẩn hoá dataset DET/REC, giữ detector YOLO26n-OBB đã train và so sánh OCR FP32 sau khi học cùng dataset.

**Luồng hiện tại:** `vn-plate-dataset_v2` → train/val sáu OCR trên RTX 5070 → chọn best → kiểm ONNX FP32 → benchmark bảy OCR trên i5-14600KF.

| Model | Xử lý trong lượt này |
|---|---|
| **VietLPR_v1** | Giữ nguyên trọng số đã chốt, trước đây tên VietLPR_v2; không train lại |
| CCT_S_VN | Fine-tune CCT-S-v1 trên ảnh toàn biển |
| PP-OCRv4_mobile | Fine-tune cả DET dòng và REC |
| PP-OCRv6_small | Fine-tune cả DET dòng và REC |
| PP-OCRv5_mobile | Fine-tune cả DET dòng và REC |
| PP-OCRv6_tiny | Fine-tune cả DET dòng và REC |
| LPRNet_VN | Fine-tune CTC trên ảnh toàn biển |

Model VietLPR_v1 cũ và bảng 14 pretrained không thuộc lượt so sánh này.
Tên V2 còn trong mã/manifest nguồn để truy vết model đã chốt; tên công bố hiện tại là **VietLPR_v1**.
YOLO26n-OBB giữ nguyên checkpoint và graph; không train lại detector toàn biển.

## Bắt đầu trên PC ở nhà

Hướng dẫn đầy đủ: **[docs/pc_training.md](docs/pc_training.md)**.
Windows 11, Python 3.11 x64, RTX 5070 12 GB để train; CPU i5-14600KF để đo.

```powershell
git pull origin main
.\scripts\setup_pc.ps1
# Chép riêng dataset và model đã chốt theo hướng dẫn, rồi:
.\.venv-train-torch\Scripts\python.exe -m ocr.adaptation.suite run
.\.venv-benchmark\Scripts\python.exe -m ocr.adaptation.benchmark run
```

Setup dùng Torch CUDA 12.8 và Paddle CUDA 12.9 trong môi trường riêng, kiểm GPU bằng forward/backward thật trước khi train dài.
Không cài các bộ này lên môi trường ThinkPad hiện tại. Kết quả mới không ghép Speed đo trên laptop.

| Notebook | Vai trò |
|---|---|
| [01_prepare_dataset](notebooks/01_prepare_dataset.ipynb) | Dựng/kiểm dataset v2 |
| [02_train_vietlpr](notebooks/02_train_vietlpr.ipynb) | Train sáu OCR, resume và export FP32 trên PC |
| [03_evaluate_models](notebooks/03_evaluate_models.ipynb) | Benchmark bảy OCR sau train và bảng cuối |
| [04_train_plate_detector](notebooks/04_train_plate_detector.ipynb) | Luồng detector nano đã chốt |
| [05_evaluate_full_pipeline](notebooks/05_evaluate_full_pipeline.ipynb) | YOLO → crop → bảy OCR; cần thêm text GT toàn cảnh |
| 06, 07 | Hồ sơ lượng tử hoá trên ThinkPad đã chốt; tách khỏi lượt FP32 này |

Notebook 02/03 mặc định không chạy train/benchmark. Đổi đúng cờ khi sẵn sàng.

## Cấu trúc

```text
configs/ocr/adaptation.json  Danh sách bảy OCR, nguồn, SHA và tham số PC
ocr/adaptation/              Dữ liệu, train theo nguồn, export, chuyển máy, benchmark
scripts/setup_pc.ps1         Cài và kiểm môi trường Windows/RTX 5070
requirements/pc-*.txt        Thư viện riêng cho Torch, Paddle và CPU benchmark
notebooks/                  Các bước có thể chạy và đọc kết quả
dataset_preparation/        Luồng chuẩn hoá DET/REC từ đầu đến cuối
plate_detection/            Dữ liệu OBB, train/export detector đã chốt
lpr/                        Ghép detector với các OCR FP32 mới
third_party/                Mã kiến trúc và tiền/hậu xử lý nguồn đã chốt
metadata/                   Recipe và SHA dataset
runs/ocr_adaptation/         Checkpoint, ONNX FP32, val và benchmark mới; ngoài Git
```

Nguồn tải cho train CCT/Paddle là checkout Git đã pin, nằm trong `third_party/*_training/` và không vendor lên Git.
Các helper cũ đang được hồ sơ lượng tử/kiểm tra nguồn gọi vẫn giữ; chúng không chọn danh sách model của bảng mới.

## Dữ liệu và đánh giá

Dataset giữ nguyên **864 train / 222 val / 124 test** ảnh DET và **1402 / 371 / 190** crop REC.
Không chia lại split, không train hoặc chọn hyperparameter trên test. CCT/LPRNet dùng ảnh DET toàn biển với text ghép theo dòng;
Paddle học DET từ polygon và REC từ crop đã chuẩn hoá.

Fine-tune chung dataset không đồng nghĩa cùng nguồn pretrained, kiến trúc, số phép tính hoặc lịch sử train.
CCT dùng checkpoint Keras CCT-S-v1 global chính thức vì bản VN công khai chỉ có ONNX;
LPRNet dùng trọng số VN, Paddle dùng checkpoint train chính thức. Nguồn và SHA được lưu trong config/frozen.
Không tuyên bố đã loại hết giao nhau giữa tập pretrained bên ngoài và test nội bộ.

Benchmark: **7 model × 5 vòng = 35 pass**, bốn lõi P vật lý, bốn intra-op/một inter-op, tắt spin wait,
OpenCV một luồng, batch một ảnh, warm-up 20; ảnh nạp sẵn RAM. Speed gồm toàn pipeline đến chuỗi cuối, loại đọc đĩa và tải model.
SD là độ lệch chuẩn mẫu của năm trung bình vòng; FPS = 1000/Speed. Rank: ACC giảm → CER tăng → Speed tăng.
Model chậm nhất trong bảng làm Speedup 1×. Không dùng box GT khi inference.

## File giữ riêng

Không push dataset, ZIP, ảnh nguồn, checkpoint, `.keras`, ONNX, venv hay `runs/`.
Để chuyển PC, chép `vn-plate-dataset_v2.zip` và `runs/pc_transfer/current_baselines.zip` riêng; `git pull` chỉ có code.
Giới hạn/truy vết dataset: [docs/dataset.md](docs/dataset.md). Quy trình mới: [docs/pc_training.md](docs/pc_training.md).

```powershell
.venv\Scripts\python.exe -B -m unittest discover -s tests/dataset -q
.venv-ocr\Scripts\python.exe -B -m unittest discover -s tests/ocr -q
.venv-ocr\Scripts\python.exe -B -m unittest discover -s tests/detection -q
```

# Vietnamese License Plate OCR Pipeline

Chuẩn hoá dataset biển số Việt Nam, train/inference theo nguồn VietLPR và so sánh hiệu năng OCR trên CPU.

Luồng chính: **Chuẩn hoá DET/REC → Train VietLPR → Evaluation → Bảng xếp hạng**.

## Bắt đầu từ đâu?

| Giai đoạn | Notebook | Kernel |
|---|---|---|
| 1. Rà nhãn/box, duyệt, crop/nắn REC, dựng và kiểm v2 | [01_prepare_dataset.ipynb](notebooks/01_prepare_dataset.ipynb) | `.venv` — Python 3.13 |
| 2. Train/val VietLPR_v1, VietLPR_v2 và xuất best ONNX | [02_train_vietlpr.ipynb](notebooks/02_train_vietlpr.ipynb) | `.venv-ocr` — Python 3.11 |
| 3. Chuẩn bị Paddle pretrained, test CPU và xem bảng | [03_evaluate_models.ipynb](notebooks/03_evaluate_models.ipynb) | `.venv-ocr` — Python 3.11 |

Trên máy đã có kết quả: mở notebook **03** và chạy từ đầu. Mặc định chỉ đọc kết quả đã lưu.
Khi mới clone, cần train/export V1/V2 và bật evaluation để tạo kết quả; trọng số và số đo không đi kèm repository.
Sau khi đổi cấu trúc, chọn lại đúng kernel và **Restart Kernel** trước khi chạy notebook mới.
Notebook **02** có cờ `RUN_TRAINING`, `RUN_EXPORT`, `RUN_SMOKE`; notebook **03** có `RUN_EVALUATION` và `RECALIBRATE`.
Bật đúng tác vụ cần thực hiện. Paddle được chuẩn bị bằng pretrained, không train lại.

## Cấu trúc dự án

```text
notebooks/                 Ba notebook theo thứ tự thực hiện
dataset_preparation/       Đọc nhãn, chuẩn hoá, siết/duyệt box, crop và dựng v2
ocr/                       Adapter train, inference, ONNX và evaluation
configs/ocr/               Cấu hình model, pretrained và nguồn đã chốt
requirements/              Thư viện tách theo môi trường
tests/dataset/             Kiểm thử chuẩn hoá dataset
tests/ocr/                 Kiểm thử train/inference/evaluation
docs/                      Hướng dẫn chi tiết và giới hạn
third_party/               Mã nguồn model/PaddleX theo phiên bản đã chốt
metadata/                  Recipe DET–REC, checksum và quyết định duyệt
sources/                   Ảnh REC nguồn còn giữ để tái tạo crop
vn-plate-dataset_v2/        Dataset DET/REC dùng train, val và test
vn-plate-dataset_v2.zip     Bản đóng gói dataset
runs/                      Checkpoint, ONNX, log, số đo và kết quả
```

Các thư mục dữ liệu và `runs/` giữ đường dẫn ổn định cho recipe/checkpoint hiện có.
Repository lưu code, cấu hình, metadata và **ZIP dataset v2**. Thư mục ảnh đã giải nén, `sources/`, môi trường Python,
trọng số và kết quả trong `runs/` được Git bỏ qua.

## Dataset đi kèm

[vn-plate-dataset_v2.zip](vn-plate-dataset_v2.zip) chứa đầy đủ ảnh và nhãn DET/REC cho train, val, test.
Sau khi clone, giải nén tại thư mục gốc:

```powershell
Expand-Archive -LiteralPath .\vn-plate-dataset_v2.zip -DestinationPath .
```

ZIP tạo sẵn thư mục `vn-plate-dataset_v2/`. SHA256:
`016c677fe03e39093f7f339e3d86de30eeb83ae788c20416d06a51fbe5e3b26e`.
Train/evaluation dùng trực tiếp dataset thành phẩm. Tái tạo crop từ nguồn cần thêm `sources/` được lưu riêng.

## Code theo từng luồng

| Phần | Các file chính |
|---|---|
| Chuẩn hoá và đề xuất box | `dataset_preparation/normalization.py`, `line_envelopes.py`, `text_bounds.py` |
| Crop, dựng, kiểm tra và đóng gói | `dataset_preparation/geometry.py`, `pipeline.py` |
| Xem ảnh DET/REC | `dataset_preparation/preview.py` |
| Train VietLPR theo nguồn | `ocr/train_torch.py`, `upstream.py`, `torch_models.py` |
| Checkpoint và export | `ocr/run_state.py`, `export.py` |
| Inference VietLPR | `ocr/runtime.py`, `ops_v1.py`, `third_party/VietLPR_v2/inference.py` |
| Chuẩn bị/chạy 7 bản Paddle | `ocr/paddle_suite.py`, `paddle_suite_convert.py`, `paddle_suite_runtime.py` |
| Chấm điểm và bảng tổng hợp | `ocr/evaluate.py`, `metrics.py`, `comparison.py` |

Mã gốc model nằm trong `third_party/`; code nối dữ liệu, môi trường và cách đo nằm trong `ocr/`.
Manifest ghi commit/SHA để đối chiếu nguồn. V1/V2 giữ loss, augment, EMA, crop và tiêu chí chọn best theo từng nguồn.
`configs/ocr/` gồm `vietlpr_v1.json`, `vietlpr_v2.json`, `paddle_suite.json` (7 bản Paddle) và `sources.json` (nguồn code).

## Dữ liệu

| Split | Ảnh DET | Crop REC |
|---|---:|---:|
| train | 864 | 1402 |
| val | 222 | 371 |
| test | 124 | 190 |
| Tổng | **1210** | **1963** |

Mỗi box DET tương ứng một dòng chữ; REC được crop/nắn từ nguồn và quad đã duyệt.
`metadata/samples.jsonl` giữ liên kết, nội dung nhãn, hai hệ tọa độ và nguồn crop.

**Đầu vào Roboflow:** workspace hiện không còn đầy đủ bản ảnh/nhãn Roboflow đầu tiên.
Notebook 01 tái tạo v2 từ dữ liệu, nguồn REC và quyết định đã giữ lại.
Đầu vào hỗ trợ cấu trúc OCR có `det_labels.txt`/`rec_labels.txt`; export YOLO/COCO cần chuyển annotation trước.
Chi tiết định dạng và giới hạn phục hồi: [docs/dataset.md](docs/dataset.md).

## Model và evaluation

- VietLPR_v1: repo `vietdai-bk/VietLPR_v2`, giữ tên V1 trong dự án để phân biệt.
- VietLPR_v2: notebook nguồn đã cung cấp, chốt bằng SHA256.
- PP-OCRv4: server, mobile; PP-OCRv5: server, mobile; PP-OCRv6: medium, small, tiny.

Bảng chính gồm **9 hệ thống**: `Rank | Model | ACC | CER | Speed | FPS | Speedup`.
Rank theo **ACC giảm dần → CER tăng dần → Speed tăng dần**. Model chậm nhất làm mốc **Speedup 1×**.

Đo toàn bộ DET → crop → REC trên cùng 124 ảnh test bằng ONNX FP32, CPU 4 luồng, OpenCV 1 luồng,
batch một ảnh biển, warm-up 20 và đo 5 lượt. ACC/CER tính toàn chuỗi biển sau chuẩn hoá; ảnh không detect được vẫn tính lỗi.
Speed loại thời gian đọc ổ đĩa và tải model. V2 gom các dòng trong cùng ảnh vào một batch REC; các hệ thống còn lại chạy tuần tự.
Đây là kết quả trên laptop với các pipeline/nguồn khởi tạo riêng; Speed có dao động giữa các lượt.

`runs/comparison_ranked.csv` là bảng tổng hợp. Dự đoán và số đo từng ảnh nằm trong `runs/<model>/` hoặc `runs/paddle_suite/<tên>/`.
Hướng dẫn source, train/resume, export và evaluation: [docs/ocr.md](docs/ocr.md).

## Môi trường và kiểm thử

| Môi trường | File thư viện | Mục đích |
|---|---|---|
| `.venv` | `requirements/dataset-dev.txt` | Dataset và notebook 01 |
| `.venv` — nhánh DB tuỳ chọn | `requirements/dataset-bounds.txt` | Hỗ trợ đề xuất box bằng DB |
| `.venv-ocr` | `requirements/ocr-torch.txt` | Train GPU V1/V2, export và evaluation CPU |
| `.venv-ocr-paddle` | `requirements/ocr-paddle.txt` | Chuyển pretrained Paddle v4 và đối chiếu số học |

`requirements/dataset.txt` là thư viện dataset cơ bản; `requirements/ocr-cpu.txt` là phần dùng chung của môi trường OCR.

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests/dataset -v
.venv-ocr\Scripts\python.exe -m unittest discover -s tests/ocr -v
.venv\Scripts\python.exe -m dataset_preparation.pipeline validate
.venv-ocr\Scripts\python.exe -m ocr.comparison
```

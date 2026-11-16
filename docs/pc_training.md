# Train sáu OCR trên PC Windows 11

Máy train: RTX 5070 12 GB. Máy benchmark: cùng PC, CPU Intel Core i5-14600KF.
Đầu ra **ONNX FP32**. Không lượng tử hoá trong lượt này. Không train lại YOLO26n-OBB hoặc VietLPR đã chốt.

## 1. Chuẩn bị và pull code

Cài Git, Python **3.11 x64**, driver NVIDIA mới hỗ trợ CUDA 12.9. Kiểm `nvidia-smi` nhận RTX 5070.
Các wheel đã chứa runtime CUDA; không cần tự thay toàn bộ CUDA trên Windows.

```powershell
git clone https://github.com/namtrungdn310/vn-lpr-pipeline.git
cd vn-lpr-pipeline
# Nếu đã clone:
# git pull origin main
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\scripts\setup_pc.ps1
```

Setup tạo ba môi trường độc lập:

| Thư mục | Vai trò |
|---|---|
| `.venv-train-torch` | Torch 2.8 CUDA 12.8, Keras backend Torch; CCT/LPRNet |
| `.venv-train-paddle` | Paddle 3.1.1 CUDA 12.9 + Paddle2ONNX 2.1.0; DET/REC và export |
| `.venv-benchmark` | ONNX Runtime 1.20.1 CPU; không cần GPU |

Script kiểm forward/backward CUDA thật. Nếu kiểm thất bại, dừng tại đó, không chạy train dài.
Paddle và Torch dùng runtime riêng, không dùng lại Paddle 2.6/CUDA 12.0 của ThinkPad.
Pin Paddle 3.1.1 vì wheel Paddle2ONNX Windows không tương thích DLL Paddle 3.2.x;
đã kiểm import và chuyển PIR → ONNX với cặp 3.1.1/2.1.0, không tự nâng riêng một gói.

Nguồn cài: [PyTorch Blackwell/cu128](https://pytorch.org/blog/pytorch-2-7/),
[Paddle Windows cu129](https://github.com/PaddlePaddle/docs/blob/develop/docs/install/pip/windows-pip_en.md),
[Paddle2ONNX Windows](https://github.com/PaddlePaddle/Paddle2ONNX/releases/tag/v2.1.0).

## 2. Chép dữ liệu và trọng số đã chốt riêng

GitHub không chứa ảnh hay model. Từ ThinkPad, chép qua USB/ổ mạng:

1. `vn-plate-dataset_v2.zip` ở thư mục gốc dự án.
2. `runs/pc_transfer/current_baselines.zip` đã tạo sẵn: model VietLPR mới mang tên `VietLPR_v1`, checkpoint/ONNX YOLO.

Đặt hai ZIP ở gốc clone trên PC, rồi chạy:

```powershell
Get-FileHash .\vn-plate-dataset_v2.zip -Algorithm SHA256
Expand-Archive -LiteralPath .\vn-plate-dataset_v2.zip -DestinationPath .
.\.venv-train-torch\Scripts\python.exe -m ocr.adaptation.transfer install .\current_baselines.zip
.\.venv-train-torch\Scripts\python.exe -m ocr.adaptation.suite prepare
```

SHA ZIP dataset: `016c677fe03e39093f7f339e3d86de30eeb83ae788c20416d06a51fbe5e3b26e`.
Script kiểm từng file theo `metadata/baseline.json`. Không cần copy `sources/` để train dataset thành phẩm.
Muốn tạo lại gói transfer trên ThinkPad:

```powershell
.\.venv-ocr\Scripts\python.exe -m ocr.adaptation.transfer create
```

VietLPR mới là **chính model trước đây tên VietLPR_v2**, không lấy nhầm model VietLPR_v1 cũ.
Các graph được sao nguyên byte và kiểm SHA. Tên gốc trong manifest chỉ phục vụ truy vết.

## 3. Train và resume

```powershell
.\.venv-train-torch\Scripts\python.exe -u -B -m ocr.adaptation.suite run
```

Thứ tự: CCT_S_VN → PP-OCRv4_mobile → PP-OCRv6_small → PP-OCRv5_mobile → PP-OCRv6_tiny → LPRNet_VN.
Paddle train DET rồi REC bằng vòng train upstream. CCT/LPRNet học toàn biển từ split DET, không crop bằng GT khi test.
Lệnh gọi đúng môi trường mỗi model. Tắt sleep trong Windows khi chạy dài; giữ thông thoáng và không chơi game đồng thời.

Nếu bị ngắt, chạy lại **cùng lệnh**. LPRNet/CCT dùng checkpoint epoch đã hoàn tất;
Paddle tiếp tục `latest` cùng optimizer/state nguồn. Đã có `frozen.json` hợp lệ thì bỏ qua model đó.
Không thay epochs/batch/config giữa lượt resume. Muốn chạy riêng:

```powershell
.\.venv-train-torch\Scripts\python.exe -u -B -m ocr.adaptation.suite run --model CCT_S_VN
.\.venv-train-torch\Scripts\python.exe -u -B -m ocr.adaptation.suite run --model PP-OCRv6_small
```

Mặc định tối đa 100 epoch. CCT/LPRNet dừng nếu 20 epoch không cải thiện val ACC/CER;
Paddle chạy lịch nguồn đã cấu hình, chọn best DET theo hmean và REC theo acc trên val.
Không đặt yêu cầu mọi model phải có cùng số epoch mới được so sánh; báo cáo rõ lịch train và nguồn khởi tạo.
Batch PC: DET 16, REC/CCT/LPRNet 64, workers 0. Nếu hết VRAM, chỉ đổi config **trước khi bắt đầu lượt mới**.
Model và log nằm tại `runs/ocr_adaptation/<model>/` và `runs/ocr_adaptation/logs/`.

CCT dùng nguyên checkpoint Keras CCT-S-v1 global chính thức. Không giả định ONNX VN có thể tiếp tục train.
Tiền xử lý CCT giữ CLAHE/denoise nguồn; ảnh nhỏ được phóng trước để giữ đủ mẫu train/val và áp dụng cùng cách khi inference.
Các model giữ dictionary nguồn để nạp head pretrained. Không tự đổi O thành 0.

## 4. Export và chốt model

Suite tự export sau khi train từng model; chỉ nhận ONNX FP32, không FP16/QDQ/MatMulNBits.
Đối chiếu số học native/ONNX trên ba mẫu **val**, không dùng test để quyết định export hay chọn best.
Có thể chạy lại export riêng khi train đã xong:

```powershell
.\.venv-train-torch\Scripts\python.exe -m ocr.adaptation.suite export --model CCT_S_VN
.\.venv-train-paddle\Scripts\python.exe -m ocr.adaptation.suite export --model PP-OCRv6_small
```

| Model | File đầu ra |
|---|---|
| VietLPR_v1 | `runs/ocr_adaptation/VietLPR_v1/onnx/{det,rec}.onnx` |
| Bốn Paddle | `runs/ocr_adaptation/<model>/onnx/{det,rec}.onnx` |
| CCT/LPRNet | `runs/ocr_adaptation/<model>/onnx/model.onnx` |

Mỗi model có `frozen.json`: nguồn, SHA, dataset và kết quả kiểm chuyển đổi. Giữ cùng các YAML/JSON trong thư mục ONNX khi chuyển máy.

## 5. Đo benchmark mới trên i5-14600KF

Đợi suite train xong, đóng các tác vụ GPU/CPU nặng, giữ chế độ nguồn Windows ổn định.

```powershell
.\.venv-benchmark\Scripts\python.exe -u -B -m ocr.adaptation.benchmark run
.\.venv-benchmark\Scripts\python.exe -B -m ocr.adaptation.benchmark table
```

35 pass = bảy model × năm vòng. Mỗi pass tiến trình riêng, session mới, bốn lõi P vật lý;
bốn intra-op/một inter-op, tắt spin wait; OpenCV một luồng; warm-up 20 trên val; test đủ 124 ảnh.
Ảnh nạp trước vào RAM, không I/O/log trong vòng đo. Batch một ảnh biển; VietLPR gom các dòng trong ảnh vào batch REC như nguồn.
Speed gồm tiền/hậu xử lý và OCR đến chuỗi cuối. Model lỗi không đọc được vẫn tính lỗi, không loại khỏi ACC/CER.

ACC/CER dùng uppercase alphanumeric, bỏ dấu phân cách. Speed ± SD là trung bình và độ lệch chuẩn mẫu của năm trung bình vòng.
FPS = 1000/Speed. Speedup dùng model chậm nhất trong bảng làm 1×. Rank ACC giảm → CER tăng → Speed tăng.
Không dùng kết quả ThinkPad hoặc bảng 14 pretrained cũ để lấp kết quả chưa đo.

Bị ngắt: đọc đường dẫn trong `runs/ocr_adaptation/active_benchmark.json`, rồi:

```powershell
.\.venv-benchmark\Scripts\python.exe -u -B -m ocr.adaptation.benchmark run --run-dir runs/ocr_adaptation/benchmark/<run>
```

Pass đủ được kiểm rồi dùng lại; pass dang dở đo lại toàn bộ ảnh. Không đọc tiếp từng ảnh từ checkpoint benchmark.
Bảng mới chỉ công bố sau đủ 35 pass, tại `comparison.csv`, `comparison.md`, `results.json` trong thư mục lượt đo.
Notebook 03 đọc cùng các bản ghi này. Chưa có kết quả mới thì hiển thị **chưa đo**, không hiện số cũ.

## 6. Giới hạn khi viết paper

Test nội bộ đã được xem trong các vòng benchmark trước; train lại không biến nó thành tập đánh giá mù mới.
Cùng split không xoá khác biệt nguồn pretrained. Không tuyên bố so sánh thuần kiến trúc hay mọi nguồn pretrained độc lập với test.
Đây là benchmark CPU trên ảnh biển đã crop, chưa phải camera → YOLO → OCR hoặc kết quả Pi5.
YOLO giữ nguyên model; muốn báo tốc độ cả pipeline cần text GT toàn cảnh và lượt đo riêng.
Kết quả lượng tử hoá notebook 06/07 thuộc lượt ThinkPad trước đây, không trộn vào bảng FP32 trên PC.

## 7. Dọn baseline cũ trên ThinkPad

Code đã bỏ config/nhánh train kiến trúc VietLPR cũ, chuyển bảng chính sang bảy OCR và không tải lại repo cũ.
Trước khi về PC, có thể dọn các model pretrained cũ, clone VietLPR cũ và file kiểm tra tạm:

```powershell
.\scripts\cleanup_retired_local.ps1
.\scripts\cleanup_retired_local.ps1 -Apply
```

Script mặc định chỉ xem trước; khi Apply kiểm VietLPR mới, gói transfer và khả năng đọc hai bảng lượng tử cũ trước/sau xoá.
Giữ dataset, `runs/ocr_adaptation/`, `runs/pc_transfer/`, model YOLO và model lượng tử đã chốt.
Các file mã nguồn còn được fingerprint của hồ sơ lượng tử dùng sẽ giữ nguyên để kiểm chứng số đo, không còn là danh sách OCR chính.

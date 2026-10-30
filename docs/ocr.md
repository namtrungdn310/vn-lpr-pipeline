# Train và so sánh OCR

Mở `notebooks/02_train_vietlpr.ipynb` để train/export hoặc `notebooks/03_evaluate_models.ipynb` để evaluation. Cả hai dùng kernel `.venv-ocr`.
Paddle dùng pretrained chính thức qua `.venv-ocr-paddle`; V1/V2 train GPU. Đánh giá ONNX FP32 trên CPU.
Dataset đã là ảnh biển. Luồng này không cần YOLO, video, tracking hoặc tạo dữ liệu giả lập.

## Nguồn đã chốt

| Hệ thống | Nguồn | Mã nhận diện |
|---|---|---|
| VietLPR_v1 | [vietdai-bk/VietLPR_v2](https://github.com/vietdai-bk/VietLPR_v2) | commit `ff3a6b3d65b64463ae61b251abb9e117c3e0ea56` |
| VietLPR_v2 | `VietLPR_v2_colab.ipynb` đã cung cấp | SHA256 `aafe7b2f63c0c6346ac00f807c96ee249de186e37021a0d0c81daf915f2adbed` |
| 7 bản PaddleOCR | V4 server/mobile, V5 server/mobile, V6 medium/small/tiny | URL/revision/SHA từng DET/REC trong `configs/ocr/paddle_suite.json` |

`configs/ocr/sources.json` ghi nguồn VietLPR và commit PaddleX dùng cho tiền/hậu xử lý.

VietLPR có sẵn đầy đủ DET → crop → REC. `ocr/` chỉ nối dữ liệu, cấu hình, checkpoint và đo chung.
V2 giữ các hàm nguồn trong `third_party/VietLPR_v2/source.py`, cùng nguyên lớp ONNX `LPR` trong `inference.py`.
`source_manifest.json` cho biết cell gốc và checksum AST. Không chạy cell cài đặt, Drive, download hoặc YOLO của notebook.
V1 dùng trực tiếp dataset/loss/validation/lịch LR trong repo; tám helper hình học ở `ops_v1.py` giữ logic upstream.

## Quy trình hiện tại — revision 3

| Bước | V1 theo repo | V2 theo notebook |
|---|---|---|
| DET | 320×320; fill/dilate mask | 128×128; shrink-map DBNet |
| Loss DET | 0,4 BCE + 0,6 Dice | BCE/OHEM + Dice |
| Augment | Độ sáng/tương phản và noise theo dataset gốc; REC không thêm augment | Hình học cùng ảnh/polygon; photo augment; jitter quad REC |
| Dữ liệu REC | Crop REC đã có | Crop REC + crop online từ GT và box DET dự đoán khớp IoU ≥0,5 trên train |
| Crop/suy luận | Pad box → warp → letterbox | Fit/unclip → warp + resize một lần; mặc định **stretch** |
| Tối ưu | AdamW; warm-up 5 epoch + cosine theo epoch | AdamW beta=(0,9;0,99); bỏ decay BN/bias/pos; warm-up 2 epoch + cosine theo step |
| EMA | Không | Có; val và export dùng EMA |
| Best DET | Mask IoU, phá hoà bằng val loss | H + 0,001×mIoU |
| Best REC | CER trung bình từng crop, phá hoà bằng val loss | ACC − 0,1×CER toàn ký tự |
| Số epoch | DET 30, REC 100; không thêm early stop | DET/REC tối đa 200, patience 20 |
| Tune E2E | Giữ hậu xử lý mặc định nguồn | Lưới threshold/unclip/box-threshold/pad-x đúng notebook, chỉ trên val |
| REC inference | Tuần tự từng dòng | Gom các dòng trong một batch như nguồn |

Đã bỏ chọn ghép `best/last`, LR plateau và letterbox chung của revision 2.
Charset V1/V2 theo nguồn: `0123456789ABCDEFGHJKLMNPRSTUVXYZ` (32 ký tự, blank CTC riêng).
V2 nguồn lấy mặc định hợp với ký tự train; dataset hiện tại không có ký tự train nằm ngoài bộ này.
Không tự sửa O→0, không bỏ ảnh không detect được khi chấm điểm.

## Điều chỉnh cần thiết cho dự án

- Dùng nguyên train/val/test của dataset v2; thay random split của V1, không dùng fallback val→test của notebook.
- Batch giảm cho GPU laptop 4 GB: V1 DET/REC 8/64; V2 32/64. `num_workers=0` cho Windows.
- V1 thêm checkpoint optimizer/scaler/RNG để resume; công thức loss, lịch LR và cách chọn best theo nguồn.
- V2 gọi nguyên `train_model()` đã tách. Checkpoint gốc nằm ở `runs/vietlpr_v2/native/`; bản bọc có thông tin dữ liệu phục vụ kiểm tra/export.
- V1 thêm ONNX bằng phép mean tương đương adaptive pooling; V2 gộp RepConv và xuất batch động như notebook.
- Báo cáo chung dùng ACC/CER theo toàn chuỗi biển và CER tổng số lỗi/tổng ký tự. Đây là cách chấm chung, không đổi tiêu chí chọn best riêng của nguồn.

Log val V1 hiển thị acc/cer dạng phần trăm; V2 dạng 0–1 theo nguồn. Bảng tổng hợp chín hệ thống dùng phần trăm.
Batch benchmark là **một ảnh biển**. V2 nhận dạng cùng lúc các dòng của ảnh đó; metadata ghi rõ `rec_batching`.
Đây là triển khai theo logic nguồn trên dataset/phần cứng của dự án, không hứa trọng số hoặc ACC giống tác giả.

## Chạy notebook

1. `01_prepare_dataset.ipynb`: chuẩn hoá và kiểm tra dữ liệu, kernel `.venv`.
2. `02_train_vietlpr.ipynb`: kiểm source/config, train/val và export V1/V2, kernel `.venv-ocr`.
3. `03_evaluate_models.ipynb`: chuẩn bị Paddle pretrained, chốt val, test CPU và xếp hạng chín hệ thống.

Notebook 02 mặc định `RUN_TRAINING=False`, `RUN_EXPORT=False`, `RUN_SMOKE=False`.
Bật tác vụ cần chạy. Train và smoke dùng GPU CUDA; export dùng best đã chọn theo nguồn.
Notebook 03 mặc định `RUN_EVALUATION=False`, chỉ đọc các kết quả đã lưu. Bật để chuẩn bị/tiếp tục đo;
`RECALIBRATE=True` nếu cần chọn lại hậu xử lý trên val. Không dùng test để chọn ngưỡng.

VietLPR nằm trong `runs/<model>/onnx/`; bảy bản Paddle nằm trong `runs/paddle_suite/<tên>/onnx/`.
Đổi vị trí file JSON cấu hình không làm khác lượt train; các giá trị mô hình, dữ liệu và hyperparameter vẫn phải khớp.

## Lệnh tương đương

```powershell
.venv-ocr\Scripts\python.exe -m pip install -r requirements/ocr-torch.txt
.venv-ocr\Scripts\python.exe -m ocr.bootstrap
.venv-ocr\Scripts\python.exe -m unittest discover -s tests/ocr -v
.venv-ocr\Scripts\python.exe -m ocr.smoke --device cuda --full-size

.venv-ocr\Scripts\python.exe -m ocr.train_torch --model vietlpr_v1 --task det --device cuda
.venv-ocr\Scripts\python.exe -m ocr.train_torch --model vietlpr_v1 --task rec --device cuda
.venv-ocr\Scripts\python.exe -m ocr.train_torch --model vietlpr_v2 --task det --device cuda
.venv-ocr\Scripts\python.exe -m ocr.train_torch --model vietlpr_v2 --task rec --device cuda
```

CLI chạy lại phải dùng `--resume` hoặc `--run-dir` mới. Không đổi tổng số epoch giữa lượt đang resume.
Torch 2.7.1+cu126 trong `.venv-ocr`; Paddle GPU 2.6.2.post120 trong `.venv-ocr-paddle`. Giữ `.venv` riêng cho chuẩn hoá dữ liệu.

## File chính

| File | Vai trò |
|---|---|
| `upstream.py` | Nạp namespace nguồn và nối cấu hình |
| `train_torch.py` | Adapter train/split/resume; gọi dataset V1 và vòng train V2 gốc |
| `runtime.py` | Gọi pipeline từng nguồn bằng ONNX CPU |
| `export.py` | Xuất best; kiểm dữ liệu/quy trình và đối chiếu số học |
| `paddle_suite.py`, `paddle_suite_convert.py` | Chuẩn bị bảy bản Paddle, chuyển V4 sang ONNX và đối chiếu số học |
| `paddle_pretrained.py` | Giải nén đúng hai file inference từ gói V4 |
| `evaluate.py` | Tune val, khoá và test VietLPR V1/V2 |
| `comparison.py` | Kiểm và xếp hạng kết quả của chín hệ thống |
| `notebook.py`, `run_state.py` | Log notebook, tiếp tục hoặc lưu lượt cũ |
| `tests/ocr/test_upstream_parity.py` | Kiểm mã đã tách, crop, mask, EMA và batch theo nguồn |

`runs/comparison_ranked.csv`: **Rank | Model | ACC | CER | Speed | FPS | Speedup**. Speed gồm DET, hậu xử lý, crop, REC và decode;
không tính đọc ảnh, load model hoặc vẽ. FPS=1000/Speed; Speedup lấy model chậm nhất làm mốc 1×.

## So sánh bảy bản PaddleOCR

Các phần chuẩn bị Paddle trong `notebooks/03_evaluate_models.ipynb` chạy độc lập với train V1/V2:

1. Chuẩn bị v4 server/mobile, v5 server/mobile, v6 medium/small/tiny.
2. Kiểm tra ngưỡng gốc trên **val**, khoá trọng số, dictionary, cấu hình và code.
3. Test toàn bộ bảy bản trên CPU, xuất bảng **ACC | CER | Speed | FPS | Speedup**.

Tất cả dùng model pretrained chính thức, không train lại. V5/V6 tải ONNX do PaddlePaddle phát hành;
v4 chuyển bằng `.venv-ocr-paddle`, đối chiếu số học Paddle/ONNX trên ba ảnh val mỗi nhánh.
V5/V6 kiểm checksum, kích thước output/dictionary và chạy thử val; không gọi đó là kiểm chuyển đổi số học cục bộ.
URL, revision và SHA256 nằm trong `configs/ocr/paddle_suite.json`.

Mỗi dòng ghép DET và REC **cùng phiên bản, cùng kích thước**; v4 dùng REC đa ngôn ngữ chính thức,
khác English REC của baseline Paddle cũ. Pipeline giữ resize theo tỷ lệ, giới hạn cạnh dài 960 (`max`,
không phóng ảnh nhỏ), làm tròn bội 32; REC theo YAML gốc và pad động, chạy từng dòng.
Tắt các model bổ sung về hướng văn bản và sửa tài liệu. Bộ này không dùng resize vuông 320 của bảng cũ.

Cùng ONNX Runtime FP32, 4 luồng, OpenCV 1 luồng, batch 1 ảnh biển, warm-up 20, đo 5 lượt.
Cache DET chỉ dùng trong chọn ngưỡng trên val; lúc test luôn chạy đầy đủ DET → crop → REC.
Mặc định giữ ngưỡng của YAML từng model, không dò lưới. Có thể thêm `tune --search` trước test để
thử mặc định và cùng lưới threshold/unclip cho cả bảy bản; bước này có thể rất lâu với server.
Speedup = Speed của **PP-OCRv4_mobile trong bộ bảy bản** / Speed của model đang xét.
Chỉ ghép bảng khi dữ liệu, code và điều kiện đo khớp nhau; không trộn số cũ vào bảng mới.

Các file:

| File | Vai trò |
|---|---|
| `paddle_suite.py` | Download, kiểm model, chọn val, test và bảng so sánh |
| `paddle_suite_runtime.py` | Nối ONNX với tiền/hậu xử lý nguồn |
| `paddle_suite_convert.py` | Chuyển hai bản v4 trong môi trường Paddle sẵn có |
| `configs/ocr/paddle_suite.json` | Danh sách bảy bản, nguồn và cách đo |
| `third_party/PaddleOCR_suite/` | Các hàm PaddleX đã tách và ghi nguồn |

```powershell
.venv-ocr\Scripts\python.exe -m ocr.paddle_suite prepare
.venv-ocr\Scripts\python.exe -m ocr.paddle_suite tune
.venv-ocr\Scripts\python.exe -m ocr.paddle_suite test
.venv-ocr\Scripts\python.exe -m ocr.paddle_suite compare
```

Model: `runs/paddle_suite/<tên>/onnx/{det,rec}.onnx`.
Kết quả: `runs/paddle_suite/<tên>/test_result.json`, `test_predictions.json` và `runs/paddle_suite/comparison.csv`.
Không cần GPU cho phần này. Model và kết quả vẫn được Git bỏ qua.
Cell cuối ghép thêm VietLPR_v1 và VietLPR_v2 vào `runs/comparison_ranked.csv`: **Rank** theo ACC giảm dần,
CER tăng dần, rồi Speed tăng dần. Speedup của bảng này lấy model chậm nhất làm mốc 1×.
`benchmark_progress.json` lưu sau từng ảnh, ngoài thời gian đo. Nếu bị ngắt, chạy lại bước test để
tiếp tục số đo còn thiếu; mỗi lần mở lại model đều warm-up trước. Code/dữ liệu/model khác sẽ bị chặn.
Khoá tiến trình tránh hai cell cùng đo một model. Có đủ 5 số đo mỗi ảnh mới xuất kết quả chính thức.
Đã đủ 5 lượt thì chạy lại dùng số đo đã lưu. Muốn đo mới, cất `benchmark_progress.json`,
`test_result.json` và `test_predictions.json` của lượt cũ trước khi chạy lại bước test.

## Bảng tổng hợp

`ocr/comparison.py` đọc và kiểm tra kết quả của 7 bản Paddle cùng V1/V2, đối chiếu dự đoán/số đo gốc,
kiểm checksum model và điều kiện CPU trước khi ghép bảng. Rank theo ACC giảm dần, CER tăng dần, rồi Speed tăng dần.
Speedup = Speed của model chậm nhất / Speed của model đang xét. Model chậm nhất có Speedup 1×.
Đây là so sánh các hệ thống với pipeline/nguồn khởi tạo riêng, không phải chỉ so kiến trúc.

```powershell
.venv-ocr\Scripts\python.exe -m ocr.comparison
```

Kết quả được ghi vào `runs/comparison_ranked.csv`. Code và cấu hình baseline English REC đã được gỡ;
bảng chính chỉ đọc kết quả của chín hệ thống hiện tại.
Bảng riêng bảy bản Paddle ở `runs/paddle_suite/comparison.csv`.
Model, checkpoint, số đo và log của lượt hiện tại nằm trong `runs/`; hồ sơ sửa dataset v2 nằm trong `runs/data_corrections/`.

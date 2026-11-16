# YOLO26n-OBB: FP32, INT8 mixed và INT4 mixed

Mở notebook **07**, kernel `.venv-ocr`. Mặc định chỉ đọc kết quả.
Bảng có ba bản hiện tại, xếp **FPS giảm dần**; FP32 dùng chung bản ghi với notebook 04.
INT8 cũ không còn hiển thị trong bảng. Đây là detection toàn biển, chưa ghép OCR.

| Tên hiển thị | Phần lượng tử hoá | Phần giữ FP32 |
|---|---|---|
| `YOLO26n-OBB_FP32` | Không | Toàn bộ graph |
| `YOLO26n-OBB_INT8_mixed` | Conv/Sigmoid/Mul backbone block 0–10, static U8S8 | Neck/head và phép khác |
| `YOLO26n-OBB_INT4_mixed` | 58 Conv 1×1 phù hợp chuyển sang MatMulNBits W4A32 | 53 Conv còn lại, activation và phép khác |

Đổi Conv 1×1 sang MatMul bằng phép biến đổi đại số tương đương, kiểm sai số FP32 trước khi lượng tử hoá.
INT4 không phải toàn model 4-bit. INT8 được chọn bằng val và kiểm kernel thực thi trên CPU.
Hiệu chuẩn static INT8 chỉ dùng 256 ảnh train; không dùng val/test để lấy khoảng activation.
Giữ nguyên checkpoint đã train và ngưỡng suy luận đã chốt.

## Model và code

- FP32: `runs/plate_detection/yolo26n_obb/weights/best.onnx`.
- INT8 hiện tại: `runs/int8_optimization/fixed_cpu/yolo/selected/model.onnx`.
- INT4: `runs/plate_detection/yolo26n_obb/quantization/int4/model.onnx`.
- `plate_detection/quantization.py`: tạo model nền/INT4 và đánh giá OBB.
- `ocr/int8_optimization.py`, `yolo_int8_refinement.py`, `cpu_selection.py`: tạo, kiểm và chọn INT8 trên val.
- `ocr/optimized_benchmark.py`, `benchmark_display.py`: đo chung và xuất bảng.

## Cách đo

258 ảnh test, năm vòng, bốn lõi hiệu năng vật lý; ảnh nạp sẵn vào RAM.
Speed gồm tiền xử lý → ONNX → hậu xử lý OBB; loại đọc đĩa, tải model, vẽ và OCR.
mAP dùng evaluator OBB Ultralytics; P/R tại confidence 0,25 và polygon IoU ≥ 0,5.
Bảng lưu tại `runs/int8_optimization/benchmark/<run>/comparison_detection.{csv,md}`.

```powershell
.venv-ocr\Scripts\python.exe -B -m ocr.benchmark_display table --group detection
```

Quy trình đo, SD và resume: [benchmark.md](benchmark.md).
Chọn ứng viên/kiểm ngưỡng chất lượng: [int8_optimization.md](int8_optimization.md).
Model, log và kết quả trong `runs/` được Git bỏ qua.

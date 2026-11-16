# VietLPR_v1: FP32, INT8 mixed và INT4 mixed

Mở notebook **06**, kernel `.venv-ocr`. Mặc định chỉ đọc kết quả.
Bảng có ba bản hiện tại, xếp **FPS giảm dần**; Số đo này thuộc ThinkPad trước lượt adaptation, không dùng cho bảng notebook 03 mới.
INT8 cũ không còn hiển thị trong bảng.

| Tên hiển thị | Phần lượng tử hoá | Phần giữ FP32 |
|---|---|---|
| `VietLPR_v1_FP32` | Không | Toàn bộ DET/REC |
| `VietLPR_v1_INT8_mixed` | Năm MatMul trọng số cố định REC, dynamic INT8 | Toàn bộ DET, Conv REC và phép khác |
| `VietLPR_v1_INT4_mixed` | Năm MatMul trọng số cố định REC, W4A32, block 32 | Toàn bộ DET, Conv REC và phép khác |

INT8 được chọn sau khi so ứng viên trên val với cùng lõi CPU của benchmark.
INT4 dùng MatMulNBits thật; activation vẫn FP32. Hai bản đều là lượng tử hoá hỗn hợp.
Không train lại, thay charset, tune ngưỡng theo test hoặc sửa model FP32 gốc.

## Model và code

- FP32: `runs/vietlpr_v2/onnx/{det,rec}.onnx`.
- INT8 hiện tại: `runs/int8_optimization/fixed_cpu/vietlpr/selected/{det,rec}.onnx`.
- INT4: `runs/vietlpr_v2/quantization/int4/{det,rec}.onnx`.
- `ocr/quantization.py`: tạo model nền/INT4 và kiểm tra model đã chuẩn bị.
- `ocr/int8_optimization.py`, `int8_refinement.py`, `cpu_selection.py`: tạo, kiểm và chọn INT8 trên val.
- `ocr/optimized_benchmark.py`, `benchmark_display.py`: đo chung và xuất bảng.

Tên nội bộ trong manifest có thể giữ hậu tố lịch sử để kiểm checksum.
Tên trên bảng dùng FP32/INT8_mixed/INT4_mixed như trên.

Calibration của các ứng viên static chỉ dùng train; dynamic INT8 không dùng khoảng activation hiệu chuẩn.
Chọn cấu hình bằng toàn bộ 222 ảnh val; test chỉ dùng báo cáo sau khi chốt model.
Chi tiết chọn ứng viên: [int8_optimization.md](int8_optimization.md).

## Đo và đọc kết quả

Cùng 124 ảnh test, năm vòng, CPU và vòng đo như notebook 03;
Speed gồm DET → crop → REC và tiền/hậu xử lý, không gồm đọc đĩa/tải model.
Bảng lưu tại `runs/int8_optimization/benchmark/<run>/comparison_quantization.{csv,md}`.

```powershell
.venv-ocr\Scripts\python.exe -B -m ocr.benchmark_display table --group quantization
```

Quy trình đo/resume và định nghĩa SD: [benchmark.md](benchmark.md).
Lượng tử hoá giảm dung lượng nhưng không bảo đảm nhanh hơn; cần đo lại trên thiết bị triển khai.

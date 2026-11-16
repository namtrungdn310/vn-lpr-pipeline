# Tối ưu INT8 và benchmark chung

Code mới giữ nguyên các module/baseline cũ để không làm mất hiệu lực fingerprint lịch sử.
Không train lại, không thay dataset, preprocessing, ngưỡng hay cách sắp dòng/ký tự.

## Tạo ứng viên

- `ocr/int8_optimization.py`: hiệu chuẩn train, tạo QDQ S8S8/U8S8/U8U8,
  thử QDQ dùng chung/riêng và giữ head YOLO FP32; chấm toàn bộ val.
- `ocr/int8_refinement.py`: thử Conv/Relu hoặc MatMul riêng ở REC, static/dynamic,
  giữ DET FP32; chấm lại cả pipeline trên val.
- `ocr/yolo_int8_refinement.py`: thử lượng tử hoá cả Conv–Sigmoid–Mul trong vùng CNN
  để giảm chuyển đổi quanh SiLU; giữ head FP32. Nếu vòng đầu chưa đạt chất lượng,
  thử thêm 512 ảnh hiệu chuẩn train hoặc giữ stem/neck/head FP32; không hạ ngưỡng chất lượng.
- `configs/ocr/int8_optimization.json` và `int8_refinement.json`: thông số thử được lưu trước test.

UINT8 activation dùng dải bất đối xứng; trọng số INT8 vẫn đối xứng theo kênh.
VietLPR đưa Relu vào vùng lượng tử hoá để runtime có thể gộp các đoạn tính liên tiếp.
Không coi số QLinearConv nhiều hơn là đủ để kết luận nhanh hơn; vẫn đo toàn pipeline.
Ở ORT 1.20.1, `MatMulConstBOnly` không được QDQMatMul áp dụng như dynamic quantizer;
vòng refinement loại MatMul attention hai đầu vào động bằng tên node rõ ràng.
Ứng viên dynamic REC chỉ đóng gói năm MatMul trọng số cố định, giữ các Conv FP32.
Mọi file triển khai là graph ONNX có trọng số lượng tử hoá thật; graph tối ưu theo CPU
chỉ dùng chẩn đoán, không thay file xuất dùng cho thiết bị khác.

Hiệu chuẩn: cùng recipe 256 ảnh train DET và 128 REC nguồn của VietLPR,
có crop dự đoán từ train; YOLO dùng 256 ảnh train, letterbox đúng luồng gốc.
Percentile nhận từng ảnh/batch 1 để phù hợp ORT 1.20.1.

Chọn bằng **val**: OCR giảm tối đa một ảnh đúng trên 222 ảnh, CER tăng ≤0,2 điểm %;
YOLO mAP50-95 và Recall tại confidence 0,25 giảm ≤1 điểm %.
Trong nhóm đạt chất lượng, chọn ứng viên Speed val thấp nhất.
Nếu không có ứng viên đạt, ghi rõ không đạt và vẫn giữ FP32 cho luồng chính.
INT8 mixed phải ghi rõ phần giữ FP32. Không hạ ngưỡng chất lượng sau khi xem test.

Các số đo val là phép sàng lọc, không phải tốc độ chính thức đưa lên bảng test.

## Cố định lõi CPU

Máy hiện tại có bốn lõi P (mỗi lõi hai luồng SMT) và tám lõi E.
`ocr/cpu_protocol.py` đọc topology Windows và giới hạn affinity của từng tiến trình
vào một logical CPU trên mỗi lõi P: **0, 2, 4, 6**. Không đổi power plan hoặc tắt SMT hệ thống.
Mỗi model vẫn dùng bốn luồng ORT, nhưng cùng bốn lõi vật lý, tránh trộn P/E do scheduler.
Máy khác phải tạo lượt mới theo topology của máy đó.

Cùng một factory SessionOptions áp dụng `session.intra_op.allow_spinning=0`
và `session.inter_op.allow_spinning=0` cho mọi adapter trước khi tạo session.
Nhờ đó thread pool DET đang rảnh không quay chờ tranh CPU với REC trên bốn lõi đã cố định.
Giữ nguyên kernel/model, số luồng và pipeline nguồn; manifest lưu rõ hai option.
Đây là thay đổi runtime chung, không gán riêng cho INT8 để làm lệch so sánh với FP32.
Tham khảo [ONNX Runtime thread management](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).

`ocr/cpu_selection.py` đo lại baseline và các ứng viên đạt chất lượng trên val với affinity này,
sau đó mới chốt model. Không lượng tử hoá hoặc train lại ở bước chọn CPU.
Manifest và mỗi pass lưu topology, affinity; worker kiểm mask trước/sau khi đo.
Các pass chưa cố định lõi CPU của lượt `20261113_164740` được giữ làm lịch sử,
không ghép vào lượt V3 có affinity cố định.

## Một lần đo, nhiều bảng

`ocr/optimized_benchmark.py` điều phối **21 cấu hình**, năm vòng độc lập = **105 lượt**:

- 14 OCR như trước.
- Ba bản thêm của V2: INT8 cũ, INT4 mixed và INT8 tối ưu. FP32 đã có trong 14 OCR.
- Bốn YOLO: FP32, INT8 cũ, INT4 mixed và INT8 tối ưu.

Notebook 03/06 dùng duy nhất `models/VietLPR_v2/test_result.json`.
Notebook 04/07 dùng duy nhất `models/YOLO26n-OBB_FP32/test_result.json`.
Mỗi notebook chỉ đọc các bản ghi trong cùng manifest; không chạy một phép đo FP32 riêng.
Tên hiển thị dùng `VietLPR_v2_FP32`, `VietLPR_v2_INT8_mixed` và `VietLPR_v2_INT4_mixed`.
YOLO dùng `YOLO26n-OBB_FP32`, `YOLO26n-OBB_INT8_mixed` và `YOLO26n-OBB_INT4_mixed`.
Hai bảng lượng tử hoá chỉ giữ FP32/INT8 mới/INT4 mixed và xếp **FPS giảm dần**;
bảng 14 OCR giữ **ACC giảm → CER tăng → Speed tăng**. INT8 cũ không còn xuất hiện trong bảng hiện tại.
`ocr/benchmark_display.py` quản lý tên và thứ tự bảng, tách khỏi fingerprint đo gốc.
Không lấy tốc độ cũ điền vào bảng mới. Mỗi bảng có phạm vi khác nhau:
OCR nhận ảnh biển, detector nhận ảnh toàn cảnh. Chưa đo YOLO→OCR trong bảng này.

Giữ protocol đã thống nhất: ORT CPU 4 luồng, OpenCV 1 luồng, batch một ảnh;
124 test OCR / 258 test detection; mỗi lượt 20 warm-up trên val.
Tiến trình/session mới mỗi model/vòng, thứ tự model/ảnh đảo seed 42.
Nạp ảnh RAM trước khi đo, tắt profiling, không IO/log giữa các ảnh trong lượt.
Speed gồm toàn bộ pipeline tiền xử lý → ONNX → hậu xử lý; không gồm đọc ổ đĩa hoặc tải model.
SD là độ lệch chuẩn mẫu của năm trung bình vòng (`ddof=1`). FPS = 1000/Speed.
Speedup lấy model chậm nhất trong từng bảng làm 1×.
Kiểm nguồn AC/power scheme, phiên bản runtime, SHA model/dataset/code, GT và raw timings.
Lượt dở chỉ được tiếp tục khi fingerprint còn khớp; một pass dở sẽ đo lại nguyên pass.

Các bảng hiện tại bỏ INT8 cũ; bản INT8 đã chọn chỉ hiển thị hậu tố `INT8`, không ghi `optimized`.
Không tự thay ONNX trong pipeline triển khai bằng bản có ACC/mAP kém hoặc chậm hơn FP32.
Kết quả laptop không đại diện Pi 5; cần đo trực tiếp khi có thiết bị.
Test hiện tại đã được đánh giá nhiều lần; paper cần nêu giới hạn độc lập và ưu tiên test ngoài mới.

## Chạy và xem kết quả

```powershell
# Tạo/chốt cả hai model rồi đo đủ bảng:
.venv-ocr\Scripts\python.exe -u -B -m ocr.benchmark_display all

# Tiếp tục một lượt đã tạo, không tạo mới:
.venv-ocr\Scripts\python.exe -u -B -m ocr.benchmark_display run --run-dir runs/int8_optimization/benchmark/<run>

# Đọc bảng sau khi hoàn tất:
.venv-ocr\Scripts\python.exe -B -m ocr.benchmark_display table --group quantization
.venv-ocr\Scripts\python.exe -B -m ocr.benchmark_display table --group detection
```

Các notebook 03/06/07 gọi cùng lệnh; mặc định `RUN_BENCHMARK=False` chỉ đọc.
Không mở đồng thời hai lượt benchmark hoặc chạy train trong lúc đo.

```text
runs/int8_optimization/
  vietlpr/{protocol,calibration,trials,selected}.json
  vietlpr/candidates/<cấu hình>/{det,rec}.onnx
  vietlpr/refinement/selected.json
  vietlpr/refinement/selected/{det,rec}.onnx   Chọn sơ bộ trước khi cố định CPU
  yolo/{protocol,calibration,trials,selected}.json
  yolo/selected/model.onnx              Quyết định vòng đầu
  yolo/refinement/selected/model.onnx   Chọn sơ bộ sau kiểm ngưỡng chất lượng
  fixed_cpu/vietlpr/selected/{det,rec}.onnx    Model dùng benchmark V3
  fixed_cpu/yolo/selected/model.onnx          Model dùng benchmark V3
  active_benchmark.json                  Lượt đang chạy, có thể chưa xong
  latest_benchmark.json                  Chỉ chuyển sau khi đủ cả 105 lượt
  benchmark/<run>/manifest.json          Snapshot chung
  benchmark/<run>/state.json             Tiến độ
  benchmark/<run>/passes/<model>/        Prediction/GT/time từng ảnh từng vòng
  benchmark/<run>/models/<model>/test_result.json
  benchmark/<run>/comparison_{ocr,quantization,detection}.{csv,md}
  benchmark/<run>/verification.json      SHA baseline cũ giữ nguyên
```

Model, ảnh, log và kết quả trong `runs/` luôn ngoài Git.

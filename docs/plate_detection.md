# Detector biển số và test toàn luồng

Luồng chính: ảnh toàn cảnh → YOLO26-OBB → crop toàn biển → VietLPR_v2.
YOLO tìm toàn biển. DET của VietLPR/Paddle tìm từng dòng chữ trong crop đó.
Năm OCR public vẫn chỉ làm đối chứng. Không đổi code, trọng số hoặc ngưỡng OCR hiện có.

## Detector nano

Dự án chỉ dùng **YOLO26n-OBB**, khởi tạo từ `yolo26n-obb.pt` pretrained DOTA.
Code dùng Ultralytics 8.4.14, Torch 2.7.1 CUDA 12.6 trong `.venv-ocr`.
Cài máy mới bằng `requirements/detection.txt`.

Mở `notebooks/04_train_plate_detector.ipynb`. Train GPU, ảnh 640, batch 4,
tối đa 150 epoch, early stop 30, seed 42. Best theo mAP val, không chọn theo test.
Không lật ảnh biển. Chạy lại tiếp tục `last.pt`; lượt hoàn tất giữ `best.pt`.

```powershell
.venv-ocr\Scripts\python.exe -u -m plate_detection.suite
```

Lệnh chỉ chạy nano: train/resume → val → xuất ONNX → test detection CPU.
Các bản s/m/l/x đã bỏ khỏi cấu hình và hàng đợi. Checkpoint nano hiện có tiếp tục được dùng,
không đổi dữ liệu hoặc tham số train. File chạy thử cũ vẫn giữ ngoài Git.

## Nguồn và phép chuyển nhãn

Nguồn: [Thaos Workspace, Vietnamese License Plate v1](https://universe.roboflow.com/thaos-workspace-ciocz/vietnamese-license-plate-ypfnp/dataset/1), CC BY 4.0.
Giữ attribution trong README nguồn và `source_audit.json` của bản dựng.
Export người dùng tải là **YOLO26**, nhưng nhãn thực tế là polygon, không phải AABB 5 cột:

| Dạng nhãn | Số hộp |
|---|---:|
| 4 góc + lặp điểm đầu, 11 cột | 5.194 |
| 5 góc + lặp điểm đầu, 13 cột | 65 |
| 6 góc + lặp điểm đầu, 15 cột | 24 |
| Tổng | 5.283 |

Bỏ điểm đóng polygon, kiểm hình học rồi lấy `minAreaRect` trong hệ pixel.
Hai lớp `license-plate-long/short` gộp thành `license-plate`.
Không tạo góc xoay từ AABB. Nhãn chữ nhật thẳng 5 cột bị từ chối.
Định dạng OBB cuối: `0 x1 y1 x2 y2 x3 y3 x4 y4`, tọa độ chuẩn hoá.

| Split | Nguồn | Dùng train/val/test |
|---|---:|---:|
| Train | 3.616 | 3.591 |
| Val | 517 | 515 |
| Test | 258 | 258 |

Ba ảnh train bị cách ly nguyên ảnh trong bản dựng: một polygon tự cắt/suy biến,
hai ảnh có một biển dưới hai pixel. Giữ nguyên các ảnh và nhãn nguồn.
89 ảnh cần đệm mép màu 114 để chứa đủ chữ nhật xoay. Ảnh đệm lưu PNG,
pixel gốc không bị nén lại; hộp được dịch và chuẩn hoá theo ảnh mới.
Các ảnh còn lại được sao chép nguyên file. Không sửa `vn-plate-dataset_v2/`.

Rà pHash và đối chiếu ảnh/crop phát hiện mười nhóm cùng biển qua split,
dù tên file khác và pixel không hoàn toàn giống. Quyết định lưu tại
`configs/detection/split_review.json`, kèm SHA ảnh nguồn để tránh áp dụng nhầm.
Giữ nhóm ở test nếu có test, nếu không giữ ở val; cách ly cả họ ảnh train liên quan.
24 ảnh đã tách khỏi bản dựng (22 train, hai val). Nguồn vẫn giữ nguyên.
Bản cuối **4.364 ảnh / 5.252 hộp**, kiểm lại pixel và nhãn bằng `plate_detection.data check`.
Train bị chặn nếu thiếu `normalization_check.json` hoặc dữ liệu khác chữ ký đã kiểm.

Nguồn đã stretch 640×640 và tạo hai biến thể train qua xoay ±15°/đổi sáng ±15%.
4.391 ảnh export không phải 4.391 cảnh gốc độc lập.
Không có trùng pixel hoặc cùng họ tên Roboflow qua split trong kiểm tra hiện tại.
Chưa chứng nhận độc lập danh tính biển/xe hoặc ảnh gần trùng khác tên;
cần kiểm nhóm này, cả nguồn train OCR, trước khi công bố benchmark cuối.

## So sánh riêng detection trước OCR

Hiện chạy bước này trước khi ghép OCR. Notebook 04 có báo cáo riêng YOLO26n-OBB.
Có thể chạy nano bằng:

```powershell
.venv-ocr\Scripts\python.exe -u -m plate_detection.suite
```

Nano: train/val → best → ONNX → đối chiếu val → test detection CPU.
Suite không gọi OCR. Trạng thái nằm trong `runs/plate_detection/suite/state.json`,
log từng lệnh trong `runs/notebook_logs/`. Chạy lại tiếp tục checkpoint và số đo còn thiếu.
Lượt kiểm tra một epoch trên vài ảnh chỉ là smoke, không được đưa vào bảng chính thức.

Bảng: `Rank | Model | mAP50 | mAP50-95 | Precision | Recall | Speed | FPS | Speedup`.
mAP chấm trực tiếp ONNX bằng evaluator OBB của Ultralytics (ProbIoU theo nguồn).
Precision/Recall trong bảng tính tại confidence **0,25**, ghép polygon IoU thật ≥ **0,5**,
không lấy điểm P/R tối ưu F1 trên test để làm ngưỡng triển khai.
Speed gồm letterbox + ONNX + giải mã, không có OCR; CPU 4 luồng,
OpenCV một luồng, FP32, một ảnh/lần, warm-up 20 trên val, năm lượt trên mọi ảnh test.
Rank theo mAP50-95 giảm → Recall giảm → Speed tăng. Chậm nhất làm Speedup 1×.
Kết quả cuối: `runs/plate_detection/comparison.csv` và `comparison.md`.
Chỉ hiển thị kết quả nano sau khi train và đo xong.

OBB là **chữ nhật xoay**, không phải bốn góc phối cảnh tự do.
135 hộp có polygon chiếm dưới 80% diện tích OBB: biển xiên mạnh sẽ dư nền ở góc.
Phép chuyển vẫn bao polygon, không tự siết vào chữ để loại nền.
Crop chỉ nắn khung xoay, không thể phục hồi hoàn toàn biến dạng hình thang từ OBB.
Chiều đọc 90°/180° cũng không được mã hoá trong OBB; crop hiện dùng góc trên trái
theo ảnh và không ép cạnh dài thành chiều ngang, tránh xoay ngang biển hai tầng.
Ảnh quay gần đứng hoặc lộn ngược cần bộ hướng đọc riêng nếu sau này triển khai.

## Test YOLO + OCR

Mở `notebooks/05_evaluate_full_pipeline.ipynb`.
Nano ghép với 14 OCR (nhóm chính 9, đối chứng 5), tổng cộng 14 tổ hợp.
Không dùng hộp GT để crop, không dùng text GT để chọn hộp.
Giữ tất cả biển dự đoán trên ảnh, không chỉ lấy hộp có điểm cao nhất.
Padding crop 3% mỗi cạnh, cấu hình chốt trước test. OCR giữ nguyên luồng nguồn.

Roboflow chỉ cung cấp lớp/polygon, chưa có chuỗi biển. Notebook tạo
`data/e2e/plates.jsonl` từ val/test để điền và duyệt thủ công:

```json
{"image":"val/images/example.jpg","split":"val","reviewed":true,"identity_group":"59A112345","plates":[{"quad":[[10,20],[110,20],[110,90],[10,90]],"text":"59A1-123.45"}]}
```

`quad` dùng pixel trên ảnh bản dựng; `image` tương đối với `runs/plate_detection/dataset`.
Mỗi ảnh cần đầy đủ biển, text chuẩn và nhóm cùng biển. Không dùng OCR dự đoán làm GT.
Ảnh âm dùng `plates: []`. Chưa điền/duyệt thì bước chấm ACC/CER bị chặn.
Kiểm nhóm cùng biển với split train và dataset OCR; không chọn lại ngưỡng theo test.

Hộp dự đoán và GT ghép một-một bằng rotated polygon IoU ≥ 0,5,
ưu tiên số cặp đạt ngưỡng, sau đó tổng IoU; không dùng text trong phép ghép.
ACC = số biển định vị đủ IoU và đọc đúng toàn chuỗi / tổng biển GT.
CER = tổng khoảng cách chỉnh sửa / tổng ký tự GT. Biển bỏ sót có chuỗi rỗng;
ký tự từ hộp dư là lỗi chèn. TP/FP/FN được lưu cùng kết quả để thấy hộp dư không có text.
CER có thể vượt 100%. Không ghép bảng này với bảng OCR từ crop có sẵn.

Speed tính từ ảnh toàn cảnh tới toàn bộ chuỗi, gồm YOLO + crop + OCR.
CPU ONNX FP32, 4 luồng, OpenCV 1 luồng, batch một ảnh toàn cảnh,
warm-up 20 trên val, đo năm lượt trên toàn bộ ảnh test.
Loại thời gian đọc ổ đĩa, tải model và ghi tiến độ. Không cache kết quả detector giữa các lượt/OCR.
Lưu tiến độ từng ảnh; khởi chạy lại tiếp tục, không đo lại các lượt đã đủ.
Rank theo ACC giảm → CER tăng → Speed tăng, model chậm nhất trong từng bảng Speedup 1×.

## Kết quả nằm ở đâu?

```text
runs/plate_detection/
  label_audit.json                 Kết quả rà nguồn
  dataset/                        OBB một lớp dùng chung, kèm source_audit.json
  yolo26n_obb/                     Chỉ dùng nano
    weights/{best,last}.pt
    onnx/{model.onnx,verified.json}
    training_protocol.json
    detection_val.json
    detection_result.json            ONNX mAP, P/R và Speed, riêng detector
    onnx_test_ap.json                Kết quả evaluator OBB gốc
runs/end_to_end/
  yolo26n_obb/                     Chỉ dùng nano
    <tên OCR>/{frozen,test_result,test_predictions,progress}.json
    comparison_ranked.csv          Bảng OCR dùng chung detector nano
```

Nguồn ảnh, bản dựng, checkpoint, ONNX và số đo đều được `.gitignore` bỏ qua.
Notebook 01–03 và các baseline OCR cũ vẫn dùng đúng dataset/đường dẫn trước đây.

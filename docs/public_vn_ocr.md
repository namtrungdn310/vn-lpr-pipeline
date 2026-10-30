# Pretrained chuyên biển số Việt Nam

Nhóm này chỉ phục vụ test đối chứng. Luồng chính vẫn là V1/V2 và 7 bản Paddle.
Các file `ocr/public_vn*.py` là adapter test; code nguồn model tải theo URL/SHA và trọng số được giữ trong `runs/`, ngoài Git.

Rà nguồn công khai ngày 30/10/2026 trên GitHub, Hugging Face, các liên kết dataset và model do tác giả cung cấp.
Chỉ thêm model có trọng số tải được, bằng chứng huấn luyện cho biển Việt Nam, và xác định được cách suy luận.
Không dùng tên repository hoặc khả năng đọc tiếng Việt làm bằng chứng model đã train cho biển số.

## Các model được thêm

| Tên trong bảng | Trọng số và nguồn | Bằng chứng chuyên biệt |
|---|---|---|
| `CCT_S_VN` | [ddkhoa2605/LicensePlateDetection](https://github.com/ddkhoa2605/LicensePlateDetection) — `cct_s_v1_vn.onnx` | Tác giả mô tả fine-tune trên 3.763 crop biển Việt Nam; SHA khác bản CCT global gốc |
| `LPRNet_VN` | [lephamcong/PBL4_Deep-Learning](https://github.com/lephamcong/PBL4_Deep-Learning) — `LPRNet_Pytorch_VietNam.pth` | Nhánh VietNam riêng, mô tả 12.320 ảnh train và 1.000 ảnh đánh giá; không dùng checkpoint China |
| `YOLOv5n_VN_Chars` | [trungdinh22/License-Plate-Recognition](https://github.com/trungdinh22/License-Plate-Recognition) — `LP_ocr_nano_62.pt` | Trọng số nhận dạng ký tự trên biển Việt Nam một/hai dòng, đi kèm hướng dẫn và dataset |
| `YOLOv5m_VN_Chars` | Cùng nguồn — `LP_ocr.pt` | Bản medium với trọng số khác nano; không đếm các bản mirror thành model mới |
| `YOLOv8n_VN_Chars` | [phatnomenal/Vietnamese-License-Plate-Detecttion](https://github.com/phatnomenal/Vietnamese-License-Plate-Detecttion) — `Letters_detection.pt` | Model 36 lớp ký tự, checkpoint ghi phiên bản 8.3.229 và lượt train `Letters_detection` |

`configs/ocr/public_vn.json` chốt commit, URL, SHA256, charset, tham số và liên kết bằng chứng cho từng model.
Bằng chứng huấn luyện dựa trên tài liệu và artifact của tác giả; dữ liệu train ngoài dự án không có đầy đủ để chứng nhận không trùng tập test.
Model yếu trên tập này vẫn được giữ trong bảng; không chọn model theo ACC test và không thay ký tự bằng quy tắc đoán biển số.

## Các nguồn không đưa vào bảng

| Nguồn đã rà | Lý do |
|---|---|
| [Fast Plate OCR model zoo](https://github.com/ankandrew/fast-plate-ocr/blob/master/docs/inference/model_zoo.md) | Các bản global/Argentina/Europe không phải checkpoint fine-tune riêng cho Việt Nam; chỉ dùng bản CCT-VN ở trên |
| [LiteALPR](https://huggingface.co/anhone3/LiteALPR) | Card mô tả train chung Brazil, China, Vietnam; không công bố checkpoint chuyên Việt Nam riêng |
| [Nguyenhhh/crnn-vietnamese-ocr](https://huggingface.co/Nguyenhhh/crnn-vietnamese-ocr/tree/main) | Có `.pth`, nhưng thiếu card, kiến trúc/charset và bằng chứng huấn luyện riêng cho biển số |
| [Qwen2-VL plate LoRA](https://huggingface.co/toanlion24/vn-plate-qwen2-vl-2b-lora) và [bản cùng tác giả](https://huggingface.co/toanlion24/vn-plate-qwen2-vl-2b) | Card mặc định thiếu mô tả dữ liệu và quy trình suy luận; chưa đủ xác minh chỉ từ tên model |
| [KienNguyen2k1 Space](https://huggingface.co/spaces/KienNguyen2k1/VietnameseLicensePlateRecognition) | Artifact mã hoá/OpenVINO và card thiếu mô tả huấn luyện; chưa có đường ONNX FP32 đã xác minh cho bảng này |
| [harivodich](https://github.com/harivodich/vietnamese-license-plate-recognition) | Tên `wirqhuy/vietnamese-license-plate-ocr` trong repo là nguồn dataset Kaggle; không phải checkpoint OCR public đã xác minh |
| [sangtrx](https://github.com/sangtrx/license-plate-recognition), [tungedng2710](https://github.com/tungedng2710/AI-Traffic-Analysis), [itsZiang](https://github.com/itsZiang/vietnamese-license-plate-recognition) | Có detector Việt Nam nhưng OCR dùng Paddle tổng quát; không có bằng chứng checkpoint OCR fine-tune riêng |
| [haminhthong](https://github.com/haminhthong/vietnamese-license-plate-recognition), [tam520/LPRS](https://github.com/tam520/LPRS), [nguyencongtuyenlp](https://github.com/nguyencongtuyenlp/license-plate-recognition) | Dùng EasyOCR tổng quát hoặc không công khai trọng số OCR chuyên biệt |
| [Nghmin](https://github.com/Nghmin/Vietnamese-License-Plate-Recognition-LPR-System), [phthanhtung](https://github.com/phthanhtung/alpr-vietnam-system) | Pipeline Việt Nam có hậu xử lý riêng, nhưng không chứng minh đã fine-tune trọng số OCR |
| [n-tan-adonis](https://github.com/n-tan-adonis/Vietnamese-License-Plate-Regconition) | Dẫn tới folder Drive; chưa xác minh được artifact nhận dạng ký tự công khai qua liên kết đó |
| [MagicXuanTung](https://github.com/MagicXuanTung/Yolov8-Detect-Vietnamese-license-plates-and-characters), [NhapMonAI](https://github.com/lhlizdabezt/NhapMonAI) | Có demo/weights, nhưng tài liệu được rà chưa đủ xác định một checkpoint OCR mới đã train riêng cho Việt Nam |
| [trcamnguyen/vnplaterec](https://huggingface.co/trcamnguyen/vnplaterec) và các fork YOLOv5 | Không tính thêm mirror thiếu model card hay chỉ đổi nơi lưu cùng họ trọng số |
| [NVIDIA LPRNet](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/lprnet/deployable_v1.0), OCR tiếng Việt giấy tờ/tài liệu | Không phải checkpoint chuyên biển Việt Nam; ngôn ngữ tiếng Việt hoặc charset Latin chưa đủ điều kiện |

Đây là các nguồn đã kiểm tra, không phải cam kết đã tìm được mọi model tồn tại trên Internet.

## Cách chạy và so sánh

- Đầu vào chung là 124 ảnh biển trong `test/det/images`, chuỗi GT ghép các dòng theo cùng quy tắc của chín model cũ.
- Không dùng polygon GT, crop REC test hoặc nhãn để tạo đầu vào cho model mới; không chạy detector xe/biển ngoài phạm vi ảnh biển đã có.
- CCT và LPRNet đọc toàn biển; YOLO phát hiện ký tự rồi sắp theo dòng. Giữ tiền/hậu xử lý nguồn, không thêm model DET dòng để ép cùng kiến trúc.
- CPU ONNX Runtime FP32, 4 luồng, OpenCV 1 luồng, batch 1 ảnh biển, warm-up 20 từ val, đo 5 lượt với cùng seed.
- Speed gồm toàn bộ tiền xử lý, các forward pass và giải mã của pipeline tương ứng; loại I/O và thời gian tải model.
- ACC/CER tính 124 ảnh một lần; Speed lấy 620 số đo. Ảnh bị điều kiện kích thước nguồn từ chối vẫn tính là chuỗi rỗng, không bỏ khỏi tập test.
- Rank: ACC giảm, CER tăng, Speed tăng. Speedup lấy model chậm nhất trong toàn bảng làm mốc 1×.
- Chín model cũ dùng lại kết quả hợp lệ đã lưu; không train và không đo lại chúng trong lần bổ sung này.

Khác biệt đáng chú ý khi đối chiếu nguồn:

1. CCT ONNX thực tế nhận **uint8 NHWC**, bên trong tính FP32; không dùng nhầm mô tả float32 NCHW trong README.
2. CCT giải mã **9 slot**, giữ số lặp như `111`; không áp dụng CTC collapse dù có đoạn README gọi là CTC.
3. YOLOv5 checkpoint có **30 lớp theo thứ tự riêng**, khác YAML 36 lớp trong repo. Dùng `names` của checkpoint và kiểm khớp khi nạp.
4. LPRNet dùng BGR và blank ở cuối charset. Khi export, ghi rõ trục batch ngầm của `MaxPool3d` 4D; đã đối chiếu kết quả Torch nguyên bản và bản export.
5. Checkpoint Torch chỉ được nạp bằng `weights_only=True`, với danh sách lớp đã rà. Không chạy script train, app hoặc cài đặt của repository bên ngoài.

Code:

| File | Nhiệm vụ |
|---|---|
| `ocr/public_vn.py` | Tải đúng SHA, chuẩn bị, chốt val, test và lưu số đo để resume |
| `ocr/public_vn_export.py` | Nạp định nghĩa nguồn đã chọn, xuất ONNX và đối chiếu Torch/ONNX trên val |
| `ocr/public_vn_runtime.py` | Tiền xử lý và giải mã theo từng họ model |
| `ocr/comparison.py` | Kiểm kết quả/số đo gốc của cả 14 hệ thống và ghép bảng |

```powershell
.venv-ocr\Scripts\python.exe -m ocr.public_vn prepare
.venv-ocr\Scripts\python.exe -m ocr.public_vn freeze
.venv-ocr\Scripts\python.exe -m ocr.public_vn test
.venv-ocr\Scripts\python.exe -m ocr.comparison
```

Trong notebook 03, `RUN_PUBLIC_VN=True` chỉ chạy nhóm mới; `RUN_EVALUATION=True` chạy cả quy trình.
Mặc định hai cờ đều `False`, chỉ đọc kết quả. Không thay đổi dataset v2 hoặc các trọng số đã train.
Model và số đo mới nằm ở `runs/public_vn_ocr/<tên>/`; bảng chung ở `runs/comparison_ranked.csv`.
Các file trọng số và dữ liệu vẫn bị Git bỏ qua.

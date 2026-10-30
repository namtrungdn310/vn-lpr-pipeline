# VietLPR_v2 — nguồn notebook

Nguồn: `VietLPR_v2_colab.ipynb` do người dùng cung cấp.
SHA256: `aafe7b2f63c0c6346ac00f807c96ee249de186e37021a0d0c81daf915f2adbed`.

- `models.py`: kiến trúc từ cell 14; chỉ rút gọn docstring.
- `source.py`: các định nghĩa gốc từ cell 8, 12, 17, 19, 20, 22, 23. Giữ nguyên AST các hàm đã tách.
- `inference.py`: nguyên nội dung `lpr_infer.py` trong cell 31, bỏ magic `%%writefile`.
- `source_manifest.json`: cell và checksum đối chiếu.

Không thực thi cell Colab, Drive, download, video, Paddle hoặc YOLO khi import.
`ocr/upstream.py` nối cấu hình/thiết bị; `train_torch.py` nối split cố định, dataset thật và checkpoint.
V2 dùng stretch, EMA, jitter và REC từ crop DET theo nguồn. Vòng train gốc được gọi trực tiếp.
Pipeline ONNX gốc có thể dùng độc lập từ các file `lpr_infer.py`, `lpr_config.json`, `det.onnx`, `rec.onnx` do export tạo.

Bản notebook nguyên byte được giữ tại `runs/vietlpr_v2/source/VietLPR_v2_colab.ipynb` để đối chiếu; không cần file Downloads khi chạy lại dự án.

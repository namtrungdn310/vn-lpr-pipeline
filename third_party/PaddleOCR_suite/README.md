# Tiền/hậu xử lý PaddleOCR

Tách từ [PaddleX](https://github.com/PaddlePaddle/PaddleX/tree/4c968d30b9d0d4a693d9eaaef0881c20f8d7d0e0), giữ công thức gốc.
`source_manifest.json` ghi URL, commit và SHA AST từng định nghĩa.

- `processors.py`: resize, normalize, DB, sắp dòng, crop phối cảnh và CTC.
- Bỏ decorator đo thời gian/kiểm dependency, docstring và lớp cha BaseOperator.
- Crop chỉ giữ hai hàm dùng cho box bốn góc; không thêm heuristic sửa biển số.
- Trọng số và dictionary tải từ nguồn PaddlePaddle chính thức, nằm trong `runs/paddle_suite/`.

Adapter ở `ocr/paddle_suite_runtime.py`. Không cần cài PaddleX hoặc nâng môi trường Paddle cũ.

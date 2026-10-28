# VietLPR_v2

`models.py` chứa FastDBNet, FastCRNN và hàm gộp nhánh RepConv.
Code được tách từ notebook nguồn `VietLPR_v2.ipynb`. Notebook gốc đã được xoá khỏi dự án.

SHA256 notebook nguồn: `aafe7b2f63c0c6346ac00f807c96ee249de186e37021a0d0c81daf915f2adbed`.

Train/val: `experiments/train_torch.py`. Xuất ONNX: `experiments/export.py`.
Cấu hình: `experiments/configs/vietlpr_v2.json`.
Hậu xử lý NumPy: `experiments/ops_v2.py`.

`python -m experiments.bootstrap --source VietLPR_v2` dùng trực tiếp mã nguồn đã tách.
Chỉ tách lại kiến trúc nếu khôi phục notebook gốc đúng SHA256 ở thư mục gốc.
Ba file mã nguồn/tài liệu trong thư mục này được lưu trong Git để dùng được khi không có notebook gốc.

# Benchmark FP32 sau adaptation

Quy trình/lệnh hiện tại nằm trong **[pc_training.md](pc_training.md#5-đo-benchmark-mới-trên-i5-14600kf)**.
Notebook 03 dùng `ocr.adaptation.benchmark`, bảy model × năm vòng = 35 pass trên i5-14600KF.
Giữ phạm vi đo RAM, warm-up 20, bốn lõi P, bốn intra-op/một inter-op, tắt spin wait và SD giữa năm trung bình vòng.

Model được chọn trên val trước test. Không ghép kết quả khác máy, code, dataset, model hoặc điều kiện nguồn.
Bảng mới nằm ở `runs/ocr_adaptation/benchmark/<run>/comparison.{csv,md}`.

`runs/int8_optimization/` là kết quả lượng tử hoá đã chốt trên ThinkPad cho notebook 06/07;
không phải bảng OCR mới và không được dùng thay kết quả adaptation chưa train/đo.

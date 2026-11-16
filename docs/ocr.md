# OCR hiện tại

Luồng đang dùng: **[train sáu OCR và benchmark FP32 trên PC](pc_training.md)**.
Cấu hình chính: `configs/ocr/adaptation.json`. Notebook 02 train; notebook 03 xuất bảng bảy model.

VietLPR_v1 là tên công bố mới của model đã chốt trước đây gọi VietLPR_v2; giữ nguyên DET/REC và hậu xử lý nguồn.
Model VietLPR_v1 cũ không tham gia so sánh mới. YOLO26n-OBB giữ nguyên.

`ocr/adaptation/` nối dữ liệu, checkpoint, export và đo CPU. CCT dùng FastPlateOCR đã pin;
Paddle gọi `tools/train.py` nguồn đã pin; LPRNet nạp lớp nguồn có SHA và dùng CTC.
Mã nguồn tải về và trọng số giữ ngoài Git.

Các helper nguồn, fingerprint và lượng tử cũ còn được notebook 04/06/07 gọi để đọc kết quả đã chốt.
Chúng không quyết định danh sách model hoặc số đo của bảng mới. Không chạy lại benchmark pretrained cũ cho paper mới.

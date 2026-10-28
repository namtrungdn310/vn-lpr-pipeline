# Dataset biển số v2

Mở `main.ipynb`, chọn kernel `.venv`, **Restart Kernel** rồi chạy từ trên xuống.
Notebook gồm đầy đủ các bước chuẩn hoá, đề xuất/duyệt box, dựng lại v2 và kiểm tra đầu ra.

Repository chỉ lưu code và metadata. Ảnh dataset, nguồn REC và ZIP được lưu riêng, không đưa lên Git.
Sau khi clone, đặt `vn-plate-dataset_v2/` và `sources/` vào thư mục gốc theo cây bên dưới.
Cài `requirements-dev.txt`, chọn kernel Python của môi trường đã cài và chạy `main.ipynb`.
Output notebook được để trống khi commit; chạy lại các cell để xem ảnh và kết quả.

```text
main.ipynb                 Quy trình chuẩn hoá, đề xuất/duyệt box, xuất v2
src/                      Logic dùng chung
metadata/                 Recipe và checksum đã chốt
sources/rec/              831 ảnh nguồn còn cần để tái tạo REC
tests/                    Kiểm thử logic chính
vn-plate-dataset_v2/       Dataset dùng để train
vn-plate-dataset_v2.zip    Bản ZIP cùng nội dung
```

Hai thuật toán box `line_envelopes.py` và `text_bounds.py` đã được khôi phục từ lịch sử trước khi dọn code.
Hai module được định dạng và chú thích bằng tiếng Việt, giữ nguyên thuật toán. Xuất xứ lưu trong `metadata/code_origin.json`.
Không xoá `sources/` nếu còn cần tái tạo REC.

## Luồng chuẩn hoá

1. Đọc dataset nguồn có ảnh và nhãn sơ bộ, giữ tên/split; bỏ đúng các ảnh đã duyệt cách ly.
2. Rà nhãn, hình học, độ mờ/chói và ảnh trùng. Ngưỡng chất lượng chỉ tính trên train.
3. Đề xuất box từng dòng bằng `propose_line()`: tìm nét chữ, fit đỉnh/chân và cạnh trái/phải trên ảnh gốc.
4. Có thể chạy thêm `glyph_quad()`: dùng bản đồ xác suất DB và nét tối để siết bốn cạnh.
5. Xem trước/sau rồi duyệt giữ, nhận đề xuất hoặc nhập bốn góc thủ công; nhãn đọc sai cũng cần đối chiếu ảnh nguồn.
6. Từ box và nguồn đã chốt, crop/nắn REC. Giữ hai hệ tọa độ riêng cho DET và REC gốc.
7. Dựng sáu file nhãn, copy ảnh DET, tạo lại toàn bộ REC và giữ review list để xuất đầy đủ v2.
8. Đối chiếu SHA, kiểm ảnh/nhãn/box/liên kết/trùng split, xem kết quả và kiểm ZIP.

`main.ipynb` chạy đề xuất trên đủ 1975 dòng. Đề xuất không tự thay box đã duyệt, kể cả khi thuật toán báo `supported`.
Các mẫu mờ, nét mảnh, ký tự đầu/cuối và chỗ giáp hai tầng vẫn cần xem trực quan.
Notebook phát lại quyết định duyệt của v2 để dựng bản kiểm tra trong thư mục tạm; không tạo v3, không tự ghi đè dataset đang train.

**Giới hạn phục hồi:** V1, nhãn sơ bộ cũ và 15 ảnh đã cách ly không còn đầy đủ sau lần xoá trước.
Không thể tuyên bố đã tái hiện mọi byte của đầu vào lịch sử đó. Mặc định `INPUT_ROOT` là v2 hiện tại để chạy lại thuật toán;
nếu còn bộ nguồn ở ngoài workspace, đổi biến này. Các ảnh nguồn giữ lại và quyết định đã duyệt đủ để dựng lại chính xác v2.

## Dữ liệu train

| Tập | Ảnh DET | Crop REC |
|---|---:|---:|
| train | 869 | 1412 |
| val | 222 | 371 |
| test | 125 | 192 |
| Tổng | **1216** | **1975** |

DET có 1975 polygon, mỗi polygon tương ứng một dòng chữ. Tên ảnh, split, nhãn và pixel của v2 được giữ nguyên khi dọn code.

```text
train/
  det/det_labels.txt
  det/images/*.jpg
  rec/rec_labels.txt
  rec/images/*.jpg
```

`val` và `test` có cấu trúc tương tự. Nhãn UTF-8, tách tab đầu tiên:

- DET: `images/ảnh.jpg<TAB>[{"transcription": "...", "points": [[x,y], ...]}]`
- REC: `images/ảnh.jpg<TAB>CHUỖI_NHÃN`

Đường dẫn ảnh tính từ thư mục chứa file nhãn. Giữ chuỗi để không mất số 0 đầu.
Bộ ký tự: `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ`. Tọa độ DET dùng biên liên tục `[0,W] × [0,H]`.

REC đã crop và nắn; khi train chỉ resize/pad giữ tỉ lệ rồi normalize theo model.
Helper ở `src/rec_preprocessing.py`. Khi inference, `prepare_rec_crop` nhận quad một dòng theo TL, TR, BR, BL.
Không đưa toàn biển hai dòng vào recognizer một dòng.

## Chạy bằng lệnh

```powershell
.venv\Scripts\python.exe -m src.pipeline validate
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m src.pipeline package
```

`validate` kiểm ảnh, nhãn, quad, liên kết DET–REC, giao nhau giữa các tầng, trùng pixel/chuỗi biển giữa các split,
và tái tạo cả 1975 REC khớp từng byte JPEG. Kết quả hiển thị trực tiếp trong notebook hoặc terminal;
`technical_ready: true` nghĩa là đã qua kiểm tra. Không ghi file báo cáo.
`package` kiểm lại rồi thay ZIP; không ghi lại ảnh hoặc nhãn trong v2.

`build_v2(workspace, destination, input_root=..., reviewed_boxes=...)` dựng cả cây dataset từ nguồn và quyết định duyệt.
Không đọc/copy JPEG REC thành phẩm làm đầu vào. Đích phải chưa tồn tại và không được chồng lên nguồn.
Cell xuất trong notebook chạy hàm này rồi đối chiếu mọi file với bản chốt.

Khi cài môi trường mới: cài `requirements-dev.txt` để dùng notebook và các công cụ kiểm tra.
Chỉ dùng thư viện xử lý dữ liệu thì cài `requirements.txt`.
Nhánh DB dùng thêm `requirements-bounds.txt`; nhánh native không cần model này.
Máy train chỉ cần cây v2 và loader phù hợp; không cần copy `sources/`, `metadata/` hoặc báo cáo.

## Logic chính

| File | Vai trò |
|---|---|
| `src/dataset.py` | Đọc nhãn, giữ chuỗi và đối chiếu đường dẫn |
| `src/normalization.py` | Rà nguồn/nhãn/trùng, nối hai nhánh đề xuất box, duyệt từng khung |
| `src/line_envelopes.py` | Tạo/siết box trên ảnh gốc theo đỉnh, chân và cạnh nét chữ |
| `src/text_bounds.py` | Đề xuất bốn cạnh theo nét tối được DB hỗ trợ |
| `src/geometry.py` | Kiểm quad, nắn phối cảnh, mã hoá JPEG, diện tích giao |
| `src/pipeline.py` | Dựng đầy đủ v2, tái tạo REC, kiểm tra và đóng gói ZIP |
| `src/rec_preprocessing.py` | Loader REC và resize/pad |
| `src/preview.py` | So sánh đề xuất trước/sau, vẽ DET một/hai tầng và xem REC |
| `src/io.py` | Đường dẫn, checksum, ghi file |

`metadata/samples.jsonl` lưu một recipe cho mỗi cặp DET–REC.
Quad DET nằm trên ảnh DET; quad crop có thể nằm trên REC gốc, tuỳ `source_area`.
Không dùng lẫn hai hệ tọa độ. `preserve_text_order` giữ trục chữ cho mẫu phối cảnh đặc biệt.

Checksum trong `metadata/baseline.json` chốt bản đã duyệt. Nếu sửa ảnh, nhãn hay recipe, kiểm tra sẽ báo lỗi;
không tự tạo lại baseline để bỏ qua lỗi. Mẫu mới hoặc sửa annotation cần được duyệt và cập nhật metadata riêng.
Quyết định mới khác v2 đã chốt sẽ bị `build_v2()` chặn; hàm này xuất lại bản chốt, không tự phê duyệt dữ liệu mới.

## Giới hạn

Ảnh mờ hoặc che khuất từ nguồn vẫn được giữ. Có 699 review cũ chưa map được và 20 liên kết DET–REC thiếu bằng chứng pixel lịch sử.
Thông tin chỉnh nhãn/cách ly được giữ gọn ở `metadata/provenance.json`.
Chưa train model hoặc đo CER/accuracy. Kiểm tra kỹ thuật không chứng nhận mọi họ ảnh nguồn độc lập giữa các split.

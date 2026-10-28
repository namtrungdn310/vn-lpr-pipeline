# So sánh PaddleOCR, VietLPR_v1 và VietLPR_v2

Mở `train_compare.ipynb` ở thư mục gốc để đọc và chạy theo từng bước.
`main.ipynb` vẫn là notebook chuẩn hoá dataset; hai luồng không ghi đè nhau.
Hiện chỉ chuẩn bị và kiểm code, chưa train đủ epoch hoặc có bảng hiệu năng cuối.

## Ba hệ thống

| Tên dùng trong bảng | Kiến trúc | Nguồn |
|---|---|---|
| PaddleOCR | PP-OCRv4 mobile DET (PPLCNetV3 + DB) và English PP-OCRv4 REC (SVTR_LCNet, CTC khi inference) | PaddleOCR v2.9.1 |
| VietLPR_v1 | TinyDBNet và TinyCRNN dùng GRU hai chiều | Repo `vietdai-bk/VietLPR_v2`, đổi tên local để phân biệt |
| VietLPR_v2 | FastDBNet dùng RepConv và FastCRNN với attention | Notebook `VietLPR_v2.ipynb` bạn cung cấp |

V1 được giữ theo source thực tế: lớp `rnn` là **GRU hai chiều**, không gọi nhầm thành LSTM.
Kiến trúc v2 được tách từ cell định nghĩa model, không chạy cell Colab, tải dữ liệu, video hoặc YOLO.
Mã nguồn ba model được gom trong `third_party/PaddleOCR/`, `third_party/VietLPR_v1/` và `third_party/VietLPR_v2/`.
`bootstrap.py` khoá commit của hai repo ngoài; v2 được tách từ notebook đã chốt SHA256.
Không dùng checkpoint có sẵn trong repo v1 vì chưa biết dữ liệu train và bộ ký tự của chúng.

Ba hệ thống mặc định **train từ đầu**, dùng chung bộ ký tự `0–9, A–Z` và đúng train/val/test hiện có.
V1 và v2 giữ kiến trúc riêng nhưng dùng vòng train mới: ảnh BGR, nhãn cố định, train/val tách riêng,
augmentation độ sáng/tương phản không làm lệch nhãn, REC có resize/pad. V2 dùng shrink-map và BCE/OHEM + Dice;
v1 giữ mask fill/dilate và BCE. Paddle dùng vòng train/loss chính thức và augmentation trong cấu hình xuất ra.
Đây là so sánh ba **hệ thống với cấu hình được ghi lại**, không tuyên bố chỉ thay kiến trúc còn mọi chi tiết khác đều giống nhau.

## Cấu trúc code

```text
experiments/
  configs/                    Cấu hình riêng cho ba hệ thống
  bootstrap.py                Tải source đúng commit, tách kiến trúc v2
  config.py                   Đọc cấu hình và đường dẫn
  data.py                     Đọc split, ảnh BGR và nhãn toàn biển
  torch_models.py             Nạp kiến trúc v1/v2, wrapper xuất model
  ops_v1.py / ops_v2.py        Hình học, hậu xử lý lấy từ nguồn tương ứng
  paddle_postprocess.py       Nhánh quad DB chính thức, chạy bằng NumPy
  train_torch.py              Train/val, best/last, resume, early stopping
  paddle.py                   Chuẩn bị YAML, train/val/export PaddleOCR
  export.py                   Xuất v1/v2 sang ONNX, đối chiếu đầu ra
  runtime.py                  Inference ONNX CPU cho cả ba hệ thống
  metrics.py                  Công thức ACC/CER/Speed/FPS chung
  evaluate.py                 Chọn ngưỡng trên val, khoá model, test và so sánh
  smoke.py                    Kiểm đường đi bằng ít mẫu, một epoch
  tests/                      Kiểm thử metric và quy trình benchmark
  requirements-*.txt           Thư viện riêng cho thí nghiệm
third_party/
  PaddleOCR/                  Repo PaddleOCR đúng commit, không đưa vào Git
  VietLPR_v1/                 Repo v1 đúng commit, không đưa vào Git
  VietLPR_v2/models.py        FastDBNet, FastCRNN, RepConv; lưu trong Git
runs/                         Checkpoint, ONNX, log và kết quả, không đưa vào Git
```

`config.py` tính đường dẫn từ thư mục repository, không phụ thuộc thư mục terminal hiện tại.
Các CLI chạy ở thư mục gốc với `python -m experiments...`.
`third_party/VietLPR_v2/` được lưu trong Git vì đây là mã nguồn duy nhất đã tách từ notebook gốc.

## Môi trường riêng

Giữ nguyên `.venv` của chuẩn hoá dataset. Thí nghiệm dùng **Python 3.11** để tương thích Paddle/PyTorch đã chốt.
Máy đã có Python 3.11. Hai framework đặt ở hai môi trường để tránh xung đột thư viện.

```powershell
py -3.11 -m venv .venv-ocr
.venv-ocr\Scripts\python.exe -m pip install -r experiments/requirements-torch.txt

py -3.11 -m venv .venv-ocr-paddle
.venv-ocr-paddle\Scripts\python.exe -m pip uninstall -y paddlepaddle
.venv-ocr-paddle\Scripts\python.exe -m pip install -r experiments/requirements-paddle.txt

.venv-ocr\Scripts\python.exe -m experiments.bootstrap
.venv-ocr-paddle\Scripts\python.exe -m pip install -r third_party/PaddleOCR/requirements.txt -r experiments/requirements-paddle.txt
```

Mặc định **train GPU, test CPU**. Requirements khoá torch **2.7.1+cu126** và Paddle GPU **2.6.2.post120**.
PyTorch có runtime CUDA 12.6 đi kèm wheel. Paddle dùng CUDA 12.x và cuDNN 8.9 trên máy;
máy hiện có CUDA Toolkit 12.1 và cuDNN 8.9.7. Hai framework dùng môi trường và tiến trình riêng.
Không cài đồng thời `paddlepaddle` CPU và `paddlepaddle-gpu` trong cùng môi trường.
Nguồn wheel: [PyTorch](https://download.pytorch.org/whl/cu126/torch/) và
[Paddle GPU CUDA 12.0](https://www.paddlepaddle.org.cn/packages/stable/cu120/paddlepaddle-gpu/).
Code sẽ báo lỗi nếu yêu cầu GPU nhưng GPU không hoạt động.

## Train và val VietLPR

Train lần lượt trên GPU laptop:

```powershell
.venv-ocr\Scripts\python.exe -m experiments.train_torch --model vietlpr_v1 --task det --device cuda
.venv-ocr\Scripts\python.exe -m experiments.train_torch --model vietlpr_v1 --task rec --device cuda
.venv-ocr\Scripts\python.exe -m experiments.train_torch --model vietlpr_v2 --task det --device cuda
.venv-ocr\Scripts\python.exe -m experiments.train_torch --model vietlpr_v2 --task rec --device cuda
```

Đổi batch size và số epoch trong `configs/`. Batch mặc định đã giảm cho GPU laptop 4 GB, vẫn phải kiểm mức dùng VRAM thực tế.
Mỗi epoch: train → val → cập nhật best nếu cải thiện. DET chọn F1 polygon @0,5; REC chọn ACC rồi CER khi hoà.
`*_last.pth` chứa cả optimizer/scaler để resume; `*_best.pth` dùng xuất ONNX.
Chạy lại cùng thư mục phải thêm `--resume`, hoặc dùng `--run-dir` khác. Không tự ghi đè một lượt train cũ.
Checkpoint best riêng DET/REC là lựa chọn ban đầu; sau đó đánh giá và chọn hậu xử lý cả luồng trên val.

Kiểm code nhanh, không lưu model thành phẩm:

```powershell
.venv-ocr\Scripts\python.exe -m experiments.smoke --device cuda --full-size
.venv-ocr-paddle\Scripts\python.exe -m experiments.paddle smoke --device gpu
.venv-ocr\Scripts\python.exe -m unittest discover -s experiments/tests -v
```

Smoke dùng thư mục tạm và 4 mẫu; `--full-size` giữ kích thước input cấu hình chính.
Không truyền `--full-size` thì DET giảm còn 64 để kiểm nhanh. Đây không phải thí nghiệm hiệu năng.
Weights bị đánh dấu `smoke_only` và bị chặn ở bước test chính.

## Train và val PaddleOCR

```powershell
.venv-ocr-paddle\Scripts\python.exe -m experiments.paddle prepare --device gpu
.venv-ocr-paddle\Scripts\python.exe -m experiments.paddle train --task det
.venv-ocr-paddle\Scripts\python.exe -m experiments.paddle train --task rec
.venv-ocr-paddle\Scripts\python.exe -m experiments.paddle val --task det
.venv-ocr-paddle\Scripts\python.exe -m experiments.paddle val --task rec
```

Paddle train theo `use_gpu: true` trong YAML do lệnh prepare ở trên tạo.
YAML và danh sách nhãn train/val ghi trong `runs/paddleocr/`; dùng ảnh ở dataset gốc, không copy hoặc chia lại ảnh.
Mặc định không tải pretrained weights. Có thể thêm fine-tune sau nhưng phải ghi rõ và tách khỏi thí nghiệm train từ đầu.
Paddle dùng checkpoint `best_accuracy` theo metric validation của framework; tên này dùng cho cả DET và REC.
Paddle tiếp tục lượt bị ngắt bằng `train --task det --resume` hoặc `train --task rec --resume`.
YAML, nhãn và checksum train/val được kiểm trước khi chạy; đổi cấu hình sau khi có weights thì dùng `--run-dir` mới.
Giới hạn 4 luồng tính toán để tránh CPU bị chậm do mở quá nhiều luồng.

## Xuất ONNX FP32

```powershell
.venv-ocr\Scripts\python.exe -m experiments.export --model vietlpr_v1
.venv-ocr\Scripts\python.exe -m experiments.export --model vietlpr_v2
.venv-ocr-paddle\Scripts\python.exe -m experiments.paddle export --task det
.venv-ocr-paddle\Scripts\python.exe -m experiments.paddle export --task rec
```

Mỗi hệ thống có `runs/<tên>/onnx/det.onnx`, `rec.onnx`, `config.json`, `export_check.json`.
V1/v2 xuất batch cố định 1, spatial size theo cấu hình; crop hai tầng nhận dạng tuần tự để giữ cùng điều kiện benchmark.
V2 gộp nhánh RepConv trước export. So kết quả framework với ONNX; dừng nếu sai số vượt ngưỡng.
Model ONNX xuất trực tiếp có sigmoid cho DET và softmax cho REC.

## Chọn val, khoá model và test

```powershell
.venv-ocr\Scripts\python.exe -m experiments.evaluate tune --model paddleocr
.venv-ocr\Scripts\python.exe -m experiments.evaluate tune --model vietlpr_v1
.venv-ocr\Scripts\python.exe -m experiments.evaluate tune --model vietlpr_v2

.venv-ocr\Scripts\python.exe -m experiments.evaluate test --model paddleocr
.venv-ocr\Scripts\python.exe -m experiments.evaluate test --model vietlpr_v1
.venv-ocr\Scripts\python.exe -m experiments.evaluate test --model vietlpr_v2
.venv-ocr\Scripts\python.exe -m experiments.evaluate compare
```

`tune` chỉ mở val, thử lưới nhỏ và chốt ngưỡng theo ACC rồi CER; ghi SHA model/config vào `frozen.json`.
`test` từ chối model thay đổi, model thiếu bước export hoặc weights chỉ chạy thử.
Chạy cả ba lệnh test bằng cùng `.venv-ocr` để giữ cùng ONNX Runtime, không cần import Paddle hoặc torch.
Mặc định CPU **4 thread**, OpenCV **1 thread**, batch **1**, warm-up **20**, lặp **5 lượt**.
Speed là trung bình toàn luồng từ ảnh BGR trong RAM đến chuỗi cuối, gồm DET/crop/REC/hậu xử lý,
không tính đọc ổ đĩa, load model, vẽ hoặc ghi ảnh. FPS = 1000/Speed. Chưa phải FPS camera/video có YOLO.

Bảng `runs/comparison.csv` đúng các cột:

| Model | ACC (%) ↑ | CER (%) ↓ | Speed (ms/ảnh) ↓ | FPS ↑ | Speedup (×) ↑ |
|---|---:|---:|---:|---:|---:|

ACC tính đúng cả chuỗi biển, ghép tầng trên trước; không bỏ mẫu không phát hiện được chữ.
CER = tổng khoảng cách chỉnh sửa / tổng ký tự nhãn chuẩn. Cùng viết hoa/bỏ dấu phân cách, không sửa O thành 0.
Speedup lấy PaddleOCR làm 1×; bảng bị chặn nếu khác tập test hoặc điều kiện máy/thread/runtime.
Các dự đoán từng ảnh và số đo lưu dưới `runs/`, không tạo lại thư mục `reports/` đã bỏ.

## Phạm vi kiểm tra

Đã xác nhận RTX A500 Laptop GPU nhận bởi cả hai framework và train thử đủ DET/REC cho ba hệ thống.
Batch cấu hình đã chạy trên GPU: v1 8/64, v2 32/64, Paddle 8/32 (DET/REC).
Các lượt này chỉ kiểm môi trường và bộ nhớ; chưa phải model thành phẩm.
Đã chạy thử DET và REC một epoch/4 mẫu cho cả ba hệ thống, xuất ONNX,
đối chiếu số học trên ảnh val và chạy luồng nhận dạng ONNX CPU. Có 14 kiểm thử thí nghiệm và 26 kiểm thử chuẩn hoá dữ liệu đã qua.
Code chuẩn bị không bảo đảm model đã train sẽ đạt ACC nào. Kiểm code bằng smoke chỉ xác nhận đường đi và export.
Số mẫu, ngưỡng, checksum và cấu hình được lưu theo từng lượt để giải thích kết quả sau train.
Chưa có Pi 5 nên bảng hiện tại chỉ mang ý nghĩa CPU laptop; khi có Pi phải đo lại.
Chưa chứng nhận mọi họ ảnh nguồn giữa các split độc lập; các giới hạn dữ liệu vẫn như README chuẩn hoá.

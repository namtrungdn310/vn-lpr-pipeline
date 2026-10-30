"""Suy luận OCR biển Việt Nam pretrained; không dùng box hoặc nhãn GT."""

from pathlib import Path

import cv2
import numpy as np

from .metrics import normalize


def session(path, threads):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    engine = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    if len(engine.get_inputs()) != 1 or engine.get_inputs()[0].type not in ("tensor(float)", "tensor(uint8)"):
        raise ValueError("Model cần một input FP32 hoặc ảnh uint8 với chuẩn hoá trong graph")
    return engine


def cct_preprocess(image):
    """Giữ tiền xử lý ảnh crop của app nguồn, không dùng tracking/video."""
    h, w = image.shape[:2]
    if h < 20 or w < 60:
        return None
    if h < 50 or w < 100:
        scale = max(64 / h, 200 / w)
        image = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 15)


def yolo_input(image, size=640):
    """Letterbox theo tỷ lệ, cạnh dài 640, đệm đến bội 32 như nguồn."""
    height, width = image.shape[:2]
    ratio = size / max(height, width)
    rw, rh = round(width * ratio), round(height * ratio)
    dw, dh = (size - rw) % 32 / 2, (size - rh) % 32 / 2
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    resized = cv2.copyMakeBorder(resized, round(dh - 0.1), round(dh + 0.1),
                                 round(dw - 0.1), round(dw + 0.1), cv2.BORDER_CONSTANT,
                                 value=(114, 114, 114))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    data = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    padding = ((data.shape[3] - width * ratio) / 2, (data.shape[2] - height * ratio) / 2)
    return data, ratio, padding


def model_input(image, variant):
    if variant["family"] == "cct":
        processed = cct_preprocess(image)
        if processed is None:
            return None
        return cv2.resize(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), (128, 64))[None]
    if variant["family"] == "lprnet":
        resized = cv2.resize(image, (94, 24)).astype(np.float32)
        return np.ascontiguousarray(((resized - 127.5) / 128).transpose(2, 0, 1)[None])
    return yolo_input(image, variant["image_size"])[0]


def yolo_boxes(output, image, variant, ratio, padding):
    """Lọc confidence, NMS theo lớp, rồi trả tọa độ trên ảnh đầu vào."""
    rows = output[0] if variant["family"] == "yolov5" else output[0].T
    scores = rows[:, 5:] * rows[:, 4:5] if variant["family"] == "yolov5" else rows[:, 4:]
    classes = scores.argmax(axis=1)
    confidence = scores[np.arange(len(scores)), classes]
    nms_confidence = 0.25 if variant["family"] == "yolov8" else variant["confidence"]
    selected = confidence > nms_confidence
    rows, classes, confidence = rows[selected], classes[selected], confidence[selected]
    if not len(rows):
        return []
    xyxy = np.concatenate((rows[:, :2] - rows[:, 2:4] / 2, rows[:, :2] + rows[:, 2:4] / 2), axis=1)
    # NMS dùng offset lớp như nguồn; không gộp các chữ khác nhau.
    offset = classes.astype(np.float32) * 7680
    boxes = np.column_stack((xyxy[:, 0] + offset, xyxy[:, 1] + offset, rows[:, 2:4]))
    kept = cv2.dnn.NMSBoxes(boxes.tolist(), confidence.tolist(), float(nms_confidence), float(variant["iou"]))
    kept = np.asarray(kept).reshape(-1)[:variant["max_det"]]
    if variant["family"] == "yolov8":
        padding = tuple(round(value - 0.1) for value in padding)
        kept = [i for i in kept if confidence[i] >= variant["confidence"]]
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - padding[0]) / ratio
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - padding[1]) / ratio
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, image.shape[1])
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, image.shape[0])
    return [(xyxy[i], variant["chars"][classes[i]], float(confidence[i])) for i in kept]


def yolo_text(boxes, family):
    if not boxes or (family == "yolov5" and not 7 <= len(boxes) <= 10):
        return ""
    if family == "yolov5":
        centers = [((q[0] + q[2]) / 2, (q[1] + q[3]) / 2, char) for q, char, _ in boxes]
        left, right = min(centers, key=lambda c: c[0]), max(centers, key=lambda c: c[0])
        if left[0] == right[0]:
            two_lines = False
        else:
            # Phương trình đường nối hai tâm ngoài cùng, sai số 3 pixel theo nguồn.
            predicted = lambda x: left[1] + (right[1] - left[1]) * (x - left[0]) / (right[0] - left[0])
            two_lines = any(abs(predicted(c[0]) - c[1]) > 3 for c in centers)
        threshold = int(sum(c[1] for c in centers)) // len(centers)
        top = [c for c in centers if int(c[1]) <= threshold]
        bottom = [c for c in centers if int(c[1]) > threshold]
    else:
        centers = [(float(q[0]), float(q[1]), char) for q, char, _ in boxes]
        two_lines = max(c[1] for c in centers) - min(c[1] for c in centers) >= 15
        threshold = float(np.mean([c[1] for c in centers]))
        top = [c for c in centers if c[1] < threshold]
        bottom = [c for c in centers if c[1] >= threshold]
    groups = [top, bottom] if two_lines else [centers]
    return "".join(c[2] for group in groups for c in sorted(group, key=lambda c: c[0]))


def decode_lprnet(output, chars):
    ids = output[0].argmax(axis=0)
    keep = (ids != len(chars) - 1) & np.r_[True, ids[1:] != ids[:-1]]
    return "".join(chars[int(i)] for i in ids[keep])


class PublicVNPipeline:
    def __init__(self, directory, variant, threads=4):
        self.variant = variant
        self.engine = session(Path(directory) / "onnx/model.onnx", threads)
        self.input = self.engine.get_inputs()[0].name

    def __call__(self, image):
        variant = self.variant
        if variant["family"].startswith("yolo"):
            data, ratio, padding = yolo_input(image, variant["image_size"])
        else:
            data = model_input(image, variant)
        if data is None:
            return {"text": "", "boxes": []}
        output = self.engine.run(None, {self.input: data})[0]
        if not np.isfinite(output).all():
            raise ValueError("Output chứa NaN/Inf")
        boxes = []
        if variant["family"] == "cct":
            text = "".join(variant["chars"][int(i)] for i in output.reshape(1, 9, 37).argmax(-1)[0]).rstrip("_")
        elif variant["family"] == "lprnet":
            text = decode_lprnet(output, variant["chars"])
        else:
            boxes = yolo_boxes(output, image, variant, ratio, padding)
            text = yolo_text(boxes, variant["family"])
        return {"text": normalize(text), "boxes": [q.tolist() for q, _, _ in boxes]}

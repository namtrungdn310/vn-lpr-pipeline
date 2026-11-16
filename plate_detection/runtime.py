"""ONNX YOLO26-OBB: letterbox, khung xoay và crop toàn biển."""

import cv2
import numpy as np

from .data import ordered_quad


def letterbox(image, size):
    h, w = image.shape[:2]
    ratio = min(size / h, size / w)
    rw, rh = round(w * ratio), round(h * ratio)
    left, top = round((size - rw) / 2 - 0.1), round((size - rh) / 2 - 0.1)
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, size - rh - top, left, size - rw - left,
                               cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255, ratio, (left, top)


def decode(output, ratio, padding, confidence=0.25, max_det=20):
    """YOLO26 end-to-end xuất [1,N,7]: cx,cy,w,h,score,class,angle."""
    if output.ndim != 3 or output.shape[0] != 1 or output.shape[2] != 7 or not np.isfinite(output).all():
        raise ValueError("Cần ONNX YOLO26-OBB end-to-end [1,N,7], FP32")
    detections = []
    for row in output[0]:
        x, y, w, h, score, cls, angle = map(float, row)
        if score < confidence:
            continue
        if cls != 0 or w <= 0 or h <= 0 or not 0 <= score <= 1:
            raise ValueError("Output không khớp detector một lớp biển số")
        center = np.array([(x - padding[0]) / ratio, (y - padding[1]) / ratio])
        u = np.array([np.cos(angle), np.sin(angle)]) * w / ratio / 2
        v = np.array([-np.sin(angle), np.cos(angle)]) * h / ratio / 2
        q = ordered_quad([center - u - v, center + u - v, center + u + v, center - u + v])
        detections.append({"quad": q.tolist(), "confidence": score})
    # Nhánh one-to-one đã chọn hộp; không thêm NMS của YOLO đời khác.
    return sorted(detections, key=lambda d: -d["confidence"])[:max_det]


class PlateDetector:
    def __init__(self, path, size=640, confidence=0.25, max_det=20, threads=4):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        self.engine = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        inputs = self.engine.get_inputs()
        if len(inputs) != 1 or inputs[0].type != "tensor(float)" or inputs[0].shape != [1, 3, size, size]:
            raise ValueError("ONNX cần input FP32 cố định [1,3,imgsz,imgsz]")
        self.input, self.size = inputs[0].name, size
        self.confidence, self.max_det = confidence, max_det

    def __call__(self, image):
        data, ratio, padding = letterbox(image, self.size)
        output = self.engine.run(None, {self.input: data})[0]
        return decode(output, ratio, padding, self.confidence, self.max_det)


def crop_plate(image, quad, padding=0.03):
    """Nắn khung xoay; giữ tỷ lệ biển hai tầng, thêm đệm nhỏ để tránh mất nét."""
    if not 0 <= padding <= 0.2:
        raise ValueError("Đệm crop ngoài khoảng 0–20%")
    q = ordered_quad(quad)
    q = q.mean(axis=0) + (q - q.mean(axis=0)) * (1 + 2 * padding)
    width = max(2, int(np.ceil(max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])))))
    height = max(2, int(np.ceil(max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])))))
    if width * height > image.shape[0] * image.shape[1] * 16:
        raise ValueError("Kích thước crop bất thường")
    target = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    transform = cv2.getPerspectiveTransform(q.astype(np.float32), target)
    crop = cv2.warpPerspective(image, transform, (width, height), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return crop, np.linalg.inv(transform)

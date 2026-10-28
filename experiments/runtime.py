"""Luồng DET → crop → REC chạy CPU; dùng chung cho chọn val và benchmark."""

import math
from pathlib import Path

import cv2
import numpy as np

from .data import order_quad
from . import ops_v1, ops_v2


def tensor(image):
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def resize_rec(image, cfg):
    model = cfg["model"]
    width, height = model["rec_w"], model["rec_h"]
    if cfg["name"] == "paddleocr":
        ratio = image.shape[1] / max(1, image.shape[0])
        resized_w = min(width, max(1, math.ceil(height * ratio)))
        resized = cv2.resize(image, (resized_w, height)).astype(np.float32) / 127.5 - 1
        output = np.zeros((height, width, 3), np.float32)
        output[:, :resized_w] = resized
        return np.ascontiguousarray(output.transpose(2, 0, 1)[None])
    if model["rec_resize"] == "pad":
        image = ops_v1.letterbox_resize(image, (width, height))
    else:
        image = cv2.resize(image, (width, height))
    return tensor(image)


def preprocess_det(image, cfg):
    size = cfg["model"]["det_size"]
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    data = tensor(resized)
    if cfg["name"] == "paddleocr":
        mean = np.float32([0.485, 0.456, 0.406])[None, :, None, None]
        std = np.float32([0.229, 0.224, 0.225])[None, :, None, None]
        data = (data - mean) / std
    return np.ascontiguousarray(data)


def paddle_boxes(probability, width, height, params):
    from .paddle_postprocess import DBPostProcess

    postprocess = DBPostProcess(
        thresh=params["threshold"],
        box_thresh=params["box_threshold"],
        max_candidates=params.get("max_candidates", 100),
        unclip_ratio=params["unclip"],
    )
    boxes, scores = postprocess.boxes_from_bitmap(
        probability,
        probability > params["threshold"],
        width,
        height,
    )
    return [(np.asarray(box, dtype=np.float32), float(score)) for box, score in zip(boxes, scores)]


def decode_boxes(probability, image, cfg):
    height, width = image.shape[:2]
    p = cfg["postprocess"]
    if cfg["name"] == "vietlpr_v1":
        up = cv2.resize(probability, (width, height), interpolation=cv2.INTER_CUBIC)
        mask = (up > p["threshold"]).astype(np.uint8) * 255
        boxes, _ = ops_v1.get_boxes_from_overlay_mask(
            mask, up, p["min_area"], p["max_lines"], p["max_single_line_height"]
        )
        result = [(ops_v1.pad_rotated_box(q, p["bbox_pad"], width, height), 1.0) for q in boxes]
    elif cfg["name"] == "paddleocr":
        result = paddle_boxes(probability, width, height, p)
    else:
        mh, mw = probability.shape
        contours, _ = cv2.findContours(
            (probability > p["threshold"]).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        result = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:6]:
            if len(contour) < 4 or min(cv2.minAreaRect(contour)[1]) < 1.5:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask, [contour - np.array([x, y], np.int32)], 1)
            score = cv2.mean(probability[y : y + h, x : x + w], mask)[0]
            if score < p["box_threshold"]:
                continue
            quad = ops_v2.fit_quad(contour, p["unclip"], p.get("pad_x", 0.0))
            if min(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[3] - quad[0])) < 3.5:
                continue
            quad[:, 0] = np.clip((quad[:, 0] + 0.5) * width / mw - 0.5, 0, width - 1)
            quad[:, 1] = np.clip((quad[:, 1] + 0.5) * height / mh - 0.5, 0, height - 1)
            result.append((quad, float(score)))
    return sorted(
        result, key=lambda pair: (float(pair[0][:, 1].mean()), float(pair[0][:, 0].mean()))
    )


def crop_line(image, quad, cfg):
    if cfg["name"] != "vietlpr_v2":
        return resize_rec(ops_v1.warp_crop(image, quad), cfg)
    model = cfg["model"]
    height, width = model["rec_h"], model["rec_w"]
    if model["rec_resize"] == "stretch":
        out_width = width
    else:
        qw = max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3]))
        qh = max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1]))
        out_width = int(np.clip(round(qw * height / max(qh, 1)), 8, width))
    target = np.float32([[0, 0], [out_width, 0], [out_width, height], [0, height]])
    transform = cv2.getPerspectiveTransform(quad.astype(np.float32), target)
    out = cv2.warpPerspective(
        image,
        transform,
        (out_width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if out_width < width:
        left = (width - out_width) // 2
        out = cv2.copyMakeBorder(
            out, 0, 0, left, width - out_width - left, cv2.BORDER_CONSTANT, value=0
        )
    return tensor(out)


def decode_ctc(output, chars):
    if output.ndim != 3 or output.shape[-1] != len(chars) + 1:
        raise ValueError(f"Output REC hoặc bộ ký tự sai: {output.shape}")
    row = output.argmax(-1)[0]
    keep = (row != 0) & np.r_[True, row[1:] != row[:-1]]
    return "".join(chars[int(index) - 1] for index in row[keep])


class Pipeline:
    def __init__(self, cfg, det_forward, rec_forward):
        self.cfg, self.det_forward, self.rec_forward = cfg, det_forward, rec_forward

    def __call__(self, image):
        probability = self.det_forward(preprocess_det(image, self.cfg))[0, 0]
        detections = decode_boxes(probability, image, self.cfg)
        texts = [
            decode_ctc(self.rec_forward(crop_line(image, quad, self.cfg)), self.cfg["chars"])
            for quad, _ in detections
        ]
        return {
            "text": "".join(texts),
            "lines": texts,
            "boxes": [quad.tolist() for quad, _ in detections],
        }


def onnx_pipeline(cfg, model_dir=None):
    import onnxruntime as ort

    directory = Path(model_dir or Path(cfg["run_dir"]) / "onnx")
    sessions = []
    for name in ("det", "rec"):
        options = ort.SessionOptions()
        options.intra_op_num_threads = cfg["benchmark"]["threads"]
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            str(directory / f"{name}.onnx"), options, providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        sessions.append(lambda x, s=session, key=input_name: s.run(None, {key: x})[0])
    return Pipeline(cfg, *sessions)

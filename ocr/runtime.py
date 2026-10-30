"""Suy luận VietLPR V1/V2 theo nguồn bằng ONNX."""

import hashlib
from pathlib import Path
import cv2
import numpy as np
from .metrics import normalize
from . import ops_v1
from .upstream import v2_config
from third_party.VietLPR_v2.inference import LPR as SourceV2


def tensor(image):
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def resize_rec(image, cfg):
    model = cfg["model"]
    width, height = (model["rec_w"], model["rec_h"])
    if model["rec_resize"] == "pad":
        image = ops_v1.letterbox_resize(image, (width, height))
    else:
        image = cv2.resize(image, (width, height))
    return tensor(image)


def preprocess_det(image, cfg):
    size = cfg["model"]["det_size"]
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    data = tensor(resized)
    return np.ascontiguousarray(data)


def decode_ctc(output, chars):
    if output.ndim != 3 or output.shape[-1] != len(chars) + 1:
        raise ValueError(f"Output REC hoặc bộ ký tự sai: {output.shape}")
    row = output.argmax(-1)[0]
    keep = (row != 0) & np.r_[True, row[1:] != row[:-1]]
    return "".join((chars[int(index) - 1] for index in row[keep]))


class ForwardSession:
    def __init__(self, forward):
        self.forward = forward

    def run(self, outputs, feeds):
        return [self.forward(feeds["input"])]


def source_v2_runtime(cfg, det_forward=None, rec_forward=None):
    source = SourceV2.__new__(SourceV2)
    source.cfg = v2_config(cfg)
    source.S, source.rh, source.rw = (
        cfg["model"]["det_size"],
        cfg["model"]["rec_h"],
        cfg["model"]["rec_w"],
    )
    source.chars = cfg["chars"]
    source.det, source.rec = ForwardSession(det_forward), ForwardSession(rec_forward)
    return source


def decode_boxes(probability, image, cfg):
    height, width = image.shape[:2]
    p = cfg["postprocess"]
    if cfg["name"] == "vietlpr_v1":
        up = cv2.resize(probability, (width, height), interpolation=cv2.INTER_CUBIC)
        mask = (up > p["threshold"]).astype(np.uint8) * 255
        boxes, _ = ops_v1.get_boxes_from_overlay_mask(
            mask, up, p["min_area"], p["max_lines"], p["max_single_line_height"]
        )
        return [(ops_v1.pad_rotated_box(q, p["bbox_pad"], width, height), 1.0) for q in boxes]
    if cfg["name"] == "vietlpr_v2":
        return source_v2_runtime(cfg, lambda x: probability[None, None]).detect(image)
    raise ValueError("Runtime này chỉ dùng VietLPR V1/V2; Paddle dùng paddle_suite_runtime")


def crop_line(image, quad, cfg):
    if cfg["name"] == "vietlpr_v2":
        return tensor(source_v2_runtime(cfg).crop(image, np.float32(quad)))
    return resize_rec(ops_v1.warp_crop(image, quad), cfg)


def rec_input(image, cfg):
    if cfg["name"] == "vietlpr_v2":
        h, w = image.shape[:2]
        return crop_line(image, np.float32([[0, 0], [w, 0], [w, h], [0, h]]), cfg)
    return resize_rec(image, cfg)


def postprocess_candidates(cfg):
    original = cfg["postprocess"]
    candidates = [original.copy()]
    if cfg["name"] == "vietlpr_v1":
        return candidates
    if cfg["name"] == "vietlpr_v2":
        for th in (0.3, 0.4, 0.5):
            for uc in (1.4, 1.6, 1.8, 2.0):
                for bt in (0.5, 0.7):
                    for px in (0.0, 0.1):
                        params = {
                            **original,
                            "threshold": th,
                            "unclip": uc,
                            "box_threshold": bt,
                            "pad_x": px,
                        }
                        candidates.append(params)
    return candidates


def inference_signature():
    paths = [
        Path(__file__).parent / n
        for n in (
            "runtime.py",
            "ops_v1.py",
            "upstream.py",
            "data.py",
            "metrics.py",
        )
    ]
    paths += [Path(__file__).resolve().parents[1] / "third_party/VietLPR_v2/inference.py"]
    return hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()


class Pipeline:
    def __init__(self, cfg, det_forward, rec_forward):
        self.cfg, self.det_forward, self.rec_forward = cfg, det_forward, rec_forward
        self.source = (
            source_v2_runtime(cfg, det_forward, rec_forward)
            if cfg["name"] == "vietlpr_v2"
            else None
        )

    def __call__(self, image):
        if self.source is not None:
            self.source.cfg = v2_config(self.cfg)
            result = self.source(image)
            return {
                "text": normalize(result["plate"]),
                "lines": [t for t, c in result["lines"]],
                "boxes": [q.tolist() for q in result["boxes"]],
                "confidence": result["conf"],
            }
        probability = self.det_forward(preprocess_det(image, self.cfg))[0, 0]
        detections = decode_boxes(probability, image, self.cfg)
        texts = [
            decode_ctc(self.rec_forward(crop_line(image, quad, self.cfg)), self.cfg["chars"])
            for quad, _ in detections
        ]
        return {
            "text": normalize("".join(texts)),
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

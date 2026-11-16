"""Nối detector toàn biển với các pipeline OCR đã chốt."""

import json
from pathlib import Path

import cv2
import numpy as np

from ocr.metrics import normalize
from plate_detection.data import load_config, sha256
from plate_detection.runtime import PlateDetector, crop_plate


def ocr_names(include_public=True):
    from ocr.adaptation.common import settings

    return [m["name"] for m in settings()["models"]]


def load_ocr(name, threads=4, load_runtime=True):
    """Chỉ dùng bảy model đã chốt sau adaptation FP32."""
    from ocr.adaptation.common import validate_frozen
    from ocr.adaptation.runtime import pipeline

    state = validate_frozen(name)
    return (pipeline(name, threads) if load_runtime else None), state


def load_detector(cfg=None, load_runtime=True):
    cfg = cfg or load_config()
    directory = Path(cfg["run_dir"]) / "onnx"
    frozen = json.loads((directory / "verified.json").read_text(encoding="utf-8"))
    location_keys = {"dataset", "raw_dataset", "run_dir", "config_path"}
    portable_config = lambda c: {k: v for k, v in c.items() if k not in location_keys}
    if (
        frozen["onnx_sha256"] != sha256(directory / "model.onnx")
        or portable_config(frozen["config"]) != portable_config(cfg)
    ):
        raise ValueError("Detector/config đã đổi sau export; cần xác nhận lại trên val")
    inf = cfg["inference"]
    return (
        PlateDetector(
            directory / "model.onnx",
            size=cfg["train"]["imgsz"],
            confidence=inf["confidence"],
            max_det=inf["max_det"],
            threads=inf["threads"],
        )
        if load_runtime
        else None
    ), frozen


class Pipeline:
    def __init__(self, detector, ocr, padding=0.03):
        self.detector, self.ocr, self.padding = detector, ocr, padding

    def __call__(self, image):
        plates = []
        for detection in self.detector(image):
            crop, inverse = crop_plate(image, detection["quad"], self.padding)
            result = self.ocr(crop)
            line_boxes = result.get("boxes", [])
            if line_boxes:
                line_boxes = cv2.perspectiveTransform(
                    np.float32(line_boxes).reshape(-1, 1, 2), inverse
                )
                line_boxes = line_boxes.reshape(-1, 4, 2).tolist()
            plates.append(
                {
                    **detection,
                    "text": normalize(result["text"]),
                    "line_boxes": line_boxes,
                    "lines": result.get("lines", []),
                }
            )
        return plates


def make_pipeline(name="VietLPR_v1", cfg=None):
    cfg = cfg or load_config()
    detector, det_state = load_detector(cfg)
    ocr, ocr_state = load_ocr(name, cfg["inference"]["threads"])
    return Pipeline(detector, ocr, cfg["inference"]["crop_padding"]), {
        "detector": det_state,
        "ocr": ocr_state,
    }

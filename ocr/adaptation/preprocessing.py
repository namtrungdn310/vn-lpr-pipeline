"""Tiền xử lý CCT nhất quán khi train và inference, không bỏ ảnh nhỏ."""

import cv2
import numpy as np

from ocr.public_vn_runtime import cct_preprocess


def cct_image(pixels):
    h, w = pixels.shape[:2]
    if h < 20 or w < 60:
        scale = max(20 / h, 60 / w)
        pixels = cv2.resize(
            pixels,
            (max(60, round(w * scale)), max(20, round(h * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
    processed = cct_preprocess(pixels)
    if processed is None:
        raise ValueError("Không thể xử lý ảnh CCT")
    return processed


def cct_input(pixels):
    return cv2.resize(cv2.cvtColor(cct_image(pixels), cv2.COLOR_BGR2RGB), (128, 64))[None].astype(
        np.float32
    )

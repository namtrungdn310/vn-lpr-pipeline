"""Xem polygon nguồn, khung OBB và crop; không sửa nhãn khi hiển thị."""

import cv2
import numpy as np

from .data import read_image
from .runtime import crop_plate


def show_labels(report, count=12, split="train"):
    import matplotlib.pyplot as plt

    rows = [r for r in report["records"] if r["split"] == split]
    # Ưu tiên mẫu nhiều nền dư và gần mép để rà phép chuyển.
    rows = sorted(rows, key=lambda r: min(r["polygon_to_obb_area_ratios"], default=1))[:count]
    fig, axes = plt.subplots(max(1, int(np.ceil(len(rows) / 3))), 3,
                             figsize=(15, max(4, int(np.ceil(len(rows) / 3)) * 4)), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, row in zip(axes.flat, rows):
        image = read_image(row["image"])
        h, w = image.shape[:2]
        from pathlib import Path
        for line in Path(row["label"]).read_text().splitlines():
            if line.strip():
                q = np.array(line.split()[1:], dtype=float).reshape(-1, 2) * [w, h]
                cv2.polylines(image, [np.int32(q)], True, (255, 160, 0), 2)
        for q in row["boxes"]:
            cv2.polylines(image, [np.int32(np.array(q) * [w, h])], True, (0, 220, 0), 2)
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(row["relative"][:42], fontsize=8)
    fig.suptitle("Xanh: OBB; cam: polygon gốc. Ưu tiên mẫu có nền dư nhiều.")
    plt.tight_layout()
    return fig


def show_prediction(image, pipeline):
    import matplotlib.pyplot as plt

    plates = pipeline(image)
    canvas = image.copy()
    for plate in plates:
        cv2.polylines(canvas, [np.int32(plate["quad"])], True, (0, 220, 0), 2)
        p = np.int32(plate["quad"])[0]
        cv2.putText(canvas, plate["text"], tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
    fig, axes = plt.subplots(1, max(2, len(plates) + 1), figsize=(15, 5), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    axes.flat[0].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    axes.flat[0].set_title("YOLO → OCR")
    for axis, plate in zip(list(axes.flat)[1:], plates):
        crop, _ = crop_plate(image, plate["quad"], pipeline.padding)
        axis.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        axis.set_title(plate["text"] or "Không đọc được")
    plt.tight_layout()
    return fig

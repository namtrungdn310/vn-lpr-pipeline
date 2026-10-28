"""Đọc đúng split đã chốt; không chia lại ngẫu nhiên."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .config import CHARS


def image(path):
    result = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if result is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    return result


def order_quad(points):
    points = np.asarray(points, np.float32).reshape(-1, 2)
    if len(points) != 4:
        raise ValueError("Cần quad bốn góc")
    sides = points[np.argsort(points[:, 0], kind="stable")]
    tl, bl = sides[:2][np.argsort(sides[:2, 1])]
    tr, br = sides[2:][np.argsort(sides[2:, 1])]
    return np.float32([tl, tr, br, bl])


def read_split(root, split, branch):
    if split not in ("train", "val", "test") or branch not in ("det", "rec"):
        raise ValueError("Split hoặc nhánh dữ liệu không hợp lệ")
    base = Path(root).resolve(strict=True) / split / branch
    rows, seen = [], set()
    for line in (base / f"{branch}_labels.txt").read_text(encoding="utf-8-sig").splitlines():
        relative, payload = line.split("\t", 1)
        path = (base / relative).resolve(strict=True)
        if not path.is_relative_to(base / "images") or path in seen:
            raise ValueError(f"Tham chiếu ảnh sai hoặc lặp: {relative}")
        seen.add(path)
        if branch == "det":
            objects = json.loads(payload)
            if not objects:
                raise ValueError("Ảnh DET không có dòng chữ")
            boxes = [order_quad(o["points"]) for o in objects]
            indices = sorted(
                range(len(boxes)),
                key=lambda i: (float(boxes[i][:, 1].mean()), float(boxes[i][:, 0].mean())),
            )
            texts = [objects[i]["transcription"] for i in indices]
            if any(not t or set(t) - set(CHARS) for t in texts):
                raise ValueError("Bộ ký tự DET khác cấu hình")
            rows.append(
                {
                    "path": path,
                    "relative": path.relative_to(Path(root).resolve()).as_posix(),
                    "boxes": [boxes[i] for i in indices],
                    "text": "".join(texts),
                    "texts": texts,
                }
            )
        else:
            if not payload or set(payload) - set(CHARS):
                raise ValueError("Bộ ký tự REC khác cấu hình")
            rows.append(
                {
                    "path": path,
                    "relative": path.relative_to(Path(root).resolve()).as_posix(),
                    "text": payload,
                }
            )
    if not rows:
        raise ValueError("Split rỗng")
    return rows


def split_signature(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["relative"].encode())
        digest.update(row["text"].encode())
        for box in row.get("boxes", []):
            digest.update(np.asarray(box, dtype=np.float32).tobytes())
        digest.update(hashlib.sha256(row["path"].read_bytes()).digest())
    return digest.hexdigest()

"""Áp dụng nhóm trùng đã duyệt; không tự kết luận hai biển giống nhau từ hash."""

import json
from pathlib import Path

import cv2
import numpy as np

from ocr.config import ROOT, save_json
from .data import read_image, sha256

REVIEW = ROOT / "configs/detection/split_review.json"


def apply_split_review(report):
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    rows = report["records"]
    excluded = set()
    for group in review["groups"]:
        members = {(m["split"], m["family"]) for m in group["members"]}
        matched = [r for r in rows if (r["split"], r["family"]) in members]
        if not matched:
            continue
        actual = {Path(r["image"]).name: r["image_sha256"] for r in matched}
        if actual != group["source_images"]:
            raise ValueError("Nguồn của nhóm trùng đã duyệt thay đổi; cần rà lại trước khi dựng")
        excluded.update(r["image"] for r in matched if r["split"] != group["keep_split"])
    report["split_quarantine"] = [r for r in rows if r["image"] in excluded]
    report["records"] = [r for r in rows if r["image"] not in excluded]
    report["usable_counts"] = {s: sum(r["split"] == s for r in report["records"]) for s in report["counts"]}
    report["split_review_sha256"] = sha256(REVIEW)
    return report


def near_duplicates(report, destination=None):
    """Tạo ứng viên pHash để người dùng rà, không tự gộp danh tính."""
    rows = report["records"]
    hashes, grays = [], []
    for row in rows:
        gray = cv2.cvtColor(read_image(row["image"]), cv2.COLOR_BGR2GRAY)
        dct = cv2.dct(cv2.resize(gray, (32, 32)).astype(np.float32))[:8, :8].ravel()
        hashes.append(int.from_bytes(np.packbits(dct > np.median(dct[1:])).tobytes(), "little"))
        grays.append(cv2.resize(gray, (96, 96)).astype(np.float32))
    hashes = np.array(hashes, dtype=np.uint64)
    lookup = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    pairs = []
    for i, row in enumerate(rows):
        if row["split"] == "train":
            continue
        indices = np.array([j for j, other in enumerate(rows) if other["split"] != row["split"]
                            and (other["split"] == "train" or j < i)], dtype=int)
        distances = lookup[(hashes[indices] ^ hashes[i]).view(np.uint8).reshape(-1, 8)].sum(axis=1)
        for j, distance in zip(indices[distances <= 4], distances[distances <= 4]):
            x, y = grays[i], grays[j]
            mx, my, vx, vy = x.mean(), y.mean(), x.var(), y.var()
            covariance = ((x - mx) * (y - my)).mean()
            ssim = (2 * mx * my + 6.5025) * (2 * covariance + 58.5225) / (
                (mx * mx + my * my + 6.5025) * (vx + vy + 58.5225))
            pairs.append({"a": row["image"], "a_split": row["split"], "b": rows[int(j)]["image"],
                          "b_split": rows[int(j)]["split"], "hamming": int(distance), "global_ssim": float(ssim)})
    result = {"candidate_pairs": pairs, "threshold_hamming": 4,
              "identity_leakage_certified": False, "note": "Cần rà cả ảnh và chữ; xe khác có thể cùng nền."}
    save_json(destination or ROOT / "runs/plate_detection/near_duplicate_review.json", result)
    return result

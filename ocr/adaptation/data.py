"""Giữ split đã chốt; chỉ tạo đầu vào train/val, không dùng test."""

import csv
import json

import cv2

from ocr.config import ROOT, save_json
from ocr.data import read_split, image, split_signature
from .common import settings, directory, digest, read_json


def verify_dataset():
    cfg = settings()
    baseline = read_json(ROOT / "metadata/baseline.json")
    if baseline["content_fingerprint"] != cfg["dataset_fingerprint"]:
        raise ValueError("Baseline dataset khác bản đã chốt")
    root = ROOT / cfg["dataset"]
    for relative, expected in baseline["files"].items():
        if digest(root / relative) != expected:
            raise ValueError(f"Dataset bị thay đổi: {relative}")
    return root


def prepare():
    root = verify_dataset()
    out = directory() / "data"
    out.mkdir(parents=True, exist_ok=True)
    previous = read_json(out / "manifest.json") if (out / "manifest.json").is_file() else None
    if previous and previous.get("dataset") == settings()["dataset_fingerprint"]:
        if all(
            (out / f"{split}_{branch}.txt").is_file()
            for split in ("train", "val")
            for branch in ("det", "rec")
        ) and all((out / f"{split}_cct.csv").is_file() for split in ("train", "val")):
            return out
    signatures = {}
    for split in ("train", "val"):
        for branch in ("det", "rec"):
            rows = read_split(root, split, branch)
            signatures[f"{split}/{branch}"] = split_signature(rows)
            lines = []
            for row in rows:
                payload = (
                    row["text"]
                    if branch == "rec"
                    else json.dumps(
                        [
                            {"transcription": text, "points": quad.tolist()}
                            for text, quad in zip(row["texts"], row["boxes"])
                        ],
                        ensure_ascii=False,
                    )
                )
                lines.append(str(row["path"].resolve()).replace("\\", "/") + "\t" + payload)
            (out / f"{split}_{branch}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        # CCT đọc toàn biển như lúc inference, không dùng box GT để crop.
        from .preprocessing import cct_image

        rows = read_split(root, split, "det")
        with (out / f"{split}_cct.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image_path", "plate_text"])
            for i, row in enumerate(rows):
                if i % 100 == 0:
                    print(f"CCT {split}: {i}/{len(rows)}", flush=True)
                if len(row["text"]) > 9:
                    raise ValueError("CCT nguồn chỉ có chín slot; không cắt nhãn")
                pixels = cct_image(image(row["path"]))
                target = out / "cct" / split / f"{i:06d}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                success, encoded = cv2.imencode(".png", pixels)
                if not success:
                    raise ValueError("Không ghi được ảnh train CCT")
                encoded.tofile(target)
                writer.writerow([target.relative_to(out).as_posix(), row["text"]])
    save_json(
        out / "manifest.json",
        {
            "dataset": settings()["dataset_fingerprint"],
            "signatures": signatures,
            "splits": ["train", "val"],
            "test_used": False,
        },
    )
    return out


def rows(split, branch="det"):
    return read_split(ROOT / settings()["dataset"], split, branch)

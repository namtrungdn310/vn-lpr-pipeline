"""Kiểm nhãn OBB Roboflow và dựng dữ liệu train một lớp biển số."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import yaml
from shapely.geometry import Polygon

from ocr.config import ROOT, save_json

CONFIG = ROOT / "configs/detection/yolo26_obb.json"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config(path=CONFIG, variant="n"):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("raw_dataset", "dataset", "run_dir"):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    if variant not in cfg["variants"]:
        raise ValueError("Dự án chỉ dùng YOLO26n-OBB (variant n)")
    cfg["model"] = f"yolo26{variant}-obb.pt"
    cfg["run_dir"] = str(Path(cfg["run_dir"]).with_name(f"yolo26{variant}_obb"))
    cfg["train"]["batch"] = cfg["variants"][variant]
    return cfg


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    return image


def ordered_quad(points):
    """Giữ chu kỳ góc; không ép cạnh dài làm chiều ngang của biển hai tầng."""
    q = np.asarray(points, dtype=np.float32)
    if q.shape != (4, 2) or not np.isfinite(q).all():
        raise ValueError("Cần bốn góc hữu hạn")
    hull = cv2.convexHull(q).reshape(-1, 2)
    if len(hull) != 4 or cv2.contourArea(hull) < 1:
        raise ValueError("Khung suy biến hoặc không lồi")
    # Đi theo chiều kim đồng hồ trên ảnh, bắt đầu ở góc trên trái gần nhất.
    center = hull.mean(axis=0)
    hull = hull[np.argsort(np.arctan2(hull[:, 1] - center[1], hull[:, 0] - center[0]))]
    return np.roll(hull, -int(np.argmin(hull.sum(axis=1))), axis=0)


def parse_label(line, width, height, classes):
    parts = line.split()
    if len(parts) == 5:
        raise ValueError("Nhãn AABB 5 cột: cần gán/duyệt OBB, không tự thêm góc xoay")
    if len(parts) < 9 or len(parts) % 2 != 1:
        raise ValueError("Cần ít nhất bốn góc polygon hoặc YOLO OBB 9 cột")
    values = np.asarray([float(v) for v in parts], dtype=np.float64)
    if not np.isfinite(values).all() or values[0] != int(values[0]) or int(values[0]) not in classes:
        raise ValueError("Lớp hoặc tọa độ không hợp lệ")
    q = values[1:].reshape(-1, 2)
    if len(q) > 4 and np.allclose(q[0], q[-1], atol=1e-7, rtol=0):
        q = q[:-1]
    if (q < 0).any() or (q > 1).any():
        raise ValueError("Tọa độ phải chuẩn hoá trong [0,1]")
    pixels = q * [width, height]
    polygon = Polygon(pixels)
    if len(q) < 4 or not polygon.is_valid or polygon.area < 4:
        raise ValueError("Polygon tự cắt hoặc suy biến")
    # Polygon có hình học thật: lấy chữ nhật xoay nhỏ nhất, không tạo góc từ AABB.
    q = ordered_quad(cv2.boxPoints(cv2.minAreaRect(pixels.astype(np.float32))))
    edges = np.roll(q, -1, axis=0) - q
    lengths = np.linalg.norm(edges, axis=1)
    if min(lengths) < 2:
        raise ValueError("Biển nhỏ hơn hai pixel hoặc suy biến")
    cosine = np.sum(edges * np.roll(edges, -1, axis=0), axis=1) / (lengths * np.roll(lengths, -1))
    if np.abs(cosine).max() > 0.08:
        raise ValueError("Không tạo được chữ nhật xoay hợp lệ")
    # Chữ nhật bao polygon có thể vượt biên; bước prepare đệm ảnh, không cắt góc.
    return q / [width, height]


def audit_dataset(directory):
    root = Path(directory).resolve()
    path = root / "data.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Giải nén export Roboflow có data.yaml vào: {root}")
    source = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = source.get("names", {})
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    if not names or any(n not in {"license-plate-long", "license-plate-short", "license-plate", "plate"}
                        for n in names.values()):
        raise ValueError(f"Cần kiểm tra lớp biển số của nguồn: {names}")
    records, errors, counts, hashes, families = [], [], {}, {}, {}
    quarantined = []
    columns, padded, loose = {}, 0, 0
    rotated = 0
    for split, candidates in (("train", ("train",)), ("val", ("valid", "val")), ("test", ("test",))):
        folders = [root / d for d in candidates if (root / d / "images").is_dir()]
        if len(folders) != 1:
            errors.append(f"Thiếu hoặc trùng thư mục {split}/images")
            continue
        folder = folders[0]
        images = sorted(p for p in (folder / "images").rglob("*") if p.suffix.lower() in EXTENSIONS)
        counts[split] = len(images)
        if not images:
            errors.append(f"Không có ảnh {split}")
        used_labels = set()
        for image_path in images:
            relative = image_path.relative_to(folder / "images")
            label = folder / "labels" / relative.with_suffix(".txt")
            try:
                if not image_path.resolve().is_relative_to(root) or not label.resolve().is_relative_to(root):
                    raise ValueError("Ảnh/nhãn trỏ ra ngoài nguồn")
                pixels = read_image(image_path)
                # Hash pixel phát hiện cả bản sao đổi tên/định dạng không mất dữ liệu.
                pixel_hash = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
                family = image_path.stem.split(".rf.")[0]
                for table, key in ((hashes, pixel_hash), (families, family)):
                    if key in table and table[key] != split:
                        raise ValueError(f"Ảnh hoặc họ Roboflow xuất hiện ở nhiều split: {family}")
                    table[key] = split
                lines = label.read_text(encoding="utf-8").splitlines()
                used_labels.add(label.resolve())
                for line in lines:
                    if line.strip():
                        n = str(len(line.split()))
                        columns[n] = columns.get(n, 0) + 1
                boxes = [parse_label(line, pixels.shape[1], pixels.shape[0], names) for line in lines if line.strip()]
                pad = [0, 0, 0, 0]
                ratios = []
                for q, line in zip(boxes, [line for line in lines if line.strip()]):
                    raw_q = np.array(line.split()[1:], dtype=float).reshape(-1, 2) * [pixels.shape[1], pixels.shape[0]]
                    area_ratio = Polygon(raw_q).area / cv2.contourArea((q * [pixels.shape[1], pixels.shape[0]]).astype(np.float32))
                    ratios.append(area_ratio)
                    loose += int(area_ratio < 0.8)
                    p = q * [pixels.shape[1], pixels.shape[0]]
                    needed = [-p[:, 0].min(), -p[:, 1].min(), p[:, 0].max() - pixels.shape[1], p[:, 1].max() - pixels.shape[0]]
                    pad = [max(a, int(np.ceil(max(0, b - 1e-4)))) for a, b in zip(pad, needed)]
                padded += int(any(pad))
                for q in boxes:
                    p = q * [pixels.shape[1], pixels.shape[0]]
                    edges = np.roll(p, -1, axis=0) - p
                    rotated += int(np.any(np.min(np.abs(edges), axis=1) / np.linalg.norm(edges, axis=1) > 0.02))
                records.append({"split": split, "image": str(image_path), "label": str(label),
                                "relative": relative.as_posix(), "boxes": [q.tolist() for q in boxes],
                                "pixel_sha256": pixel_hash, "image_sha256": sha256(image_path),
                                "label_sha256": sha256(label), "family": family,
                                "padding_ltrb": pad, "polygon_to_obb_area_ratios": ratios,
                                "size": [pixels.shape[1], pixels.shape[0]]})
            except (ValueError, OSError) as error:
                message = str(error)
                if split == "train" and message in {"Polygon tự cắt hoặc suy biến", "Biển nhỏ hơn hai pixel hoặc suy biến"}:
                    # Loại cả ảnh train lỗi khỏi bản dựng; giữ nguyên nguồn, không bỏ riêng một hộp.
                    quarantined.append({"image": str(image_path), "label": str(label), "reason": message})
                else:
                    errors.append(f"{image_path.relative_to(root)}: {error}")
        orphan = {p.resolve() for p in (folder / "labels").rglob("*.txt")} - used_labels
        if orphan:
            errors.append(f"{split}: {len(orphan)} nhãn không có ảnh hợp lệ")
    if records and not rotated:
        errors.append("Tất cả nhãn đều thẳng: cần xác minh/gán nhãn xoay trước khi train OBB")
    if any(not any(r["split"] == split for r in records) for split in ("train", "val", "test")):
        errors.append("Mỗi split cần ít nhất một ảnh hợp lệ sau khi rà nhãn")
    return {"ready": not errors, "source": str(root), "source_names": names, "counts": counts,
            "rotated_boxes": rotated, "errors": errors, "records": records,
            "label_columns": columns, "padded_images": padded, "loose_obb_boxes": loose,
            "quarantined_train_images": quarantined,
            "usable_counts": {split: sum(r["split"] == split for r in records) for split in counts},
            "identity_leakage_certified": False,
            "note": "Đã kiểm pixel và họ tên file; vẫn cần duyệt cùng xe/biển, ảnh gần trùng và nguồn OCR."}


def prepare_dataset(cfg, source=None):
    from .quality import apply_split_review

    report = audit_dataset(source or cfg["raw_dataset"])
    report = apply_split_review(report)
    save_json(ROOT / "runs/plate_detection/label_audit.json", report)
    if not report["ready"]:
        raise ValueError("Nhãn chưa đạt; xem runs/plate_detection/label_audit.json\n" + "\n".join(report["errors"][:10]))
    destination = Path(cfg["dataset"]).resolve()
    protected = [(ROOT / name).resolve() for name in ("vn-plate-dataset_v2", "sources", "metadata", "third_party")]
    raw = Path(report["source"])
    if (destination.exists() or not destination.is_relative_to((ROOT / "runs/plate_detection").resolve())
            or any(destination.is_relative_to(p) or p.is_relative_to(destination) for p in [raw, *protected])):
        raise ValueError("Đầu ra cần thư mục mới trong runs/plate_detection, tách khỏi nguồn")
    destination.mkdir(parents=True)
    for record in report["records"]:
        relative = Path(record["relative"])
        left, top, right, bottom = record["padding_ltrb"]
        boxes = np.asarray(record["boxes"], dtype=float).reshape(-1, 4, 2)
        if any((left, top, right, bottom)):
            relative = relative.with_suffix(".png")
        folder = destination / record["split"]
        target = folder / "images" / relative
        label = folder / "labels" / relative.with_suffix(".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        if any((left, top, right, bottom)):
            image = cv2.copyMakeBorder(read_image(record["image"]), top, bottom, left, right,
                                       cv2.BORDER_CONSTANT, value=(114, 114, 114))
            encoded = cv2.imencode(".png", image)[1]
            encoded.tofile(str(target))
            width, height = record["size"]
            boxes = (boxes * [width, height] + [left, top]) / [width + left + right, height + top + bottom]
        else:
            shutil.copy2(record["image"], target)
        if (boxes < -1e-6).any() or (boxes > 1 + 1e-6).any():
            raise ValueError("Đệm ảnh chưa chứa đủ khung")
        boxes = boxes.clip(0, 1)  # Chỉ khử sai số dấu phẩy động sát biên.
        label.write_text("".join("0 " + " ".join(f"{v:.8f}" for xy in q for v in xy) + "\n"
                                 for q in boxes), encoding="utf-8")
    data = {"path": str(destination), "train": "train/images", "val": "val/images", "test": "test/images",
            "names": {0: "license-plate"}}
    (destination / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    save_json(destination / "source_audit.json", report)
    return destination / "data.yaml"


def dataset_signature(directory):
    root = Path(directory)
    files = [root / "data.yaml", root / "source_audit.json"]
    for split in ("train", "val", "test"):
        files += sorted(p for p in (root / split).rglob("*") if p.is_file() and p.suffix != ".cache")
    return hashlib.sha256("\n".join(f"{p.relative_to(root).as_posix()}:{sha256(p)}" for p in files).encode()).hexdigest()


def check_prepared(cfg):
    """Đối chiếu bản dựng với recipe, pixel nguồn và quyết định chia nhóm."""
    from .quality import REVIEW

    root = Path(cfg["dataset"])
    report = audit_dataset(root)
    if not report["ready"]:
        raise ValueError("Bản dựng chưa đạt: " + "; ".join(report["errors"][:10]))
    source = json.loads((root / "source_audit.json").read_text(encoding="utf-8"))
    if source.get("split_review_sha256") != sha256(REVIEW):
        raise ValueError("Quyết định nhóm trùng khác bản dựng")
    padded = 0
    for row in source["records"]:
        if sha256(row["image"]) != row["image_sha256"] or sha256(row["label"]) != row["label_sha256"]:
            raise ValueError("Ảnh/nhãn gốc đã thay đổi")
        relative = Path(row["relative"])
        left, top, right, bottom = row["padding_ltrb"]
        q = np.asarray(row["boxes"], dtype=float).reshape(-1, 4, 2)
        if any((left, top, right, bottom)):
            padded += 1
            relative = relative.with_suffix(".png")
            original = read_image(row["image"])
            h, w = original.shape[:2]
            actual = read_image(root / row["split"] / "images" / relative)
            if not np.array_equal(original, actual[top:top+h, left:left+w]):
                raise ValueError("Pixel nguồn bị đổi khi đệm")
            q = (q * [w, h] + [left, top]) / [w + left + right, h + top + bottom]
        elif sha256(root / row["split"] / "images" / relative) != row["image_sha256"]:
            raise ValueError("Ảnh sao chép không khớp nguồn")
        label = root / row["split"] / "labels" / relative.with_suffix(".txt")
        lines = [line.split() for line in label.read_text().splitlines() if line.strip()]
        if any(line[0] != "0" or len(line) != 9 for line in lines):
            raise ValueError("Nhãn cuối chưa đúng OBB một lớp")
        actual = np.array([[float(v) for v in line[1:]] for line in lines]).reshape(-1, 4, 2)
        if actual.shape != q.shape or not np.allclose(actual, q.clip(0, 1), rtol=0, atol=1e-7):
            raise ValueError("Nhãn dựng không khớp phép chuyển đã duyệt")
    if len(report["records"]) != len(source["records"]):
        raise ValueError("Có ảnh thừa hoặc thiếu trong bản dựng")
    result = {"ready": True, "dataset_signature": dataset_signature(root),
              "counts": report["counts"], "images": len(report["records"]),
              "boxes": sum(len(r["boxes"]) for r in report["records"]), "padded_images": padded,
              "source_pixels_verified": True, "labels_verified": True,
              "split_review_sha256": sha256(REVIEW), "identity_leakage_certified": False,
              "note": "Đã xử lý nhóm trùng đã xác nhận; chưa chứng nhận mọi danh tính khác tên/góc chụp."}
    save_json(root / "normalization_check.json", result)
    save_json(ROOT / "runs/plate_detection/prepared_audit.json", report)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("audit", "prepare", "check"))
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    cfg = load_config()
    if args.action == "prepare":
        print(prepare_dataset(cfg, args.source))
    elif args.action == "check":
        print(json.dumps(check_prepared(cfg), ensure_ascii=False, indent=2))
    else:
        report = audit_dataset(args.source or cfg["raw_dataset"])
        if not args.source:
            from .quality import apply_split_review
            report = apply_split_review(report)
        output = "prepared_audit.json" if Path(report["source"]) == Path(cfg["dataset"]) else "label_audit.json"
        save_json(ROOT / "runs/plate_detection" / output, report)
        print(json.dumps({k: v for k, v in report.items() if k != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

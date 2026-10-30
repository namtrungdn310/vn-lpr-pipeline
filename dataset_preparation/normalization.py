"""Kiểm nguồn, chuẩn hoá nhãn và đề xuất box để duyệt."""

from collections import defaultdict
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import unicodedata

import cv2
import numpy as np
from PIL import Image

from .dataset import CHARACTERS, Dataset, validate_text
from .geometry import rectify_crop, validate_quad
from .io import sha256_file
from .line_envelopes import find_row_overlaps, propose_line
from .text_bounds import glyph_quad


def normalize_text(text: str) -> str:
    """Đề xuất bỏ dấu phân cách, giữ nguyên chữ số và số 0 đầu."""
    text = unicodedata.normalize("NFC", text).strip().upper()
    return "".join(char for char in text if char not in " -.\t\r\n")


def read_source(workspace: Path, input_root: Path) -> Dataset:
    """Đọc nhãn ban đầu, bỏ đúng các ảnh đã được duyệt cách ly trước đó."""
    import json

    dataset = Dataset.read(input_root, strict_text=False)
    history = json.loads((Path(workspace) / "metadata/provenance.json").read_text(encoding="utf-8"))
    removed = set(history["quarantined_files"])
    return Dataset(
        dataset.root,
        {p: rows for p, rows in dataset.det.items() if p not in removed},
        {p: text for p, text in dataset.rec.items() if p not in removed},
    )


def audit_source(dataset: Dataset) -> dict:
    """Cờ mờ/chói và ảnh gần giống chỉ dùng để xem lại, không tự xoá."""
    metrics, label_changes, findings = [], [], []
    hashes, pixels, plates = defaultdict(list), defaultdict(list), defaultdict(list)
    dhashes = []
    annotations = list(dataset.rec.items()) + [
        (f"{name}#{index}", obj["transcription"])
        for name, objects in dataset.det.items()
        for index, obj in enumerate(objects)
    ]
    for name, text in annotations:
        normalized = normalize_text(text)
        if normalized != text:
            label_changes.append({"sample": name, "before": text, "proposed": normalized})
        if not normalized or set(normalized) - set(CHARACTERS):
            findings.append({"sample": name, "reason": "invalid_text"})
    for name in sorted(set(dataset.det) | set(dataset.rec)):
        path = dataset.root / name
        with Image.open(path) as image:
            image.load()
            rgb = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        metrics.append(
            {
                "image": name,
                "split": name.split("/")[0],
                "branch": name.split("/")[1],
                "width": rgb.shape[1],
                "height": rgb.shape[0],
                "focus": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                "contrast": float(gray.std()),
                "bright_fraction": float((gray >= 250).mean()),
                "dark_fraction": float((gray <= 5).mean()),
            }
        )
        hashes[sha256_file(path)].append(name)
        pixels[(rgb.shape, rgb.tobytes())].append(name)
        if name in dataset.det:
            small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
            bits = np.packbits(small[:, 1:] > small[:, :-1]).tobytes()
            dhashes.append((name, int.from_bytes(bits, "big")))
            objects = dataset.det[name]
            for index, obj in enumerate(objects):
                try:
                    validate_quad(obj["points"], rgb.shape[1], rgb.shape[0])
                except ValueError as error:
                    findings.append({"sample": f"{name}#{index}", "reason": str(error)})
            plates["".join(normalize_text(obj["transcription"]) for obj in objects)].append(name)
    thresholds = {}
    for branch in ("det", "rec"):
        train = [row for row in metrics if row["split"] == "train" and row["branch"] == branch]
        if train:
            thresholds[branch] = {
                "focus_low": float(np.quantile([row["focus"] for row in train], 0.05)),
                "contrast_low": float(np.quantile([row["contrast"] for row in train], 0.05)),
            }
    for row in metrics:
        limits = thresholds.get(row["branch"], {})
        row["review_flags"] = []
        if row["focus"] < limits.get("focus_low", 0):
            row["review_flags"].append("low_focus")
        if row["contrast"] < limits.get("contrast_low", 0):
            row["review_flags"].append("low_contrast")
        if row["bright_fraction"] > 0.4:
            row["review_flags"].append("bright_source")
    near = []
    for (first, a), (second, b) in combinations(dhashes, 2):
        if first.split("/")[0] == second.split("/")[0]:
            continue
        distance = (a ^ b).bit_count()
        if distance <= 8:
            near.append({"first": first, "second": second, "distance": distance})
    cross_split = lambda groups: [
        names for names in groups.values() if len({name.split("/")[0] for name in names}) > 1
    ]
    return {
        "images": len(metrics),
        "image_metrics": metrics,
        "train_thresholds": thresholds,
        "text_proposals": label_changes,
        "findings": findings,
        "exact_file_groups": [names for names in hashes.values() if len(names) > 1],
        "exact_cross_split_pixels": cross_split(pixels),
        "full_plate_cross_split": cross_split(plates),
        "near_cross_split_candidates": sorted(near, key=lambda row: row["distance"]),
        "quality_flagged_images": sum(bool(row["review_flags"]) for row in metrics),
    }


@lru_cache(maxsize=1)
def db_detector():
    """DB cũ chỉ hỗ trợ tìm nét chữ; không sửa nhãn bằng nhận dạng OCR."""
    import rapidocr_onnxruntime
    from rapidocr_onnxruntime.ch_ppocr_v3_det.text_detect import TextDetector

    model = Path(rapidocr_onnxruntime.__file__).parent / "models/ch_PP-OCRv3_det_infer.onnx"
    return TextDetector(
        {
            "model_path": str(model),
            "use_cuda": False,
            "limit_side_len": 768,
            "limit_type": "max",
            "unclip_ratio": 1.2,
            "box_thresh": 0.4,
            "intra_op_num_threads": 2,
            "inter_op_num_threads": 1,
        }
    )


def db_support(rgb):
    """Bản đồ xác suất chữ từ DB, dùng chọn thành phần nét trong glyph_quad."""
    from rapidocr_onnxruntime.ch_ppocr_v3_det.utils import transform

    detector = db_detector()
    height, width = rgb.shape[:2]
    scale = min(5.0, 640 / max(height, width))
    enlarged = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    padded = cv2.copyMakeBorder(
        enlarged, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=(180, 180, 180)
    )
    tensor, _ = transform(
        {"image": cv2.cvtColor(padded, cv2.COLOR_RGB2BGR)}, detector.preprocess_op
    )
    prediction = detector.infer(tensor[None].astype(np.float32))[0][0, 0]
    probability = cv2.resize(prediction, (padded.shape[1], padded.shape[0]))
    probability = probability[40 : 40 + enlarged.shape[0], 40 : 40 + enlarged.shape[1]]
    return cv2.resize(probability, (width, height))


def map_crop_points(points, crop_size, inner):
    width, height = crop_size
    grid = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    matrix = cv2.getPerspectiveTransform(grid, np.float32(points))
    return cv2.perspectiveTransform(np.float32([inner]), matrix)[0]


def propose_glyph_line(image: Image.Image, points: list) -> dict:
    """Nhánh siết bốn cạnh cũ: mở ROI, DB + nét tối, rồi map về ảnh gốc."""
    seed = rectify_crop(image, points, preserve_text_order=True)
    width, height = seed.size
    margin = max(2.0, height * 0.08)
    expanded = map_crop_points(
        points,
        seed.size,
        [
            [-margin, -margin],
            [width - 1 + margin, -margin],
            [width - 1 + margin, height - 1 + margin],
            [-margin, height - 1 + margin],
        ],
    )
    expanded = np.clip(expanded, [0, 0], [image.width - 1, image.height - 1]).tolist()
    analysis = rectify_crop(image, expanded, preserve_text_order=True)
    rgb = np.asarray(analysis)
    result = glyph_quad(rgb, db_support(rgb))
    if "inner" in result:
        native = map_crop_points(expanded, analysis.size, result["inner"])
        result["points"] = np.clip(native, [0, 0], [image.width - 1, image.height - 1]).tolist()
    return result


def propose_boxes(dataset: Dataset, *, method="native", image_names=None) -> list[dict]:
    """Đề xuất lại box từ ảnh + khung ban đầu; mọi đề xuất đều cần duyệt."""
    if method not in ("native", "db"):
        raise ValueError("method phải là native hoặc db")
    selected = set(image_names) if image_names is not None else set(dataset.det)
    if selected - set(dataset.det):
        raise ValueError("Ảnh đề xuất không có trong nhãn DET")
    proposals = []
    for name, objects in dataset.det.items():
        if name not in selected:
            continue
        with Image.open(dataset.root / name) as opened:
            image = opened.convert("RGB")
        for index, obj in enumerate(objects):
            row = {
                "det_image": name,
                "object_index": index,
                "text": normalize_text(obj["transcription"]),
                "input_points": obj["points"],
                "proposal_points": None,
                "review_required": True,
            }
            try:
                seed = validate_quad(obj["points"], image.width, image.height)
                neighbours = [other["points"] for j, other in enumerate(objects) if j != index]
                if method == "native":
                    result = propose_line(np.asarray(image), seed, neighbours, row["text"])
                else:
                    result = propose_glyph_line(image, seed)
                if "points" in result:
                    row["proposal_points"] = validate_quad(
                        result["points"], image.width, image.height
                    )
                row["diagnostics"] = result
            except (ValueError, cv2.error) as error:
                row["diagnostics"] = {"status": "uncertain", "reason": str(error)}
            proposals.append(row)
    valid_rows = []
    for row in proposals:
        with Image.open(dataset.root / row["det_image"]) as image:
            try:
                points = validate_quad(
                    row["proposal_points"] or row["input_points"], image.width, image.height
                )
                valid_rows.append({**row, "points": points})
            except ValueError:
                pass
    conflicts = find_row_overlaps(valid_rows)
    conflicted = {(r["det_image"], index) for r in conflicts for index in r["object_indices"]}
    for row in proposals:
        if (row["det_image"], row["object_index"]) in conflicted:
            row["diagnostics"]["status"] = "uncertain"
            row["diagnostics"]["reason"] = "proposed_rows_overlap"
    return proposals


def review_boxes(dataset: Dataset, proposals: list[dict], decisions: dict) -> list[dict]:
    """Duyệt từng box: keep, proposal hoặc manual; phải có lý do rõ ràng."""
    expected = {(row["det_image"], row["object_index"]) for row in proposals}
    if len(expected) != len(proposals) or set(decisions) != expected:
        raise ValueError("Cần quyết định cho từng đề xuất; không tự chấp nhận box")
    approved = []
    for row in proposals:
        key = row["det_image"], row["object_index"]
        decision = decisions[key]
        if not str(decision.get("reason", "")).strip():
            raise ValueError("Quyết định cần có lý do")
        action = decision.get("action")
        if action == "keep":
            points = row["input_points"]
        elif action == "proposal":
            points = row["proposal_points"]
            if points is None:
                raise ValueError("Đề xuất không có box hợp lệ")
        elif action == "manual":
            points = decision["points"]
        else:
            raise ValueError("action phải là keep, proposal hoặc manual")
        with Image.open(dataset.root / row["det_image"]) as image:
            points = validate_quad(points, image.width, image.height)
        text = decision.get("text", row["text"])
        validate_text(text)
        approved.append(
            {
                "det_image": key[0],
                "object_index": key[1],
                "points": points,
                "text": text,
                "reason": decision["reason"],
            }
        )
    if find_row_overlaps(approved):
        raise ValueError("Box đã chọn còn giao nhau giữa các tầng")
    return approved

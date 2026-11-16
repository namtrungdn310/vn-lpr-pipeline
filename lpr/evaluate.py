"""Đo toàn luồng trên ảnh toàn cảnh; hộp GT chỉ dùng khi chấm điểm."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
import random
import time

import cv2
import numpy as np

from ocr.config import ROOT, save_json
from ocr.metrics import edit_distance, normalize, speed_scores
from ocr.paddle_suite_state import BenchmarkState, model_lock
from plate_detection.data import EXTENSIONS, audit_dataset, load_config, ordered_quad, read_image, sha256
from .pipeline import load_detector, load_ocr, make_pipeline, ocr_names

MANIFEST = ROOT / "data/e2e/plates.jsonl"
OUTPUT = ROOT / "runs/end_to_end"


def output_dir(cfg):
    return OUTPUT / Path(cfg["run_dir"]).name


def annotation_template(cfg, destination=MANIFEST):
    """Tạo bản cần duyệt; không dùng OCR dự đoán làm nhãn chuẩn."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Giữ nguyên nhãn đã tạo; chọn file khác")
    report = audit_dataset(cfg["dataset"])
    if not report["ready"]:
        raise ValueError("Cần dataset OBB hợp lệ trước")
    rows = []
    for record in report["records"]:
        if record["split"] == "train":
            continue
        image = read_image(record["image"])
        rows.append({"image": Path(record["image"]).relative_to(cfg["dataset"]).as_posix(),
                     "split": record["split"], "reviewed": False, "identity_group": "",
                     "plates": [{"quad": (np.array(q) * [image.shape[1], image.shape[0]]).tolist(), "text": ""}
                                for q in record["boxes"]]})
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return destination


def load_manifest(path, dataset):
    root = Path(dataset).resolve()
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    seen, hashes, groups, texts = set(), {}, {}, {}
    for row in rows:
        image = (root / row["image"]).resolve()
        split = row["split"]
        if split not in ("val", "test") or not image.is_relative_to(root / split / "images"):
            raise ValueError("Ảnh phải thuộc đúng val/test của detector; không dùng train để test")
        if (image in seen or row.get("reviewed") is not True
                or not isinstance(row.get("identity_group"), str) or not row["identity_group"].strip()):
            raise ValueError("Nhãn trùng hoặc chưa duyệt/text/nhóm cùng biển chưa được xác nhận")
        seen.add(image)
        pixels = read_image(image)
        h = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
        for table, key in ((hashes, h), (groups, row["identity_group"])):
            if key in table and table[key] != split:
                raise ValueError("Ảnh/biển trùng giữa val và test")
            table[key] = split
        for plate in row["plates"]:
            q = np.asarray(plate["quad"], dtype=np.float32)
            ordered_quad(q)
            if not cv2.isContourConvex(q) or (q < 0).any() or (q > [pixels.shape[1], pixels.shape[0]]).any():
                raise ValueError("Quad GT không hợp lệ")
            if not isinstance(plate.get("text"), str):
                raise ValueError("Text GT phải là chuỗi đã đọc và duyệt")
            plate["text"] = normalize(plate["text"])
            if not plate["text"]:
                raise ValueError("Phải điền nội dung biển GT; nhãn lớp Roboflow không phải nhãn OCR")
            if plate["text"] in texts and texts[plate["text"]] != split:
                raise ValueError("Cùng nội dung biển xuất hiện ở val và test")
            texts[plate["text"]] = split
        row["path"] = str(image)
        row["sha256"] = sha256(image)
    if not all(any(r["split"] == split and r["plates"] for r in rows) for split in ("val", "test")):
        raise ValueError("Cần val và test có ít nhất một biển đã gán text")
    expected = {p.resolve() for split in ("val", "test") for p in (root / split / "images").rglob("*")
                if p.suffix.lower() in EXTENSIONS}
    if seen != expected:
        raise ValueError("Manifest phải đủ toàn bộ ảnh val/test; không bỏ riêng ảnh khó")
    return rows


def check_ocr_overlap(rows):
    """Chặn trùng nội dung đã biết với tập OCR; nguồn public chưa chứng nhận độc lập."""
    from ocr.data import read_split

    dataset = ROOT / "vn-plate-dataset_v2"
    train = {r["text"] for r in read_split(dataset, "train", "det")}
    val = {r["text"] for r in read_split(dataset, "val", "det")}
    overlap = [(r["image"], p["text"]) for r in rows for p in r["plates"]
               if p["text"] in (train | val if r["split"] == "test" else train)]
    if overlap:
        raise ValueError(f"{len(overlap)} biển trùng nội dung train/val OCR; cần chia tập theo danh tính trước khi test")


def signature(rows):
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def code_signature():
    files = [ROOT / folder / file for folder, names in {
        "lpr": ("pipeline.py", "evaluate.py"),
        "plate_detection": ("data.py", "runtime.py"),
        "ocr": ("metrics.py", "paddle_suite_state.py")}.items() for file in names]
    return hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()


def polygon_iou(a, b):
    a, b = ordered_quad(a), ordered_quad(b)
    intersection = cv2.intersectConvexConvex(a, b)[0]
    return float(intersection / max(1e-9, cv2.contourArea(a) + cv2.contourArea(b) - intersection))


def score_predictions(predictions, rows, threshold=0.5):
    """Ghép hình học một-một; không chọn hộp theo text đúng với GT."""
    from scipy.optimize import linear_sum_assignment

    correct = errors = characters = total = tp = fp = fn = 0
    for predicted, row in zip(predictions, rows, strict=True):
        actual = row["plates"]
        matrix = np.array([[polygon_iou(g["quad"], p["quad"]) for p in predicted] for g in actual])
        matched = {}
        if len(actual) and len(predicted):
            # Ưu tiên nhiều cặp đạt IoU; sau đó tối đa tổng IoU.
            utility = (matrix >= threshold).astype(float) * (max(matrix.shape) + 1) + matrix
            gi, pi = linear_sum_assignment(-utility)
            matched = {int(g): int(p) for g, p in zip(gi, pi) if matrix[g, p] >= threshold}
        tp += len(matched)
        fn += len(actual) - len(matched)
        fp += len(predicted) - len(matched)
        total += len(actual)
        for g, plate in enumerate(actual):
            text = normalize(predicted[matched[g]]["text"]) if g in matched else ""
            reference = normalize(plate["text"])
            correct += int(text == reference)
            errors += edit_distance(reference, text)
            characters += len(reference)
        # Biển phát hiện dư tạo ký tự chèn; không bỏ qua false positive.
        errors += sum(len(normalize(p["text"])) for i, p in enumerate(predicted) if i not in matched.values())
    if not total:
        raise ValueError("Không có biển GT")
    return {"ACC": 100 * correct / total, "CER": 100 * errors / characters,
            "samples": total, "correct": correct, "reference_characters": characters,
            "character_errors": errors, "TP": tp, "FP": fp, "FN": fn}


def freeze(cfg, name, manifest=MANIFEST):
    rows = load_manifest(manifest, cfg["dataset"])
    check_ocr_overlap(rows)
    val = [r for r in rows if r["split"] == "val"]
    pipe, artifacts = make_pipeline(name, cfg)
    predictions = [pipe(read_image(r["path"])) for r in val]
    frozen = {"config": cfg, "artifacts": artifacts, "val_signature": signature(val),
              "code_signature": code_signature(), "OCR": name,
              "selection": "Fixed detector/OCR defaults; chỉ kiểm val, không tìm ngưỡng trên test",
              "val_metrics": score_predictions(predictions, val, cfg["benchmark"]["match_iou"])}
    save_json(output_dir(cfg) / name / "frozen.json", frozen)
    return frozen["val_metrics"]


def benchmark(cfg, name, manifest=MANIFEST):
    import onnxruntime as ort

    directory = output_dir(cfg) / name
    with model_lock(directory):
        rows = load_manifest(manifest, cfg["dataset"])
        check_ocr_overlap(rows)
        test, val = ([r for r in rows if r["split"] == split] for split in ("test", "val"))
        frozen = json.loads((directory / "frozen.json").read_text(encoding="utf-8"))
        pipe, artifacts = make_pipeline(name, cfg)
        if (frozen["config"] != cfg or frozen["artifacts"] != artifacts
                or frozen["val_signature"] != signature(val) or frozen["code_signature"] != code_signature()):
            raise ValueError("Model/config/val/code đã đổi sau bước chốt val")
        params = cfg["benchmark"]
        if min(params["warmup"], params["repeats"], cfg["inference"]["threads"]) < 1:
            raise ValueError("Số lượt đo và số luồng phải dương")
        protocol = {"OCR": name, "artifacts": artifacts, "test_signature": signature(test),
                    "val_signature": signature(val), "code_signature": code_signature(), "config": cfg,
                    "runtime": ort.__version__, "platform": platform.platform(), "processor": platform.processor()}
        state = BenchmarkState(directory / "progress.json", protocol, len(test), params["repeats"])
        pixels = [read_image(r["path"]) for r in test]
        cv2.setNumThreads(1)
        if state.completed < len(test) * params["repeats"]:
            warm = read_image(val[0]["path"])
            for _ in range(params["warmup"]):
                pipe(warm)
        for repeat in range(params["repeats"]):
            order = list(range(len(test)))
            random.Random(cfg["seed"] + repeat).shuffle(order)
            for index in order:
                if len(state.times[index]) > repeat:
                    continue
                start = time.perf_counter_ns()
                predicted = pipe(pixels[index])
                elapsed = (time.perf_counter_ns() - start) / 1e6
                state.record(index, repeat, json.dumps(predicted, sort_keys=True), elapsed)
            print(f"YOLO26 + {name}: {repeat + 1}/{params['repeats']}", flush=True)
        predictions = [json.loads(p) for p in state.predictions]
        result = {"Model": name, **score_predictions(predictions, test, params["match_iou"]),
                  **speed_scores([t for times in state.times for t in times]),
                  "frames": len(test), "batch_unit": "frame", "timing_scope": "frame -> YOLO + crop + OCR; excludes disk IO/model load",
                  "threads": cfg["inference"]["threads"], "opencv_threads": 1,
                  "precision": "FP32", "provider": "CPUExecutionProvider", "protocol": protocol}
        save_json(directory / "test_predictions.json", [
            {"image": r["image"], "GT": r["plates"], "prediction": p, "times_ms": t}
            for r, p, t in zip(test, predictions, state.times)])
        save_json(directory / "test_result.json", result)
        return result


def table(names, cfg, manifest=MANIFEST):
    from ocr.comparison import markdown_table, save_table

    rows = load_manifest(manifest, cfg["dataset"])
    check_ocr_overlap(rows)
    test = [r for r in rows if r["split"] == "test"]
    val = [r for r in rows if r["split"] == "val"]
    _, det_state = load_detector(cfg, load_runtime=False)
    results = []
    common = None
    for name in names:
        directory = output_dir(cfg) / name
        result = json.loads((directory / "test_result.json").read_text(encoding="utf-8"))
        protocol = result["protocol"]
        _, ocr_state = load_ocr(name, cfg["inference"]["threads"], load_runtime=False)
        if (protocol["test_signature"] != signature(test) or protocol["code_signature"] != code_signature()
                or protocol["config"] != cfg or protocol["OCR"] != name
                or protocol["val_signature"] != signature(val)
                or protocol["artifacts"] != {"detector": det_state, "ocr": ocr_state}):
            raise ValueError("Kết quả test toàn luồng đã cũ hoặc khác điều kiện")
        conditions = {k: protocol[k] for k in ("test_signature", "val_signature", "config", "runtime", "platform", "processor")}
        if common is not None and common != conditions:
            raise ValueError("Không thể ghép các điều kiện đo khác nhau")
        common = conditions
        details = json.loads((directory / "test_predictions.json").read_text(encoding="utf-8"))
        if [(d["image"], d["GT"]) for d in details] != [(r["image"], r["plates"]) for r in test]:
            raise ValueError("Chi tiết không khớp test")
        if any(len(d["times_ms"]) != cfg["benchmark"]["repeats"] for d in details):
            raise ValueError("Chưa đủ lượt đo")
        scores = score_predictions([d["prediction"] for d in details], test, cfg["benchmark"]["match_iou"])
        scores.update(speed_scores([t for d in details for t in d["times_ms"]]))
        if any(result[k] != v for k, v in scores.items()):
            raise ValueError("Chỉ số không khớp số đo gốc")
        results.append(result)
    if not results:
        raise ValueError("Chưa có model để xếp hạng")
    slowest = max(r["Speed"] for r in results)
    ranked = [{"Rank": i, **{k: r[k] for k in ("Model", "ACC", "CER", "Speed", "FPS")},
               "Speedup": slowest / r["Speed"]}
              for i, r in enumerate(sorted(results, key=lambda r: (-r["ACC"], r["CER"], r["Speed"])), 1)]
    save_table(ranked, output_dir(cfg) / "comparison_ranked.csv")
    return markdown_table(ranked)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("template", "freeze", "test", "table"))
    parser.add_argument("--model", choices=ocr_names(), nargs="+", default=["VietLPR_v2"])
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--variant", choices=tuple(load_config()["variants"]), default="n")
    args = parser.parse_args()
    cfg = load_config(variant=args.variant)
    if args.action == "template":
        print(annotation_template(cfg, args.manifest))
    elif args.action == "table":
        print(table(args.model, cfg, args.manifest))
    else:
        for name in args.model:
            result = freeze(cfg, name, args.manifest) if args.action == "freeze" else benchmark(cfg, name, args.manifest)
            print(json.dumps({k: v for k, v in result.items() if k != "protocol"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

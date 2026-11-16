"""Báo cáo riêng YOLO26n-OBB: mAP, Precision/Recall và tốc độ ONNX CPU."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import cv2
import numpy as np

from ocr.config import ROOT, save_json
from ocr.metrics import speed_scores
from ocr.paddle_suite_state import BenchmarkState, model_lock
from .data import EXTENSIONS, dataset_signature, load_config, ordered_quad, read_image, sha256
from .runtime import PlateDetector
from .train import ultralytics_api


def code_signature():
    return hashlib.sha256(b"".join((Path(__file__).parent / name).read_bytes()
                                  for name in ("evaluate.py", "runtime.py", "data.py", "quality.py"))).hexdigest()


def split_rows(cfg, split):
    root = Path(cfg["dataset"]) / split
    rows = []
    for image in sorted(p for p in (root / "images").rglob("*") if p.suffix.lower() in EXTENSIONS):
        label = root / "labels" / image.relative_to(root / "images").with_suffix(".txt")
        rows.append({"image": str(image), "label": str(label), "image_sha256": sha256(image),
                     "label_sha256": sha256(label)})
    if not rows:
        raise ValueError("Tập đo không có ảnh")
    return rows


def operating_scores(predictions, ground_truth, threshold=0.5):
    """P/R tại confidence cố định, ghép hộp bằng polygon IoU thật."""
    from scipy.optimize import linear_sum_assignment

    tp = fp = fn = 0
    for detections, actual in zip(predictions, ground_truth, strict=True):
        matched = 0
        if detections and actual:
            matrix = np.zeros((len(actual), len(detections)))
            for i, a in enumerate(actual):
                a = ordered_quad(a)
                for j, b in enumerate(detections):
                    b = ordered_quad(b["quad"])
                    area = cv2.intersectConvexConvex(a, b)[0]
                    matrix[i, j] = area / max(1e-9, cv2.contourArea(a) + cv2.contourArea(b) - area)
            utility = (matrix >= threshold) * (max(matrix.shape) + 1) + matrix
            a, b = linear_sum_assignment(-utility)
            matched = int((matrix[a, b] >= threshold).sum())
        tp += matched
        fp += len(detections) - matched
        fn += len(actual) - matched
    return {"Precision": 100 * tp / max(1, tp + fp), "Recall": 100 * tp / max(1, tp + fn),
            "TP": tp, "FP": fp, "FN": fn}


def benchmark(cfg):
    import onnxruntime as ort

    run = Path(cfg["run_dir"])
    with model_lock(run):
        verified = json.loads((run / "onnx/verified.json").read_text(encoding="utf-8"))
        if (verified["config"] != cfg or verified["onnx_sha256"] != sha256(run / "onnx/model.onnx")
                or verified["dataset_signature"] != dataset_signature(cfg["dataset"])):
            raise ValueError("Model/dataset khác bản đã export và kiểm val")
        test, val = split_rows(cfg, "test"), split_rows(cfg, "val")
        protocol = {"config": cfg, "test": test, "val": val, "onnx_sha256": verified["onnx_sha256"],
                    "code_signature": code_signature(), "runtime": ort.__version__,
                    "platform": platform.platform(), "processor": platform.processor()}
        ap_file = run / "onnx_test_ap.json"
        if ap_file.is_file():
            ap = json.loads(ap_file.read_text(encoding="utf-8"))
            if ap["protocol"] != protocol:
                raise ValueError("Lượt test cũ khác model, dữ liệu hoặc code; không ghi đè")
        else:
            # mAP dùng evaluator OBB của nguồn, không tự viết lại cách tính AP.
            YOLO = ultralytics_api(cfg)
            metrics = YOLO(str(run / "onnx/model.onnx"), task="obb").val(
                data=str(Path(cfg["dataset"]) / "data.yaml"), split="test", device="cpu",
                imgsz=cfg["train"]["imgsz"], batch=1, workers=0, rect=False, conf=0.001,
                max_det=300, plots=True, project=str(run), name="onnx_test", exist_ok=True)
            ap = {"protocol": protocol, "mAP50": 100 * float(metrics.box.map50),
                  "mAP50-95": 100 * float(metrics.box.map),
                  "official_precision_at_best_F1": 100 * float(metrics.box.mp),
                  "official_recall_at_best_F1": 100 * float(metrics.box.mr)}
            save_json(ap_file, ap)
        cv2.setNumThreads(1)
        inf, params = cfg["inference"], cfg["benchmark"]
        detector = PlateDetector(run / "onnx/model.onnx", size=cfg["train"]["imgsz"],
                                 confidence=inf["confidence"], max_det=inf["max_det"], threads=inf["threads"])
        state = BenchmarkState(run / "detection_progress.json", protocol, len(test), params["repeats"])
        pixels = [read_image(r["image"]) for r in test]
        if state.completed < len(test) * params["repeats"]:
            warm = read_image(val[0]["image"])
            for _ in range(params["warmup"]):
                detector(warm)
        for repeat in range(params["repeats"]):
            order = list(range(len(test)))
            random.Random(cfg["seed"] + repeat).shuffle(order)
            for index in order:
                if len(state.times[index]) > repeat:
                    continue
                start = time.perf_counter_ns()
                detections = detector(pixels[index])
                elapsed = (time.perf_counter_ns() - start) / 1e6
                state.record(index, repeat, json.dumps(detections, sort_keys=True), elapsed)
            print(f"{cfg['model']}: CPU test {repeat + 1}/{params['repeats']}", flush=True)
        predictions = [json.loads(p) for p in state.predictions]
        actual = []
        for row, image in zip(test, pixels):
            h, w = image.shape[:2]
            actual.append([np.array(line.split()[1:], dtype=float).reshape(4, 2).tolist()
                           for line in Path(row["label"]).read_text().splitlines() if line.strip()])
            actual[-1] = [(np.array(q) * [w, h]).tolist() for q in actual[-1]]
        scores = operating_scores(predictions, actual, params["match_iou"])
        result = {"Model": Path(cfg["model"]).stem, **{k: ap[k] for k in ("mAP50", "mAP50-95")},
                  **scores, **speed_scores([t for times in state.times for t in times]),
                  "images": len(test), "protocol": protocol, "timing_scope": "letterbox + ONNX OBB + decode",
                  "precision": "FP32", "provider": "CPUExecutionProvider", "opencv_threads": 1}
        save_json(run / "detection_predictions.json", [
            {"image": r["image"], "GT": gt, "prediction": p, "times_ms": times}
            for r, gt, p, times in zip(test, actual, predictions, state.times)])
        save_json(run / "detection_result.json", result)
        return result


def comparison(allow_partial=False):
    results, missing, common = [], [], None
    for variant in load_config()["variants"]:
        cfg = load_config(variant=variant)
        run = Path(cfg["run_dir"])
        path = run / "detection_result.json"
        if not path.is_file():
            missing.append(variant)
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        protocol = r["protocol"]
        if (protocol["config"] != cfg or protocol["code_signature"] != code_signature()
                or protocol["onnx_sha256"] != sha256(run / "onnx/model.onnx")
                or protocol["test"] != split_rows(cfg, "test") or protocol["val"] != split_rows(cfg, "val")):
            raise ValueError("Bảng detection chứa kết quả cũ hoặc khác cấu hình")
        conditions = {k: protocol[k] for k in ("test", "val", "runtime", "platform", "processor")}
        conditions["benchmark"] = cfg["benchmark"]
        conditions["inference"] = cfg["inference"]
        conditions["imgsz"] = cfg["train"]["imgsz"]
        if common is not None and common != conditions:
            raise ValueError("Các detector khác điều kiện đo")
        common = conditions
        details = json.loads((run / "detection_predictions.json").read_text(encoding="utf-8"))
        if [d["image"] for d in details] != [d["image"] for d in protocol["test"]]:
            raise ValueError("Chi tiết test không khớp")
        if any(len(d["times_ms"]) != cfg["benchmark"]["repeats"] for d in details):
            raise ValueError("Số lượt đo chưa đủ")
        scores = operating_scores([d["prediction"] for d in details], [d["GT"] for d in details], cfg["benchmark"]["match_iou"])
        scores.update(speed_scores([t for d in details for t in d["times_ms"]]))
        if any(r[k] != v for k, v in scores.items()):
            raise ValueError("Kết quả khác số đo gốc")
        ap = json.loads((run / "onnx_test_ap.json").read_text(encoding="utf-8"))
        if ap["protocol"] != protocol or any(r[k] != ap[k] for k in ("mAP50", "mAP50-95")):
            raise ValueError("mAP không khớp model đã chấm")
        results.append(r)
    if missing and not allow_partial:
        raise FileNotFoundError("Chưa train/đo xong: " + ", ".join(missing))
    if not results:
        return [], "Chưa có kết quả detection chính thức."
    slowest = max(r["Speed"] for r in results)
    keys = ("Model", "mAP50", "mAP50-95", "Precision", "Recall", "Speed", "FPS")
    rows = [{"Rank": i, **{k: r[k] for k in keys}, "Speedup": slowest / r["Speed"]}
            for i, r in enumerate(sorted(results, key=lambda x: (-x["mAP50-95"], -x["Recall"], x["Speed"])), 1)]
    target = ROOT / "runs/plate_detection/comparison.csv"
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("Rank", *keys, "Speedup"))
        writer.writeheader()
        writer.writerows(rows)
    header = "| Rank | Model | mAP50 (%) | mAP50-95 (%) | Precision (%) | Recall (%) | Speed (ms/ảnh) | FPS | Speedup |"
    text = header + "\n|---:|---|---:|---:|---:|---:|---:|---:|---:|\n"
    text += "\n".join("| " + " | ".join(str(r[k]) if k in ("Rank", "Model") else f"{r[k]:.2f}"
                                     for k in ("Rank", *keys, "Speedup")) + " |" for r in rows)
    if missing:
        text += "\n\nBảng tạm, chưa có: " + ", ".join(missing)
    (target.with_suffix(".md")).write_text(text + "\n", encoding="utf-8")
    return rows, text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("test", "table"))
    parser.add_argument("--variant", choices=tuple(load_config()["variants"]), default="n")
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    if args.action == "test":
        result = benchmark(load_config(variant=args.variant))
        print(json.dumps({k: v for k, v in result.items() if k != "protocol"}, ensure_ascii=False, indent=2))
    else:
        print(comparison(args.partial)[1])

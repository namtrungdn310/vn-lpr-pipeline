"""Chọn hậu xử lý trên val, khoá model, rồi so ACC/CER/Speed/FPS trên CPU."""

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import cv2

from .config import NAMES, ROOT, load_config, save_json
from .data import image, read_split, split_signature
from .metrics import speed_scores, text_scores
from .runtime import onnx_pipeline

LABELS = {"paddleocr": "PaddleOCR", "vietlpr_v1": "VietLPR_v1", "vietlpr_v2": "VietLPR_v2"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def deployed_config(name, run_dir=None):
    root_cfg = load_config(name)
    run = Path(run_dir or root_cfg["run_dir"])
    path = run / "onnx/config.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg["name"] != name:
        raise ValueError("Tên model không khớp thư mục ONNX")
    cfg["run_dir"] = str(run.resolve())
    return cfg


def validate_export(cfg):
    run = Path(cfg["run_dir"])
    check = json.loads((run / "onnx/export_check.json").read_text(encoding="utf-8"))
    if set(check) != {"det", "rec"} or any(row.get("smoke_only", True) for row in check.values()):
        raise ValueError(
            "Model chưa xuất đủ hoặc chỉ là weights chạy thử; không dùng làm kết quả chính"
        )


def tune(cfg, *, search=True):
    """Không mở tập test; ACC là tiêu chí chính, CER phá hoà."""
    validate_export(cfg)
    cv2.setNumThreads(1)
    rows = read_split(cfg["dataset"], "val", "det")
    pixels = [image(row["path"]) for row in rows]
    pipeline = onnx_pipeline(cfg)
    candidates = [copy.deepcopy(cfg["postprocess"])]
    if search:
        for threshold in (0.25, 0.35, 0.45):
            if cfg["name"] == "vietlpr_v1":
                candidates.append({**cfg["postprocess"], "threshold": threshold})
            else:
                for unclip in (1.2, 1.5, 1.8):
                    candidates.append(
                        {**cfg["postprocess"], "threshold": threshold, "unclip": unclip}
                    )
    best, selected, records = (-1.0, -float("inf")), None, []
    for params in candidates:
        pipeline.cfg["postprocess"] = params
        predictions = [pipeline(p)["text"] for p in pixels]
        metrics = text_scores(predictions, [row["text"] for row in rows])
        records.append({"postprocess": params, **metrics})
        score = metrics["ACC"], -metrics["CER"]
        if score > best:
            best, selected = score, copy.deepcopy(params)
    cfg["postprocess"] = selected
    run = Path(cfg["run_dir"])
    save_json(run / "onnx/config.json", cfg)
    save_json(
        run / "val_selection.json",
        {
            "split": "val",
            "signature": split_signature(rows),
            "selected": selected,
            "trials": records,
        },
    )
    files = {f"{task}.onnx": digest(run / "onnx" / f"{task}.onnx") for task in ("det", "rec")}
    files["config.json"] = digest(run / "onnx/config.json")
    save_json(
        run / "onnx/frozen.json",
        {
            "files": files,
            "val_signature": split_signature(rows),
            "selection": {"ACC": best[0], "CER": -best[1]},
        },
    )
    return {"ACC": best[0], "CER": -best[1], "postprocess": selected}


def benchmark(cfg):
    import onnxruntime as ort

    validate_export(cfg)
    run = Path(cfg["run_dir"])
    frozen = json.loads((run / "onnx/frozen.json").read_text(encoding="utf-8"))
    for name, expected in frozen["files"].items():
        if digest(run / "onnx" / name) != expected:
            raise ValueError("Model hoặc config đã đổi sau khi chốt val; cần chọn lại trên val")
    rows = read_split(cfg["dataset"], "test", "det")
    pixels = [image(row["path"]) for row in rows]
    val = read_split(cfg["dataset"], "val", "det")
    if split_signature(val) != frozen["val_signature"]:
        raise ValueError("Val đã đổi sau khi chốt model")
    cv2.setNumThreads(1)
    pipeline = onnx_pipeline(cfg)
    warm = image(val[0]["path"])
    params = cfg["benchmark"]
    if params["warmup"] < 1 or params["repeats"] < 1:
        raise ValueError("Cần warm-up và ít nhất một lượt đo")
    for _ in range(params["warmup"]):
        pipeline(warm)
    elapsed = [[] for _ in rows]
    predictions = [None] * len(rows)
    for repeat in range(params["repeats"]):
        order = list(range(len(rows)))
        random.Random(cfg["seed"] + repeat).shuffle(order)
        for index in order:
            start = time.perf_counter_ns()
            prediction = pipeline(pixels[index])
            duration = (time.perf_counter_ns() - start) / 1_000_000
            text = prediction["text"]
            if predictions[index] is not None and predictions[index] != text:
                raise RuntimeError("Dự đoán thay đổi giữa các lần chạy cùng model")
            predictions[index] = text
            elapsed[index].append(duration)
    scores = text_scores(predictions, [row["text"] for row in rows])
    scores.update(speed_scores([t for times in elapsed for t in times]))
    result = {
        "Model": LABELS[cfg["name"]],
        **scores,
        "test_signature": split_signature(rows),
        "threads": params["threads"],
        "opencv_threads": 1,
        "batch_size": 1,
        "precision": "FP32",
        "runtime": ort.__version__,
        "provider": "CPUExecutionProvider",
        "warmup": params["warmup"],
        "repeats": params["repeats"],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "timing_scope": "preprocess+DET+postprocess+crop+REC+decode; excludes disk IO and model load",
        "artifact_sha256": frozen["files"],
        "code_signature": hashlib.sha256(
            "".join(
                digest(Path(__file__).parent / name)
                for name in (
                    "runtime.py",
                    "ops_v1.py",
                    "ops_v2.py",
                    "paddle_postprocess.py",
                    "data.py",
                    "metrics.py",
                    "evaluate.py",
                )
            ).encode("ascii")
        ).hexdigest(),
    }
    details = [
        {
            "image": row["relative"],
            "GT": row["text"],
            "prediction": text,
            "correct": row["text"] == text,
            "times_ms": times,
        }
        for row, text, times in zip(rows, predictions, elapsed)
    ]
    save_json(run / "test_predictions.json", details)
    save_json(run / "test_result.json", result)
    return result


def compare(results, destination):
    if len(results) != 3 or {r["Model"] for r in results} != set(LABELS.values()):
        raise ValueError("Bảng chính cần đủ ba hệ thống, không dùng kết quả giả")
    comparable = (
        "test_signature",
        "threads",
        "opencv_threads",
        "batch_size",
        "precision",
        "runtime",
        "provider",
        "platform",
        "processor",
        "repeats",
        "warmup",
        "timing_scope",
        "code_signature",
    )
    for field in comparable:
        if len({str(r[field]) for r in results}) != 1:
            raise ValueError(f"Điều kiện benchmark khác nhau: {field}")
    baseline = next(row["Speed"] for row in results if row["Model"] == "PaddleOCR")
    rows = [
        {key: result[key] for key in ("Model", "ACC", "CER", "Speed", "FPS")}
        | {"Speedup": baseline / result["Speed"]}
        for result in results
    ]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("Model", "ACC", "CER", "Speed", "FPS", "Speedup")
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("tune", "test", "compare"))
    parser.add_argument("--model", choices=NAMES)
    parser.add_argument("--no-search", action="store_true")
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    if args.action == "compare":
        results = [
            json.loads(
                (Path(load_config(name)["run_dir"]) / "test_result.json").read_text(
                    encoding="utf-8"
                )
            )
            for name in NAMES
        ]
        result = compare(results, ROOT / "runs/comparison.csv")
    else:
        if args.model is None:
            parser.error("Cần --model")
        cfg = deployed_config(args.model, args.run_dir)
        result = tune(cfg, search=not args.no_search) if args.action == "tune" else benchmark(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

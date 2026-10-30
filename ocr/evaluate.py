"""Chốt hậu xử lý trên val và đo VietLPR V1/V2 trên CPU."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import cv2

from .config import NAMES, load_config, save_json
from .data import image, read_split, split_signature
from .metrics import speed_scores, text_scores
from .runtime import onnx_pipeline, postprocess_candidates, inference_signature

LABELS = {"vietlpr_v1": "VietLPR_v1", "vietlpr_v2": "VietLPR_v2"}


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
    if cfg.get("training_revision") != load_config(cfg["name"]).get(
        "training_revision"
    ):
        raise ValueError("Model thuộc quy trình cũ; cần train/export lại theo nguồn đã chốt")
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
    candidates = postprocess_candidates(cfg) if search else [copy.deepcopy(cfg["postprocess"])]
    best, selected, records = (-1.0, -float("inf")), None, []
    for params in candidates:
        pipeline.cfg["postprocess"] = params
        predictions = [pipeline(p)["text"] for p in pixels]
        metrics = text_scores(predictions, [row["text"] for row in rows])
        records.append({"postprocess": params, **metrics})
        score = metrics["ACC"], -metrics["CER"]
        if score > best:
            best, selected = score, copy.deepcopy(params)
    if cfg["name"] == "vietlpr_v2" and best[0] <= records[0]["ACC"]:
        best = records[0]["ACC"], -records[0]["CER"]
        selected = copy.deepcopy(records[0]["postprocess"])
    cfg["postprocess"] = selected
    run = Path(cfg["run_dir"])
    save_json(run / "onnx/config.json", cfg)
    if cfg["name"] == "vietlpr_v2":
        from .upstream import v2_config

        save_json(run / "onnx/lpr_config.json", v2_config(cfg))
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
    if cfg["name"] == "vietlpr_v2":
        for name in ("lpr_config.json", "lpr_infer.py"):
            files[name] = digest(run / "onnx" / name)
    save_json(
        run / "onnx/frozen.json",
        {
            "files": files,
            "val_signature": split_signature(rows),
            "inference_signature": inference_signature(),
            "selection": {"ACC": best[0], "CER": -best[1]},
        },
    )
    return {"ACC": best[0], "CER": -best[1], "postprocess": selected}


def code_signature():
    """Khoá cả code suy luận và cách đo của VietLPR."""
    return hashlib.sha256(
        "".join(
            digest(Path(__file__).parent / name)
            for name in ("runtime.py", "ops_v1.py", "upstream.py", "data.py", "metrics.py", "evaluate.py")
        ).encode("ascii")
    ).hexdigest()


def benchmark(cfg):
    import onnxruntime as ort

    validate_export(cfg)
    run = Path(cfg["run_dir"])
    frozen = json.loads((run / "onnx/frozen.json").read_text(encoding="utf-8"))
    if frozen.get("inference_signature") != inference_signature():
        raise ValueError("Code suy luận đã đổi; chạy lại bước chọn val trước khi test")
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
        "batch_unit": "plate",
        "rec_batching": "all_lines" if cfg["name"] == "vietlpr_v2" else "sequential",
        "precision": "FP32",
        "weights_source": cfg.get("weights_source", "dataset_trained"),
        "runtime": ort.__version__,
        "provider": "CPUExecutionProvider",
        "warmup": params["warmup"],
        "repeats": params["repeats"],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "timing_scope": "preprocess+DET+postprocess+crop+REC+decode; excludes disk IO and model load",
        "artifact_sha256": frozen["files"],
        "code_signature": code_signature(),
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("tune", "test"))
    parser.add_argument("--model", choices=NAMES, required=True)
    parser.add_argument("--no-search", action="store_true")
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    cfg = deployed_config(args.model, args.run_dir)
    result = tune(cfg, search=not args.no_search) if args.action == "tune" else benchmark(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

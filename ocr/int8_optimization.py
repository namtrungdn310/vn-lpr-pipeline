"""Thử INT8 bằng train/val; giữ nguyên nguồn và kết quả đã chốt."""

import argparse
from collections import Counter
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
import onnx
import onnxruntime as ort

from . import quantization as old
from .config import ROOT, save_json
from .data import image, read_split, split_signature
from .metrics import text_scores
from .runtime import onnx_pipeline
from .paddle_suite_state import model_lock
from plate_detection import quantization as yq
from plate_detection.data import load_config, read_image, dataset_signature, sha256
from plate_detection.evaluate import split_rows, operating_scores

CONFIG = ROOT / "configs/ocr/int8_optimization.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def settings():
    cfg = read_json(CONFIG)
    cfg["run_dir"] = str((ROOT / cfg["run_dir"]).resolve())
    return cfg


def signature():
    return hashlib.sha256(Path(__file__).read_bytes() + CONFIG.read_bytes()
                          + old.code_signature().encode() + yq.code_signature().encode()).hexdigest()


def execute(*args):
    subprocess.run([sys.executable, "-u", "-B", "-m", "ocr.int8_optimization", *map(str, args)],
                   cwd=ROOT, check=True, env={**os.environ, "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "4"})


def quantize(source, target, tensors, spec):
    """U8 dùng dải bất đối xứng; không thay preprocessing/model đã học."""
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static

    key = onnx.load(source).graph.input[0].name
    class Reader(CalibrationDataReader):
        def __init__(self):
            self.items = (x[i:i + 1] for x in tensors() for i in range(x.shape[0]))

        def get_next(self):
            x = next(self.items, None)
            return {key: x} if x is not None else None

    signed = spec["activation"] == "s8"
    excluded = [n.name for n in onnx.load(source).graph.node if n.op_type == "Conv" and n.name.startswith("/model.23/")] if spec.get("exclude_head") else []
    quantize_static(str(source), str(target), Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8 if signed else QuantType.QUInt8,
                    weight_type=QuantType.QInt8 if spec["weight"] == "s8" else QuantType.QUInt8,
                    per_channel=True, op_types_to_quantize=spec["ops"], nodes_to_exclude=excluded,
                    calibrate_method=CalibrationMethod.MinMax if spec["method"] == "minmax" else CalibrationMethod.Percentile,
                    extra_options={"ActivationSymmetric": signed, "WeightSymmetric": spec["weight"] == "s8",
                                   "CalibTensorRangeSymmetric": signed, "CalibPercentile": 99.99,
                                   "DedicatedQDQPair": spec["dedicated"], "MatMulConstBOnly": True})
    onnx.checker.check_model(str(target))


def optimized_inventory(path):
    """Graph riêng chẩn đoán; không dùng graph theo CPU này làm file triển khai."""
    with tempfile.TemporaryDirectory() as temp:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads, opts.inter_op_num_threads = 4, 1
        opts.log_severity_level = 3
        opts.optimized_model_filepath = str(Path(temp) / "optimized.onnx")
        session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        counts = dict(Counter(n.op_type for n in onnx.load(opts.optimized_model_filepath).graph.node))
        del session
    return counts


def val_measure(pipe, pixels, warm, cfg):
    for _ in range(cfg["val_warmup"]):
        pipe(warm)
    means, reference = [], None
    for repeat in range(cfg["val_rounds"]):
        order = list(range(len(pixels)))
        random.Random(cfg["seed"] + repeat).shuffle(order)
        predictions, times = [None] * len(pixels), []
        for index in order:
            start = time.perf_counter_ns()
            predicted = pipe(pixels[index])
            elapsed = (time.perf_counter_ns() - start) / 1e6
            predictions[index] = predicted
            times.append(elapsed)
        # Chỉ so chuỗi OCR hoặc hộp detector, không so confidence OCR.
        stable = [p["text"] for p in predictions] if isinstance(predictions[0], dict) else predictions
        if reference is not None and stable != reference:
            raise ValueError("Dự đoán val không ổn định")
        reference = stable
        means.append(statistics.mean(times))
    return predictions, {"Speed": statistics.mean(means), "SD": statistics.stdev(means), "round_means_ms": means}


def assess(task, folder):
    """Tiến trình riêng cho mỗi ứng viên; test không được đọc ở bước này."""
    cfg = settings()
    cv2.setNumThreads(cfg["opencv_threads"])
    folder = Path(folder)
    if task == "vietlpr":
        source, _, _ = old.source_model(old.settings())
        rows = read_split(source["dataset"], "val", "det")
        pipe = onnx_pipeline(source, folder)
        pixels = [image(r["path"]) for r in rows]
        predictions, timing = val_measure(pipe, pixels, pixels[0], cfg)
        scores = text_scores([p["text"] for p in predictions], [r["text"] for r in rows])
        graphs = {t: optimized_inventory(folder / f"{t}.onnx") for t in ("det", "rec")}
    else:
        import torch
        torch.set_num_threads(cfg["threads"])
        source = load_config()
        rows = split_rows(source, "val")
        pipe = yq.detector(folder / "model.onnx", cfg, source)
        pixels = [read_image(r["image"]) for r in rows]
        predictions, timing = val_measure(pipe, pixels, pixels[0], cfg)
        scores = operating_scores(predictions, yq.ground_truth(rows, pixels), source["benchmark"]["match_iou"])
        del pipe
        gc.collect()
        metrics = yq.ultralytics_api(source)(str(folder / "model.onnx"), task="obb").val(
            data=str(Path(source["dataset"]) / "data.yaml"), split="val", device="cpu", imgsz=640,
            batch=1, workers=0, rect=False, conf=0.001, max_det=300, plots=False,
            project=str(folder), name="native_val", exist_ok=True)
        scores.update({"mAP50": 100 * float(metrics.box.map50), "mAP50-95": 100 * float(metrics.box.map)})
        graphs = {"model": optimized_inventory(folder / "model.onnx")}
    result = {"task": task, "folder": str(folder), "split": "val", "scores": scores,
              "timing": timing, "optimized_ops": graphs, "signature": signature(),
              "artifacts": {p.name: sha256(p) for p in folder.glob("*.onnx")}}
    save_json(folder / "assessment.json", result)
    print("VAL", task, folder.name, scores, timing, flush=True)


def eligible(task, result, baseline, cfg):
    a, b = result["scores"], baseline["scores"]
    if task == "vietlpr":
        return a["correct"] >= b["correct"] - cfg["ocr_max_correct_loss"] and a["CER"] <= b["CER"] + cfg["ocr_max_cer_increase_pp"]
    return a["mAP50-95"] >= b["mAP50-95"] - cfg["yolo_max_map_loss_pp"] and a["Recall"] >= b["Recall"] - cfg["yolo_max_recall_loss_pp"]


def choose(task, candidates, baseline, cfg):
    accepted = [r for r in candidates if eligible(task, r, baseline, cfg)]
    # Nếu không đạt ngưỡng, vẫn lưu ứng viên tốt nhất để báo cáo, không tự thay FP32.
    if accepted:
        return min(accepted, key=lambda r: r["timing"]["Speed"]), True
    key = (lambda r: (r["scores"]["correct"], -r["scores"]["CER"])) if task == "vietlpr" else (lambda r: (r["scores"]["mAP50-95"], r["scores"]["Recall"]))
    return max(candidates, key=key), False


def prepare_task(task):
    from onnxruntime.quantization.shape_inference import quant_pre_process

    cfg = settings()
    run = Path(cfg["run_dir"]) / task
    run.mkdir(parents=True, exist_ok=True)
    if (run / "selected.json").is_file():
        return validate_selected(task)
    cv2.setNumThreads(1)
    if task == "vietlpr":
        source, original, _ = old.source_model(old.settings())
        samples, recipe = old.calibration(old.settings(), source)
        originals = {t: original / f"{t}.onnx" for t in ("det", "rec")}
        val_sig = split_signature(read_split(source["dataset"], "val", "det"))
    else:
        source = load_config()
        original = Path(source["run_dir"]) / "onnx/model.onnx"
        verified = read_json(original.with_name("verified.json"))
        if verified["onnx_sha256"] != sha256(original) or verified["dataset_signature"] != dataset_signature(source["dataset"]):
            raise ValueError("YOLO khác nguồn đã kiểm")
        rows = yq.calibration_rows(cfg, source)
        samples = {"model": None}
        recipe = {"split": "train", "samples": len(rows), "rows": rows}
        originals = {"model": original}
        val_sig = hashlib.sha256(json.dumps(split_rows(source, "val"), sort_keys=True).encode()).hexdigest()
    protocol = {"signature": signature(), "config": cfg, "source": {k: sha256(p) for k, p in originals.items()},
                "val_signature": val_sig, "runtime": ort.__version__, "selection_split": "val", "calibration_split": "train"}
    if (run / "protocol.json").is_file() and read_json(run / "protocol.json") != protocol:
        raise ValueError("Không trộn thử nghiệm khác code/config/nguồn")
    save_json(run / "protocol.json", protocol)
    save_json(run / "calibration.json", recipe)
    base = run / "fp32"
    base.mkdir(exist_ok=True)
    prepared = {}
    for subtask, path in originals.items():
        shutil.copy2(path, base / path.name)
        pre = run / (subtask + "_preprocessed.onnx")
        quant_pre_process(path, pre, skip_optimization=True, skip_symbolic_shape=True)
        prepared[subtask] = pre
    if not (base / "assessment.json").exists():
        execute("assess", "--task", task, "--folder", base)
    baseline = read_json(base / "assessment.json")
    results = []
    for spec in cfg["candidates"][task]:
        folder = run / "candidates" / spec["name"]
        folder.mkdir(parents=True, exist_ok=True)
        print("CANDIDATE", task, spec["name"], flush=True)
        if not (folder / "assessment.json").is_file():
            for subtask, path in prepared.items():
                if task == "vietlpr":
                    factory = lambda st=subtask: iter(samples[st])
                else:
                    factory = lambda: (yq.letterbox(read_image(r["image"]), 640)[0] for r in rows)
                quantize(path, folder / originals[subtask].name, factory, spec)
            execute("assess", "--task", task, "--folder", folder)
        result = read_json(folder / "assessment.json")
        if result["signature"] != signature():
            raise ValueError("Assessment khác code")
        result["spec"] = spec
        result["eligible"] = eligible(task, result, baseline, cfg)
        results.append(result)
        save_json(run / "trials.json", results)
    # Tổ hợp DET/REC kiểm trên cùng val, không dựa test.
    if task == "vietlpr":
        fast = min(results, key=lambda r: r["timing"]["Speed"])
        for subtask in ("det", "rec"):
            folder = run / "candidates" / f"{subtask}_only"
            folder.mkdir(exist_ok=True)
            for part in ("det", "rec"):
                src = Path(fast["folder"]) if part == subtask else base
                shutil.copy2(src / f"{part}.onnx", folder / f"{part}.onnx")
            if not (folder / "assessment.json").is_file():
                execute("assess", "--task", task, "--folder", folder)
            r = read_json(folder / "assessment.json")
            r["spec"] = {"name": f"{subtask}_only", "quantized_part": subtask, "from": fast["spec"]}
            r["eligible"] = eligible(task, r, baseline, cfg)
            results.append(r)
        save_json(run / "trials.json", results)
    selected, passed = choose(task, results, baseline, cfg)
    target = run / "selected"
    target.mkdir(exist_ok=True)
    for filename in selected["artifacts"]:
        shutil.copy2(Path(selected["folder"]) / filename, target / filename)
    state = {"protocol": protocol, "baseline_val": baseline, "selected_val": selected,
             "quality_gate_passed": passed, "artifacts": selected["artifacts"], "folder": str(target),
             "scope": selected["spec"], "training_performed": False}
    save_json(run / "selected.json", state)
    print("SELECTED", task, selected["spec"], "quality_gate", passed, flush=True)
    return state


def validate_selected(task):
    state = read_json(Path(settings()["run_dir"]) / task / "selected.json")
    if state["protocol"]["signature"] != signature():
        raise ValueError("Nguồn tối ưu đã đổi code/config")
    for filename, expected in state["artifacts"].items():
        if sha256(Path(state["folder"]) / filename) != expected:
            raise ValueError("Model được chọn đã thay đổi")
    return state


def main():
    logging.getLogger().setLevel(logging.WARNING)
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "assess"))
    p.add_argument("--task", choices=("vietlpr", "yolo"), required=True)
    p.add_argument("--folder", type=Path)
    args = p.parse_args()
    if args.action == "assess":
        assess(args.task, args.folder)
    else:
        with model_lock(Path(settings()["run_dir"]) / args.task):
            prepare_task(args.task)


if __name__ == "__main__":
    main()

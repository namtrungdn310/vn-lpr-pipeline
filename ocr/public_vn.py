"""Chuẩn bị và đo các trọng số OCR công khai chuyên biển số Việt Nam."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import re
import shutil
import time

import cv2
import numpy as np

from .config import ROOT, save_json
from .data import image, read_split, split_signature
from .metrics import speed_scores, text_scores
from .paddle_suite import digest, download
from .paddle_suite_state import BenchmarkState, model_lock
from .public_vn_runtime import PublicVNPipeline, model_input

CATALOG = ROOT / "configs/ocr/public_vn.json"
TIMING_SCOPE = "plate image to final text; excludes disk IO and model load"


def load_catalog():
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    for key in ("dataset", "run_dir"):
        catalog[key] = str((ROOT / catalog[key]).resolve())
    names = [v["name"] for v in catalog["variants"]]
    if len(names) != len(set(names)) or not names:
        raise ValueError("Tên model trùng hoặc danh sách rỗng")
    if min(catalog["benchmark"][key] for key in ("threads", "warmup", "repeats")) < 1:
        raise ValueError("Số luồng, warm-up và số lượt đo phải dương")
    for v in catalog["variants"]:
        if not re.fullmatch(r"[A-Za-z0-9_]+", v["name"]) or v["family"] not in ("cct", "lprnet", "yolov5", "yolov8"):
            raise ValueError("Tên hoặc loại model chưa được kiểm tra")
        prefix = f"https://github.com/{v['repository']}/blob/{v['revision']}/"
        if not v["training_evidence"].startswith(prefix) or len(v["weights"]["sha256"]) != 64:
            raise ValueError("Thiếu bằng chứng huấn luyện và SHA của trọng số đã chốt")
    return catalog


def variant(catalog, name):
    return next(v for v in catalog["variants"] if v["name"] == name)


def run_dir(catalog, name):
    variant(catalog, name)
    return Path(catalog["run_dir"]) / name


def profile(catalog, name):
    return {"variant": variant(catalog, name), "dataset": catalog["dataset"],
            "benchmark": catalog["benchmark"], "source_files": catalog["source_files"]}


def code_signature():
    names = ("public_vn.py", "public_vn_export.py", "public_vn_runtime.py", "data.py", "metrics.py",
             "paddle_suite_state.py")
    return hashlib.sha256(b"".join((Path(__file__).parent / n).read_bytes() for n in names)).hexdigest()


def artifacts(catalog, name):
    directory = run_dir(catalog, name)
    v = variant(catalog, name)
    files = {"onnx/model.onnx": digest(directory / "onnx/model.onnx"),
             "source/" + v["weights"]["filename"]: digest(directory / "source" / v["weights"]["filename"])}
    if "plate_config" in v:
        relative = "source/" + v["plate_config"]["filename"]
        files[relative] = digest(directory / relative)
    for source in catalog["source_files"]:
        path = Path(catalog["run_dir"]) / "source" / source["filename"]
        if digest(path) != source["sha256"]:
            raise ValueError("Nguồn chuyển đổi không khớp SHA đã chốt")
        files["../source/" + source["filename"]] = source["sha256"]
    if files["source/" + v["weights"]["filename"]] != v["weights"]["sha256"]:
        raise ValueError("Trọng số khác bản công khai đã chốt")
    return files


def validate_prepared(catalog, name):
    directory = run_dir(catalog, name)
    state = json.loads((directory / "prepared.json").read_text(encoding="utf-8"))
    if state["profile"] != profile(catalog, name) or state["artifacts"] != artifacts(catalog, name):
        raise ValueError("Model hoặc cấu hình đã thay đổi sau khi chuẩn bị")
    return state


def prepare(catalog, name):
    import onnx

    v = variant(catalog, name)
    directory = run_dir(catalog, name)
    if (directory / "prepared.json").is_file():
        validate_prepared(catalog, name)
        print(name, "đã sẵn sàng.", flush=True)
        return
    for source in catalog["source_files"]:
        download(source["url"], Path(catalog["run_dir"]) / "source" / source["filename"], source["sha256"])
    for source in [v["weights"]] + ([v["plate_config"]] if "plate_config" in v else []):
        download(source["url"], directory / "source" / source["filename"], source["sha256"])
    cv2.setNumThreads(1)
    val = read_split(catalog["dataset"], "val", "det")
    samples = [model_input(image(row["path"]), v) for row in val[:3]]
    if any(p is None for p in samples):
        raise ValueError("Ảnh smoke val không qua điều kiện nguồn")
    if v["family"] == "cct":
        (directory / "onnx").mkdir(exist_ok=True)
        shutil.copyfile(directory / "source" / v["weights"]["filename"], directory / "onnx/model.onnx")
        checks = {"checked_on": "val", "conversion": "public_onnx; SHA verified"}
    else:
        from .public_vn_export import export

        checks = export(v, directory, Path(catalog["run_dir"]) / "source", samples)
    graph = onnx.load(directory / "onnx/model.onnx", load_external_data=False)
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.graph.initializer):
        raise ValueError("Không nhận model có dữ liệu tensor bên ngoài")
    if any(t.data_type == onnx.TensorProto.FLOAT16 for t in graph.graph.initializer):
        raise ValueError("Model phải tính FP32 cho benchmark CPU")
    if any(n.op_type in ("QuantizeLinear", "DequantizeLinear", "ConvInteger", "MatMulInteger",
                         "QLinearConv", "QLinearMatMul") for n in graph.graph.node):
        raise ValueError("Không trộn model lượng tử hoá vào bảng FP32")
    onnx.checker.check_model(graph)
    pipe = PublicVNPipeline(directory, v, catalog["benchmark"]["threads"])
    checks["smoke_predictions"] = [pipe(image(r["path"]))["text"] for r in val[:3]]
    save_json(directory / "prepared.json", {"profile": profile(catalog, name),
              "artifacts": artifacts(catalog, name), "checks": checks, "trained_on_dataset": False})
    print(name, "đã chuẩn bị và kiểm val.", flush=True)


def freeze(catalog, name):
    """Chạy val với cấu hình nguồn cố định, không dò theo test."""
    state = validate_prepared(catalog, name)
    directory = run_dir(catalog, name)
    val = read_split(catalog["dataset"], "val", "det")
    cv2.setNumThreads(1)
    pipe = PublicVNPipeline(directory, variant(catalog, name), catalog["benchmark"]["threads"])
    predicted = []
    for i, row in enumerate(val, 1):
        predicted.append(pipe(image(row["path"]))["text"])
        if i % 50 == 0 or i == len(val):
            print(f"{name}: val {i}/{len(val)}", flush=True)
    frozen = {"profile": profile(catalog, name), "artifacts": state["artifacts"],
              "code_signature": code_signature(), "val_signature": split_signature(val),
              "selection": text_scores(predicted, [r["text"] for r in val]),
              "selection_method": "fixed public source defaults; no test selection"}
    save_json(directory / "onnx/frozen.json", frozen)
    return frozen


def validate_frozen(catalog, name, val):
    validate_prepared(catalog, name)
    frozen = json.loads((run_dir(catalog, name) / "onnx/frozen.json").read_text(encoding="utf-8"))
    if (frozen["profile"] != profile(catalog, name) or frozen["artifacts"] != artifacts(catalog, name)
            or frozen["code_signature"] != code_signature() or frozen["val_signature"] != split_signature(val)):
        raise ValueError("Code, model hoặc val đã khác bản chốt")
    return frozen


def benchmark(catalog, name):
    import onnxruntime as ort

    directory = run_dir(catalog, name)
    with model_lock(directory):
        val = read_split(catalog["dataset"], "val", "det")
        frozen = validate_frozen(catalog, name, val)
        rows = read_split(catalog["dataset"], "test", "det")
        pixels = [image(row["path"]) for row in rows]
        params = catalog["benchmark"]
        protocol = {"name": name, "profile": profile(catalog, name), "artifacts": frozen["artifacts"],
                    "code_signature": code_signature(), "val_signature": frozen["val_signature"],
                    "test_signature": split_signature(rows), "runtime": ort.__version__,
                    "platform": platform.platform(), "processor": platform.processor()}
        state = BenchmarkState(directory / "benchmark_progress.json", protocol, len(rows), params["repeats"])
        pipe = PublicVNPipeline(directory, variant(catalog, name), params["threads"])
        cv2.setNumThreads(1)
        if state.completed < len(rows) * params["repeats"]:
            warm = image(val[0]["path"])
            for _ in range(params["warmup"]):
                pipe(warm)
        for repeat in range(params["repeats"]):
            order = list(range(len(rows)))
            random.Random(params["seed"] + repeat).shuffle(order)
            for i in order:
                if len(state.times[i]) > repeat:
                    continue
                start = time.perf_counter_ns()
                prediction = pipe(pixels[i])["text"]
                elapsed = (time.perf_counter_ns() - start) / 1_000_000
                state.record(i, repeat, prediction, elapsed)
            print(f"{name}: test {repeat + 1}/{params['repeats']}", flush=True)
        result = {"Model": name, **text_scores(state.predictions, [r["text"] for r in rows]),
                  **speed_scores([t for item in state.times for t in item]), **protocol,
                  "threads": params["threads"], "opencv_threads": 1, "batch_size": 1, "batch_unit": "plate",
                  "precision": "FP32", "provider": "CPUExecutionProvider", "warmup": params["warmup"],
                  "repeats": params["repeats"], "timing_scope": TIMING_SCOPE,
                  "pipeline_type": "whole_plate_recognition" if variant(catalog, name)["family"] in ("cct", "lprnet") else "character_detection_and_ordering",
                  "artifact_sha256": frozen["artifacts"], "weights_source": "public_vietnam_plate_pretrained",
                  "trained_on_dataset": False, "external_training_overlap": "unknown"}
        save_json(directory / "test_predictions.json", [
            {"image": r["relative"], "GT": r["text"], "prediction": p, "correct": p == r["text"], "times_ms": t}
            for r, p, t in zip(rows, state.predictions, state.times)
        ])
        save_json(directory / "test_result.json", result)
        return result


def main():
    catalog = load_catalog()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "freeze", "test"))
    parser.add_argument("--model", choices=[v["name"] for v in catalog["variants"]], nargs="+")
    args = parser.parse_args()
    for name in args.model or [v["name"] for v in catalog["variants"]]:
        if args.action == "prepare":
            prepare(catalog, name)
        elif args.action == "freeze":
            freeze(catalog, name)
        else:
            benchmark(catalog, name)


if __name__ == "__main__":
    main()

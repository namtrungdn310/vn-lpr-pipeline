"""Lượng tử hoá detector đã train; chốt trên val, đo riêng trên test."""

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time

import cv2
import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort

from ocr.config import ROOT, save_json
from ocr.benchmark import check_power, environment, power_context, round_order
from ocr.paddle_suite_state import model_lock
from .data import dataset_signature, load_config, read_image, sha256
from .evaluate import operating_scores, split_rows
from .runtime import PlateDetector, letterbox
from .train import compare_outputs, ultralytics_api

CONFIG = ROOT / "configs/detection/yolo26_quantization.json"
NAMES = {"fp32": "YOLO26n-OBB_FP32", "int8": "YOLO26n-OBB_INT8", "int4": "YOLO26n-OBB_INT4_mixed"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def now():
    return datetime.now(timezone.utc).isoformat()


def settings():
    cfg = read_json(CONFIG)
    cfg["run_dir"] = str((ROOT / cfg["run_dir"]).resolve())
    if cfg["rounds"] < 2 or min(cfg[k] for k in ("threads", "warmup", "calibration_train_images")) < 1:
        raise ValueError("Cấu hình hiệu chuẩn/đo không hợp lệ")
    return cfg


def code_signature():
    paths = [Path(__file__), CONFIG, ROOT / "ocr/benchmark.py"]
    paths += [ROOT / "plate_detection" / f for f in ("runtime.py", "evaluate.py", "train.py", "data.py")]
    return hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()


def inventory(path):
    graph = onnx.load(path)
    return {"sha256": sha256(path), "bytes": Path(path).stat().st_size,
            "ops": dict(Counter(n.op_type for n in graph.graph.node))}


def rewrite_pointwise(source, target):
    """Conv 1x1 stride 1, group 1, không pad: chuyển đúng đại số sang MatMul."""
    model = onnx.load(source)
    weights = {w.name: numpy_helper.to_array(w) for w in model.graph.initializer}
    nodes, added, replacements = [], [], []
    for index, node in enumerate(model.graph.node):
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        weight = weights.get(node.input[1]) if node.op_type == "Conv" else None
        eligible = (weight is not None and weight.ndim == 4 and weight.shape[2:] == (1, 1)
                    and attrs.get("group", 1) == 1 and attrs.get("strides", [1, 1]) == [1, 1]
                    and not any(attrs.get("pads", [0, 0, 0, 0]))
                    and attrs.get("auto_pad", b"NOTSET") in (b"NOTSET", b"VALID"))
        if not eligible:
            nodes.append(deepcopy(node))
            continue
        prefix = f"pointwise_int4_{index}"
        names = {k: prefix + "_" + k for k in ("nhwc", "weight", "mm", "biased", "bias")}
        added.append(numpy_helper.from_array(weight[:, :, 0, 0].T.copy(), names["weight"]))
        nodes.append(helper.make_node("Transpose", [node.input[0]], [names["nhwc"]],
                                      name=prefix + "_in", perm=[0, 2, 3, 1]))
        nodes.append(helper.make_node("MatMul", [names["nhwc"], names["weight"]], [names["mm"]], name=prefix))
        output = names["mm"]
        if len(node.input) > 2 and node.input[2]:
            added.append(numpy_helper.from_array(weights[node.input[2]].copy(), names["bias"]))
            nodes.append(helper.make_node("Add", [output, names["bias"]], [names["biased"]], name=prefix + "_add"))
            output = names["biased"]
        nodes.append(helper.make_node("Transpose", [output], list(node.output),
                                      name=prefix + "_out", perm=[0, 3, 1, 2]))
        replacements.append({"conv": node.name, "matmul": prefix, "weight_elements": int(weight.size)})
    if not replacements:
        raise ValueError("Không có Conv 1x1 phù hợp; không xuất file INT4 giả")
    used = {name for n in nodes for name in n.input} | {v.name for v in model.graph.output}
    retained = [deepcopy(w) for w in model.graph.initializer if w.name in used]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained + added)
    # ValueInfo cũ có thể mang shape không còn cần thiết sau khi thay phép tính.
    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    onnx.save(model, target)
    return {"converted_convs": len(replacements), "converted_weight_elements": sum(r["weight_elements"] for r in replacements),
            "replacements": replacements}


def int4_model(source, target, block):
    from onnxruntime.quantization.matmul_4bits_quantizer import DefaultWeightOnlyQuantConfig, MatMul4BitsQuantizer

    cfg = DefaultWeightOnlyQuantConfig(block_size=block, is_symmetric=True, accuracy_level=0,
                                      op_types_to_quantize=("MatMul",))
    quantizer = MatMul4BitsQuantizer(onnx.load(source), algo_config=cfg)
    quantizer.process()
    quantizer.model.save_model_to_file(str(target), use_external_data_format=False)
    onnx.checker.check_model(str(target))


def calibration_rows(cfg, source):
    rows = split_rows(source, "train")
    random.Random(cfg["seed"]).shuffle(rows)
    return rows[:cfg["calibration_train_images"]]


def int8_model(source, target, rows, size):
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static

    input_name = onnx.load(source).graph.input[0].name

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.items = iter(rows)
            self.count = 0

        def get_next(self):
            item = next(self.items, None)
            if item is None:
                return None
            self.count += 1
            if self.count == 1 or self.count % 32 == 0:
                print(f"Calibration train: {self.count}/{len(rows)}", flush=True)
            return {input_name: letterbox(read_image(item["image"]), size)[0]}

    quantize_static(str(source), str(target), Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    per_channel=True, calibrate_method=CalibrationMethod.MinMax,
                    op_types_to_quantize=["Conv"],
                    extra_options={"ActivationSymmetric": True, "WeightSymmetric": True,
                                   "CalibTensorRangeSymmetric": True})
    onnx.checker.check_model(str(target))


def detector(path, cfg, source):
    inf = source["inference"]
    return PlateDetector(path, size=source["train"]["imgsz"], confidence=inf["confidence"],
                         max_det=inf["max_det"], threads=cfg["threads"])


def native_ap(path, split, output):
    """mAP giữ evaluator OBB gốc; P/R cố định được tính riêng."""
    import torch

    cfg, source = settings(), load_config()
    torch.set_num_threads(cfg["threads"])
    cv2.setNumThreads(cfg["opencv_threads"])
    protocol = {"model_sha256": sha256(path), "dataset_signature": dataset_signature(source["dataset"]),
                "split": split, "code_signature": code_signature(), "runtime": ort.__version__}
    output = Path(output)
    if output.is_file():
        previous = read_json(output)
        if previous["protocol"] != protocol:
            raise ValueError("mAP cũ khác model/dataset/code")
        return previous
    YOLO = ultralytics_api(source)
    metrics = YOLO(str(path), task="obb").val(
        data=str(Path(source["dataset"]) / "data.yaml"), split=split, device="cpu",
        imgsz=source["train"]["imgsz"], batch=1, workers=0, rect=False, conf=0.001,
        max_det=300, plots=False, project=str(output.parent), name=output.stem + "_native", exist_ok=True)
    result = {"protocol": protocol, "mAP50": 100 * float(metrics.box.map50),
              "mAP50-95": 100 * float(metrics.box.map)}
    save_json(output, result)
    return result


def child(*args):
    subprocess.run([sys.executable, "-u", "-m", "plate_detection.quantization", *map(str, args)],
                   cwd=ROOT, check=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})


def evaluate_ap(path, split, output):
    child("ap", "--model", path, "--split", split, "--output", output)
    return read_json(output)


def prepare():
    from onnxruntime.quantization.shape_inference import quant_pre_process

    cfg, source = settings(), load_config()
    run, original = Path(cfg["run_dir"]), Path(source["run_dir"]) / "onnx/model.onnx"
    verified = read_json(original.with_name("verified.json"))
    signature = dataset_signature(source["dataset"])
    if verified["onnx_sha256"] != sha256(original) or verified["dataset_signature"] != signature or verified["config"] != source:
        raise ValueError("Model/dataset khác bản đã kiểm sau train")
    protocol = {"config": cfg, "source_config": source, "source_sha256": sha256(original),
                "dataset_signature": signature, "code_signature": code_signature(), "environment": environment()}
    run.mkdir(parents=True, exist_ok=True)
    if (run / "protocol.json").is_file() and read_json(run / "protocol.json") != protocol:
        raise ValueError("Lượt cũ khác cấu hình/code; không trộn hoặc ghi đè")
    save_json(run / "protocol.json", protocol)
    if (run / "prepared.json").is_file():
        return validate_prepared()
    baseline = [original, original.with_name("verified.json")]
    baseline += [Path(source["run_dir"]) / p for p in
                 ("weights/best.pt", "weights/last.pt", "detection_result.json", "detection_predictions.json", "onnx_test_ap.json")]
    if not (run / "baseline_snapshot.json").is_file():
        save_json(run / "baseline_snapshot.json", {str(p): sha256(p) for p in baseline})
    for name in NAMES:
        (run / name).mkdir(exist_ok=True)
    shutil.copy2(original, run / "fp32/model.onnx")
    rows = calibration_rows(cfg, source)
    save_json(run / "calibration.json", {"split": "train", "samples": len(rows), "rows": rows,
                                         "preprocessing": "native letterbox 114, BGR->RGB, NCHW FP32 /255"})
    pre = run / "preprocessed.onnx"
    if not pre.is_file():
        quant_pre_process(original, pre, skip_optimization=True, skip_symbolic_shape=True)
    print("Tạo INT8 S8S8 QDQ từ Conv gốc.", flush=True)
    int8_model(pre, run / "int8/model.onnx", rows, source["train"]["imgsz"])
    rewritten = run / "pointwise_fp32.onnx"
    rewrite = rewrite_pointwise(original, rewritten)
    # Kiểm phép đổi đại số trước khi làm tròn trọng số.
    first, second = detector(original, cfg, source), detector(rewritten, cfg, source)
    checks = []
    for row in split_rows(source, "val")[:8]:
        tensor = letterbox(read_image(row["image"]), source["train"]["imgsz"])[0]
        a = first.engine.run(None, {first.input: tensor})[0]
        b = second.engine.run(None, {second.input: tensor})[0]
        selected, error = compare_outputs(a, b, source["inference"]["confidence"])
        checks.append({"image": row["image"], "selected_boxes": selected, "max_score_error": error})
    if not sum(c["selected_boxes"] for c in checks):
        raise ValueError("Kiểm tương đương chưa có hộp nào")
    save_json(run / "pointwise_equivalence.json", {**rewrite, "checks": checks})
    del first, second
    fpval = evaluate_ap(run / "fp32/model.onnx", "val", run / "fp32/val.json")
    i8val = evaluate_ap(run / "int8/model.onnx", "val", run / "int8/val.json")
    trials = []
    for block in cfg["int4_block_sizes"]:
        folder = run / "candidates" / f"int4_block{block}"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / "model.onnx"
        int4_model(rewritten, target, block)
        graph = inventory(target)
        if graph["ops"].get("MatMulNBits", 0) != rewrite["converted_convs"]:
            raise ValueError("INT4 không có đủ phép toán đóng gói 4-bit")
        scores = evaluate_ap(target, "val", folder / "val.json")
        trials.append({"block_size": block, "val": scores, "graph": graph, "path": str(target)})
    chosen = max(trials, key=lambda r: (r["val"]["mAP50-95"], r["val"]["mAP50"]))
    shutil.copy2(chosen["path"], run / "int4/model.onnx")
    models = {}
    for name, scores, scope in (("fp32", fpval, "ONNX gốc FP32"),
                               ("int8", i8val, "Conv W8A8 S8S8 QDQ; phép khác FP32"),
                               ("int4", chosen["val"], cfg["int4_scope"])):
        graph = inventory(run / name / "model.onnx")
        if name == "int8" and graph["ops"].get("QuantizeLinear", 0) == 0:
            raise ValueError("INT8 không có graph lượng tử hoá")
        models[name] = {"val": scores, "scope": scope, "graph": graph}
    state = {"protocol": protocol, "models": models, "int4_selection": chosen,
             "int4_trials": trials, "rewrite": rewrite, "training_performed": False, "selection_split": "val"}
    save_json(run / "prepared.json", state)
    return state


def validate_prepared():
    cfg, source = settings(), load_config()
    run = Path(cfg["run_dir"])
    state = read_json(run / "prepared.json")
    protocol = state["protocol"]
    if protocol != read_json(run / "protocol.json") or protocol["config"] != cfg or protocol["source_config"] != source:
        raise ValueError("Cấu hình khác lượt đã chốt")
    if protocol["code_signature"] != code_signature() or protocol["environment"] != environment():
        raise ValueError("Code hoặc môi trường đã đổi")
    if protocol["dataset_signature"] != dataset_signature(source["dataset"]):
        raise ValueError("Dataset đã đổi")
    for name in NAMES:
        if inventory(run / name / "model.onnx") != state["models"][name]["graph"]:
            raise ValueError("Model đã đổi sau khi chốt val")
    verify_baseline(run)
    return state


def verify_baseline(run):
    for path, expected in read_json(Path(run) / "baseline_snapshot.json").items():
        if sha256(path) != expected:
            raise ValueError("Baseline gốc bị thay đổi: " + path)


def ground_truth(rows, pixels):
    actual = []
    for row, image in zip(rows, pixels, strict=True):
        h, w = image.shape[:2]
        actual.append([(np.array(line.split()[1:], dtype=float).reshape(4, 2) * [w, h]).tolist()
                       for line in Path(row["label"]).read_text().splitlines() if line.strip()])
    return actual


def measure_in_memory(pipeline, pixels, warm, cfg, index):
    for _ in range(cfg["warmup"]):
        pipeline(warm)
    predictions, times = [None] * len(pixels), [None] * len(pixels)
    order = list(range(len(pixels)))
    random.Random(cfg["seed"] + index).shuffle(order)
    for item in order:
        start = time.perf_counter_ns()
        prediction = pipeline(pixels[item])
        elapsed = (time.perf_counter_ns() - start) / 1e6
        predictions[item], times[item] = prediction, elapsed
    return predictions, times


def speed_worker(name, index):
    cfg, source = settings(), load_config()
    state = validate_prepared()
    if name not in NAMES or index not in range(cfg["rounds"]):
        raise ValueError("Sai model/lượt")
    run = Path(cfg["run_dir"])
    power = read_json(run / "benchmark_manifest.json")["power"]
    before = check_power(power, cfg)
    cv2.setNumThreads(cfg["opencv_threads"])
    rows = split_rows(source, "test")
    pixels = [read_image(row["image"]) for row in rows]
    actual = ground_truth(rows, pixels)
    warm = read_image(split_rows(source, "val")[0]["image"])
    pipe = detector(run / name / "model.onnx", cfg, source)
    predictions, times = measure_in_memory(pipe, pixels, warm, cfg, index)
    after = check_power(power, cfg)
    scores = operating_scores(predictions, actual, source["benchmark"]["match_iou"])
    record = {"name": name, "round": index, "model_sha256": state["models"][name]["graph"]["sha256"],
              "code_signature": code_signature(), "dataset_signature": state["protocol"]["dataset_signature"],
              "power_before": before, "power_after": after, "scores": scores,
              "details": [{"image": r["image"], "GT": gt, "prediction": p, "time_ms": t}
                          for r, gt, p, t in zip(rows, actual, predictions, times, strict=True)]}
    target = run / "passes" / name / f"round_{index + 1:02d}.json"
    save_json(target, record)
    print(f"{name} {index + 1}/{cfg['rounds']}: {statistics.mean(times):.3f} ms/ảnh", flush=True)


def aggregate(records, name, state, rows, cfg, source):
    if len(records) != cfg["rounds"]:
        raise ValueError("Chưa đủ lượt đo")
    reference, means = None, []
    actual = ground_truth(rows, [read_image(r["image"]) for r in rows])
    for index, record in enumerate(records):
        if (record["name"] != name or record["round"] != index
                or record["model_sha256"] != state["models"][name]["graph"]["sha256"]
                or record["code_signature"] != code_signature()
                or record["dataset_signature"] != state["protocol"]["dataset_signature"]):
            raise ValueError("Lượt đo khác model/code/dataset")
        details = record["details"]
        if [d["image"] for d in details] != [r["image"] for r in rows]:
            raise ValueError("Danh sách test khác nhau")
        predictions = [d["prediction"] for d in details]
        if [d["GT"] for d in details] != actual:
            raise ValueError("Nhãn GT khác dataset")
        if reference is not None and reference != predictions:
            raise ValueError("Dự đoán thay đổi giữa các lượt")
        if (record["power_before"] != record["power_after"]
                or (cfg["require_ac"] and record["power_before"]["ac"] is False)):
            raise ValueError("Điều kiện nguồn thay đổi trong lượt đo")
        reference = predictions
        scores = operating_scores(predictions, actual, source["benchmark"]["match_iou"])
        if scores != record["scores"] or any(not math.isfinite(d["time_ms"]) or d["time_ms"] <= 0 for d in details):
            raise ValueError("Chỉ số khác số đo gốc")
        means.append(statistics.mean(d["time_ms"] for d in details))
    speed = statistics.mean(means)
    return {"Model": NAMES[name], **scores, "Speed": speed, "SD": statistics.stdev(means), "FPS": 1000 / speed,
            "Size": state["models"][name]["graph"]["bytes"] / 1e6, "round_means_ms": means,
            "images": len(rows), "rounds": cfg["rounds"], "scope": state["models"][name]["scope"]}


def table():
    cfg, source = settings(), load_config()
    run, state = Path(cfg["run_dir"]), validate_prepared()
    rows, results = split_rows(source, "test"), []
    for name in NAMES:
        paths = [run / "passes" / name / f"round_{i + 1:02d}.json" for i in range(cfg["rounds"])]
        if not all(p.is_file() for p in paths) or not (run / name / "test_ap.json").is_file():
            return "Chưa đo xong; xem state.json trong thư mục quantization."
        result = aggregate([read_json(p) for p in paths], name, state, rows, cfg, source)
        ap = read_json(run / name / "test_ap.json")
        if (ap["protocol"]["model_sha256"] != state["models"][name]["graph"]["sha256"]
                or ap["protocol"]["dataset_signature"] != state["protocol"]["dataset_signature"]
                or ap["protocol"]["code_signature"] != code_signature() or ap["protocol"]["split"] != "test"):
            raise ValueError("mAP khác model đã chốt")
        result.update({k: ap[k] for k in ("mAP50", "mAP50-95")})
        save_json(run / name / "test_result.json", result)
        results.append(result)
    slowest = max(r["Speed"] for r in results)
    results.sort(key=lambda r: (-r["mAP50-95"], -r["Recall"], r["Speed"]))
    import csv

    keys = ("Rank", "Model", "mAP50", "mAP50-95", "Precision", "Recall", "Speed", "SD", "FPS", "Speedup", "Size")
    for index, r in enumerate(results, 1):
        r.update(Rank=index, Speedup=slowest / r["Speed"])
    with (run / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    text = "| Rank | Model | mAP50 (%) ↑ | mAP50-95 (%) ↑ | Precision (%) ↑ | Recall (%) ↑ | Speed ± SD (ms/ảnh) ↓ | FPS ↑ | Speedup (×) ↑ | Size (MB) ↓ |\n"
    text += "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for r in results:
        values = [str(r["Rank"]), r["Model"], *(f"{r[k]:.2f}" for k in ("mAP50", "mAP50-95", "Precision", "Recall")),
                  f"{r['Speed']:.2f} ± {r['SD']:.2f}", *(f"{r[k]:.2f}" for k in ("FPS", "Speedup", "Size"))]
        text += "| " + " | ".join(values) + " |\n"
    (run / "comparison.md").write_text(text, encoding="utf-8")
    return text


def benchmark():
    cfg, source = settings(), load_config()
    state = validate_prepared()
    run = Path(cfg["run_dir"])
    manifest = {"protocol": state["protocol"], "power": power_context(),
                "orders": [round_order(list(NAMES), i, cfg["seed"]) for i in range(cfg["rounds"])],
                "timing_scope": cfg["timing_scope"], "file_io_in_timed_pass": False,
                "std_definition": "sample SD of per-round mean latency (ddof=1)"}
    if (run / "benchmark_manifest.json").is_file():
        old = read_json(run / "benchmark_manifest.json")
        check_power(old["power"], cfg)
        manifest["power"] = old["power"]
        if manifest != old:
            raise ValueError("Điều kiện đo khác lượt cũ")
    check_power(manifest["power"], cfg)
    save_json(run / "benchmark_manifest.json", manifest)
    for name in NAMES:
        evaluate_ap(run / name / "model.onnx", "test", run / name / "test_ap.json")
    done = 0
    for index, order in enumerate(manifest["orders"]):
        for name in order:
            path = run / "passes" / name / f"round_{index + 1:02d}.json"
            save_json(run / "state.json", {"status": "benchmark", "model": name, "round": index + 1,
                                           "completed_passes": done, "total_passes": cfg["rounds"] * len(NAMES), "updated_utc": now()})
            if not path.is_file():
                child("speed", "--name", name, "--round", index)
            done += 1
    text = table()
    verify_baseline(run)
    save_json(run / "verification.json", {"baseline_files_unchanged": len(read_json(run / "baseline_snapshot.json")),
                                          "models": len(NAMES), "rounds": cfg["rounds"], "test_images": len(split_rows(source, "test")),
                                          "calibration_split": "train", "selection_split": "val", "training_performed": False,
                                          "int4_scope": cfg["int4_scope"], "file_io_in_timed_pass": False})
    save_json(run / "state.json", {"status": "complete", "completed_passes": done,
                                   "total_passes": done, "finished_utc": now()})
    refresh_notebook(text)
    print(text, flush=True)


def refresh_notebook(text):
    import nbformat

    path = ROOT / "notebooks/07_quantize_yolo26_obb.ipynb"
    notebook = nbformat.read(path, as_version=4)
    cell = next(c for c in notebook.cells if c.cell_type == "code" and "# saved_comparison" in c.source)
    cell.outputs = [nbformat.v4.new_output("display_data", data={"text/markdown": text, "text/plain": text})]
    cell.execution_count = None
    nbformat.validate(notebook)
    nbformat.write(notebook, path)


def main():
    logging.getLogger().setLevel(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "test", "all", "table", "ap", "speed"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--split", choices=("val", "test"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--name", choices=tuple(NAMES))
    parser.add_argument("--round", type=int)
    args = parser.parse_args()
    if args.action == "ap":
        native_ap(args.model, args.split, args.output)
    elif args.action == "speed":
        speed_worker(args.name, args.round)
    elif args.action == "table":
        print(table())
    else:
        run = Path(settings()["run_dir"])
        with model_lock(run):
            try:
                if args.action in ("prepare", "all"):
                    save_json(run / "state.json", {"status": "preparing", "updated_utc": now()})
                    prepare()
                if args.action in ("test", "all"):
                    benchmark()
            except Exception as error:
                save_json(run / "state.json", {"status": "error", "error": str(error), "updated_utc": now()})
                raise


if __name__ == "__main__":
    main()

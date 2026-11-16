"""Lượng tử hoá VietLPR V2, chọn bằng val và so FP32/INT8/INT4 hỗn hợp."""

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
import platform
import random
import shutil
import time

import cv2
import numpy as np
import onnx
import onnxruntime as ort

from .config import ROOT, save_json
from .data import image, read_split, split_signature
from .evaluate import deployed_config, validate_export
from .metrics import speed_scores, text_scores
from .paddle_suite import digest
from .paddle_suite_state import model_lock
from .runtime import inference_signature, onnx_pipeline, tensor

CONFIG = ROOT / "configs/ocr/vietlpr_v2_quantization.json"
VARIANTS = ("fp32", "int8", "int4")
LABELS = {"fp32": "VietLPR_v2 FP32", "int8": "VietLPR_v2 INT8", "int4": "VietLPR_v2 INT4 mixed"}
MEASUREMENT_PROTOCOL = "notebook03_in_memory_v1"


def settings():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    cfg["run_dir"] = str((ROOT / cfg["run_dir"]).resolve())
    if cfg["source_model"] != "vietlpr_v2" or min(cfg[k] for k in (
            "threads", "warmup", "repeats", "calibration_train_images", "calibration_rec_images")) < 1:
        raise ValueError("Cấu hình lượng tử hoá không hợp lệ")
    return cfg


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_model(cfg):
    source = deployed_config(cfg["source_model"])
    validate_export(source)
    directory = Path(source["run_dir"]) / "onnx"
    frozen = read_json(directory / "frozen.json")
    if frozen["inference_signature"] != inference_signature():
        raise ValueError("Code suy luận khác bản FP32 đã duyệt")
    for name, expected in frozen["files"].items():
        if digest(directory / name) != expected:
            raise ValueError("Nguồn FP32 thay đổi sau khi chốt")
    val = read_split(source["dataset"], "val", "det")
    if frozen["val_signature"] != split_signature(val):
        raise ValueError("Val khác nguồn FP32 đã chốt")
    source["benchmark"]["threads"] = cfg["threads"]
    return source, directory, frozen


def code_signature():
    return hashlib.sha256(Path(__file__).read_bytes() + CONFIG.read_bytes()
                          + inference_signature().encode()
                          + (Path(__file__).parent / "paddle_suite_state.py").read_bytes()).hexdigest()


def model_hashes(directory):
    return {f"{task}.onnx": digest(Path(directory) / f"{task}.onnx") for task in ("det", "rec")}


def inventory(path):
    """Đếm phép toán và trọng số để ghi rõ phạm vi lượng tử hoá."""
    from collections import Counter

    model = onnx.load(path)
    onnx.checker.check_model(model)
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer):
        raise ValueError("Model cần lưu tensor trong một file ONNX")
    weights = {t.name: t for t in model.graph.initializer}
    matmul_weights = {n.input[1] for n in model.graph.node if n.op_type == "MatMul"
                      and len(n.input) > 1 and n.input[1] in weights}
    total = sum(int(np.prod(t.dims)) for t in weights.values() if t.data_type == onnx.TensorProto.FLOAT)
    return {"bytes": Path(path).stat().st_size, "ops": dict(Counter(n.op_type for n in model.graph.node)),
            "float_initializer_elements": total,
            "constant_matmul_weight_elements": sum(int(np.prod(weights[n].dims)) for n in matmul_weights),
            "initializer_types": dict(Counter(onnx.TensorProto.DataType.Name(t.data_type) for t in weights.values()))}


def calibration(cfg, source):
    """Chỉ dùng train; REC có cả crop do DET dự đoán để sát đường suy luận."""
    train = read_split(source["dataset"], "train", "det")
    indices = list(range(len(train)))
    random.Random(cfg["seed"]).shuffle(indices)
    chosen = [train[i] for i in indices[:cfg["calibration_train_images"]]]
    pipe = onnx_pipeline(source)
    det_data, rec_data = [], []
    for row in chosen:
        pixels = image(row["path"])
        resized = cv2.resize(pixels, (source["model"]["det_size"],) * 2, interpolation=cv2.INTER_LINEAR)
        det_data.append(np.ascontiguousarray(resized.transpose(2, 0, 1)[None], dtype=np.float32) * (1 / 255.0))
        detected = pipe.source.detect(pixels)
        crops = [pipe.source.crop(pixels, q) for q, _ in detected]
        if crops:
            rec_data.append(np.ascontiguousarray(np.stack(crops).transpose(0, 3, 1, 2), dtype=np.float32) * (1 / 255.0))
    rec_rows = read_split(source["dataset"], "train", "rec")
    ids = list(range(len(rec_rows)))
    random.Random(cfg["seed"] + 1).shuffle(ids)
    selected_rec = [rec_rows[i] for i in ids[:cfg["calibration_rec_images"]]]
    for row in selected_rec:
        pixels = image(row["path"])
        h, w = pixels.shape[:2]
        crop = pipe.source.crop(pixels, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
        rec_data.append(tensor(crop))
    recipe = {"split": "train", "det_signature": split_signature(chosen),
              "rec_signature": split_signature(selected_rec), "det_samples": len(det_data),
              "rec_batches": len(rec_data), "rec_samples": sum(x.shape[0] for x in rec_data),
              "rec_source": "FP32 predicted train DET crops + existing train REC crops"}
    return {"det": det_data, "rec": rec_data}, recipe


def static_int8(path, output, samples, method):
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static

    class Reader(CalibrationDataReader):
        def __init__(self):
            # Histogram calibration của ORT 1.20 yêu cầu các batch cùng kích thước.
            self.samples = (sample[i:i + 1] for sample in samples for i in range(sample.shape[0]))

        def get_next(self):
            item = next(self.samples, None)
            return {"input": item} if item is not None else None

    quantize_static(str(path), str(output), Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    per_channel=True, op_types_to_quantize=["Conv", "MatMul"],
                    calibrate_method=CalibrationMethod.MinMax if method == "minmax" else CalibrationMethod.Percentile,
                    extra_options={"ActivationSymmetric": True, "WeightSymmetric": True,
                                   "CalibTensorRangeSymmetric": True, "CalibPercentile": 99.99,
                                   "DedicatedQDQPair": True})


def weight_int4(path, output, block_size):
    from onnxruntime.quantization.matmul_4bits_quantizer import DefaultWeightOnlyQuantConfig, MatMul4BitsQuantizer

    config = DefaultWeightOnlyQuantConfig(block_size=block_size, is_symmetric=True,
                                          accuracy_level=0, op_types_to_quantize=("MatMul",))
    quantizer = MatMul4BitsQuantizer(onnx.load(path), algo_config=config)
    quantizer.process()
    quantizer.model.save_model_to_file(str(output), use_external_data_format=False)


def val_scores(source, directory, rows):
    pipe = onnx_pipeline(source, directory)
    predicted = [pipe(image(row["path"]))["text"] for row in rows]
    return text_scores(predicted, [row["text"] for row in rows])


def prepare(cfg):
    from onnxruntime.quantization.shape_inference import quant_pre_process

    logging.getLogger().setLevel(logging.WARNING)
    cv2.setNumThreads(1)
    run = Path(cfg["run_dir"])
    source, original, frozen = source_model(cfg)
    val = read_split(source["dataset"], "val", "det")
    protocol = {"settings": cfg, "source": frozen, "code_signature": code_signature(),
                "runtime": ort.__version__, "val_signature": split_signature(val)}
    with model_lock(run):
        if (run / "prepared.json").is_file():
            state = read_json(run / "prepared.json")
            if state["protocol"] != protocol:
                raise ValueError("Cấu hình/code/model khác lượt lượng tử hoá cũ; cần run_dir mới")
            for name in VARIANTS:
                if model_hashes(run / name) != state["models"][name]["artifacts"]:
                    raise ValueError("Model lượng tử hoá đã thay đổi")
            print("Dùng lại ba bản đã chốt trên val.", flush=True)
            return state
        fp32 = run / "fp32"
        fp32.mkdir(parents=True, exist_ok=True)
        for task in ("det", "rec"):
            shutil.copy2(original / f"{task}.onnx", fp32 / f"{task}.onnx")
        data, recipe = calibration(cfg, source)
        save_json(run / "calibration.json", recipe)
        print("Calibration train:", recipe, flush=True)
        pre = run / "preprocessed"
        pre.mkdir(exist_ok=True)
        for task in ("det", "rec"):
            quant_pre_process(original / f"{task}.onnx", pre / f"{task}.onnx", skip_optimization=True,
                              skip_symbolic_shape=False, auto_merge=True)
        models = {"fp32": {"artifacts": model_hashes(fp32), "val": val_scores(source, fp32, val),
                           "scope": "DET/REC FP32", "candidate": "original"}}
        if any(abs(models["fp32"]["val"][key] - frozen["selection"][key]) > 1e-9 for key in ("ACC", "CER")):
            raise ValueError("Kết quả val FP32 không khớp baseline nguồn")
        for precision, candidates in (("int8", cfg["int8_candidates"]), ("int4", cfg["int4_block_sizes"])):
            trials = []
            for candidate in candidates:
                directory = run / "candidates" / f"{precision}_{candidate}"
                directory.mkdir(parents=True, exist_ok=True)
                for task in ("det", "rec"):
                    output = directory / f"{task}.onnx"
                    if precision == "int8":
                        static_int8(pre / f"{task}.onnx", output, data[task], candidate)
                    elif task == "det":
                        shutil.copy2(original / "det.onnx", output)
                    else:
                        weight_int4(original / "rec.onnx", output, candidate)
                graphs = {task: inventory(directory / f"{task}.onnx") for task in ("det", "rec")}
                if precision == "int4" and graphs["rec"]["ops"].get("MatMulNBits", 0) != 5:
                    raise ValueError("INT4 chưa lượng tử hoá đủ năm MatMul trọng số của REC")
                if precision == "int8" and any(not graph["ops"].get("QuantizeLinear", 0) for graph in graphs.values()):
                    raise ValueError("INT8 chưa có graph lượng tử hoá thực")
                scores = val_scores(source, directory, val)
                trial = {"candidate": candidate, "val": scores, "graphs": graphs, "directory": str(directory)}
                trials.append(trial)
                print(precision, candidate, "val:", scores, flush=True)
            selected = max(trials, key=lambda row: (row["val"]["ACC"], -row["val"]["CER"]))
            target = run / precision
            target.mkdir(exist_ok=True)
            for task in ("det", "rec"):
                shutil.copy2(Path(selected["directory"]) / f"{task}.onnx", target / f"{task}.onnx")
            models[precision] = {"artifacts": model_hashes(target), "val": selected["val"],
                                 "candidate": selected["candidate"], "trials": trials,
                                 "scope": "Conv/MatMul W8A8 QDQ; unsupported ops FP32" if precision == "int8" else cfg["int4_scope"]}
        for name, state in models.items():
            state["graphs"] = {task: inventory(run / name / f"{task}.onnx") for task in ("det", "rec")}
            state["total_bytes"] = sum(g["bytes"] for g in state["graphs"].values())
            for file in ("config.json", "lpr_config.json", "lpr_infer.py"):
                shutil.copy2(original / file, run / name / file)
            save_json(run / name / "frozen.json", {"protocol": protocol, "model": state, "calibration": recipe})
        state = {"protocol": protocol, "models": models, "calibration": recipe}
        save_json(run / "prepared.json", state)
        return state


def validate_prepared(cfg):
    source, _, frozen = source_model(cfg)
    state = read_json(Path(cfg["run_dir"]) / "prepared.json")
    protocol = state["protocol"]
    if (protocol["settings"] != cfg or protocol["source"] != frozen
            or protocol["code_signature"] != code_signature() or protocol["runtime"] != ort.__version__):
        raise ValueError("Kết quả lượng tử hoá khác nguồn/code/runtime hiện tại")
    val = read_split(source["dataset"], "val", "det")
    if protocol["val_signature"] != split_signature(val):
        raise ValueError("Val đã đổi sau khi chọn cấu hình")
    for name in VARIANTS:
        if state["models"][name]["artifacts"] != model_hashes(Path(cfg["run_dir"]) / name):
            raise ValueError("Model đã đổi sau khi chốt")
    return source, state


def measure_in_memory(pipeline, pixels, warm, params, seed):
    """Cùng vòng đo notebook 03: chỉ suy luận và giữ số đo trong RAM."""
    for _ in range(params["warmup"]):
        pipeline(warm)
    elapsed = [[] for _ in pixels]
    predictions = [None] * len(pixels)
    for repeat in range(params["repeats"]):
        order = list(range(len(pixels)))
        random.Random(seed + repeat).shuffle(order)
        for index in order:
            start = time.perf_counter_ns()
            prediction = pipeline(pixels[index])
            duration = (time.perf_counter_ns() - start) / 1_000_000
            text = prediction["text"]
            if predictions[index] is not None and predictions[index] != text:
                raise RuntimeError("Dự đoán thay đổi giữa các lần chạy cùng model")
            predictions[index] = text
            elapsed[index].append(duration)
    return predictions, elapsed


def benchmark(cfg, name):
    source, state = validate_prepared(cfg)
    run = Path(cfg["run_dir"]) / name
    rows = read_split(source["dataset"], "test", "det")
    val = read_split(source["dataset"], "val", "det")
    protocol = {"prepared": state["protocol"], "model": state["models"][name],
                "test_signature": split_signature(rows), "platform": platform.platform(),
                "processor": platform.processor(), "runtime": ort.__version__,
                "measurement_protocol": MEASUREMENT_PROTOCOL}
    with model_lock(run):
        pixels = [image(row["path"]) for row in rows]
        cv2.setNumThreads(1)
        pipe = onnx_pipeline(source, run)
        warm = image(val[0]["path"])
        predictions, elapsed = measure_in_memory(pipe, pixels, warm, cfg, cfg["seed"])
        scores = text_scores(predictions, [r["text"] for r in rows])
        scores.update(speed_scores([t for times in elapsed for t in times]))
        result = {"Model": LABELS[name], **scores, "protocol": protocol,
                  "Size_MB": state["models"][name]["total_bytes"] / 1e6,
                  "scope": state["models"][name]["scope"], "threads": cfg["threads"],
                  "opencv_threads": 1, "provider": "CPUExecutionProvider", "batch_size": 1,
                  "batch_unit": "plate", "rec_batching": "all_lines",
                  "timing_scope": "preprocess+DET+crop+REC+decode; excludes disk IO/model load"}
        save_json(run / "test_predictions.json", [
            {"image": row["relative"], "GT": row["text"], "prediction": text, "times_ms": times}
            for row, text, times in zip(rows, predictions, elapsed)])
        save_json(run / "test_result.json", result)
        print(name, scores, flush=True)
        return result


def table(cfg=None):
    cfg = cfg or settings()
    source, state = validate_prepared(cfg)
    test = read_split(source["dataset"], "test", "det")
    results = []
    for name in VARIANTS:
        directory = Path(cfg["run_dir"]) / name
        result = read_json(directory / "test_result.json")
        details = read_json(directory / "test_predictions.json")
        if (result["protocol"]["prepared"] != state["protocol"]
                or result["protocol"]["test_signature"] != split_signature(test)
                or result["protocol"].get("measurement_protocol") != MEASUREMENT_PROTOCOL
                or result["protocol"]["model"] != state["models"][name]
                or result["Size_MB"] != state["models"][name]["total_bytes"] / 1e6):
            raise ValueError("Kết quả khác model/test đã chốt")
        if [(r["image"], r["GT"]) for r in details] != [(r["relative"], r["text"]) for r in test]:
            raise ValueError("Danh sách test khác số đo")
        if any(len(r["times_ms"]) != cfg["repeats"] for r in details):
            raise ValueError("Chưa đủ lượt đo")
        scores = text_scores([r["prediction"] for r in details], [r["GT"] for r in details])
        scores.update(speed_scores([t for r in details for t in r["times_ms"]]))
        if any(result[k] != v for k, v in scores.items()):
            raise ValueError("Chỉ số khác số đo gốc")
        results.append(result)
    for key in ("test_signature", "platform", "processor", "runtime"):
        if len({r["protocol"][key] for r in results}) != 1:
            raise ValueError("Điều kiện benchmark khác nhau")
    slowest = max(r["Speed"] for r in results)
    fields = ("Rank", "Model", "ACC", "CER", "Speed", "FPS", "Speedup", "Size_MB")
    ranked = [{"Rank": i, **{k: r[k] for k in ("Model", "ACC", "CER", "Speed", "FPS", "Size_MB")},
               "Speedup": slowest / r["Speed"]}
              for i, r in enumerate(sorted(results, key=lambda r: (-r["ACC"], r["CER"], r["Speed"])), 1)]
    target = Path(cfg["run_dir"]) / "comparison.csv"
    with target.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ranked)
    text = "| Rank | Model | ACC (%) ↑ | CER (%) ↓ | Speed (ms/ảnh) ↓ | FPS ↑ | Speedup (×) | Size (MB) ↓ |\n"
    text += "|---:|---|---:|---:|---:|---:|---:|---:|\n"
    text += "\n".join("| " + " | ".join(str(r[k]) if k in ("Rank", "Model") else f"{r[k]:.2f}" for k in fields) + " |" for r in ranked)
    (target.with_suffix(".md")).write_text(text + "\n", encoding="utf-8")
    return ranked, text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "test", "table", "all"))
    args = parser.parse_args()
    cfg = settings()
    if args.action in ("prepare", "all"):
        prepare(cfg)
    if args.action in ("test", "all"):
        for name in VARIANTS:
            benchmark(cfg, name)
    if args.action in ("table", "all"):
        print(table(cfg)[1])


if __name__ == "__main__":
    main()

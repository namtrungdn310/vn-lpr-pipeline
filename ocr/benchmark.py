"""Một lượt đo chung cho 14 OCR và hai bản lượng tử hoá của VietLPR V2."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time

import cv2
import onnxruntime as ort
import psutil

from .config import ROOT, save_json
from .data import image, read_split, split_signature
from .metrics import speed_scores, text_scores
from .paddle_suite import digest
from .paddle_suite_state import model_lock
from . import paddle_suite, public_vn, quantization
from .evaluate import deployed_config, validate_export
from .runtime import inference_signature, onnx_pipeline

CONFIG = ROOT / "configs/ocr/benchmark.json"
POINTER = ROOT / "runs/ocr_benchmark/latest.json"
QUANT_NAMES = {"VietLPR_v2_INT8": "int8", "VietLPR_v2_INT4": "int4"}
QUANT_TABLE = ("VietLPR_v2", *QUANT_NAMES)
PROTOCOL = "all_ocr_in_memory_rotated_rounds_v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def settings():
    cfg = read_json(CONFIG)
    for key in ("dataset", "run_dir"):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    if min(cfg[k] for k in ("threads", "warmup", "rounds")) < 1 or cfg["rounds"] < 2:
        raise ValueError("Cần ít nhất hai lượt độc lập để tính độ lệch chuẩn")
    return cfg


def names():
    return ["VietLPR_v1", "VietLPR_v2", *paddle_suite.MODEL_NAMES,
            *(v["name"] for v in public_vn.load_catalog()["variants"]), *QUANT_NAMES]


def round_order(models, round_index, seed):
    result = list(models)
    random.Random(seed + round_index).shuffle(result)
    return result


def code_signature():
    files = [Path(__file__), CONFIG, ROOT / "ocr/metrics.py"]
    return hashlib.sha256(b"".join(p.read_bytes() for p in files)
                          + quantization.code_signature().encode()
                          + paddle_suite.code_signature().encode()
                          + public_vn.code_signature().encode()).hexdigest()


def power_context():
    battery = psutil.sensors_battery()
    scheme = None
    if os.name == "nt":
        response = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, check=True)
        scheme = response.stdout.strip()
    return {"ac": battery.power_plugged if battery else None, "power_scheme": scheme}


def check_power(expected, cfg):
    current = power_context()
    if cfg["require_ac"] and current["ac"] is False:
        raise RuntimeError("Cần cắm sạc để benchmark; lượt đang đo chưa được chấp nhận")
    if current["power_scheme"] != expected["power_scheme"]:
        raise RuntimeError("Chế độ nguồn đã đổi; giữ cùng chế độ trước khi tiếp tục")
    return current


def environment():
    return {"platform": platform.platform(), "processor": platform.processor(),
            "runtime": ort.__version__, "opencv": cv2.__version__,
            "python": platform.python_version(), "cpu_logical": psutil.cpu_count(),
            "cpu_physical": psutil.cpu_count(logical=False)}


def load_model(name, cfg, val, create=True):
    """Kiểm nguồn đã chốt rồi dùng lại đúng pipeline, không train/tune/export."""
    threads = cfg["threads"]
    if name in ("VietLPR_v1", "VietLPR_v2"):
        source = deployed_config(name.lower())
        validate_export(source)
        folder = Path(source["run_dir"]) / "onnx"
        frozen = read_json(folder / "frozen.json")
        if (frozen["inference_signature"] != inference_signature()
                or frozen["val_signature"] != split_signature(val)
                or any(digest(folder / p) != h for p, h in frozen["files"].items())):
            raise ValueError("Model hoặc val VietLPR đã đổi")
        source["benchmark"]["threads"] = threads
        state = {"frozen": frozen, "source_config_sha256": digest(folder / "config.json"),
                 "source_code": inference_signature(), "precision": "FP32",
                 "size_bytes": sum((folder / f"{t}.onnx").stat().st_size for t in ("det", "rec"))}
        return onnx_pipeline(source) if create else None, state
    if name in paddle_suite.MODEL_NAMES:
        from .paddle_suite_runtime import PaddlePipeline

        catalog = paddle_suite.load_catalog()
        frozen = paddle_suite.validate_frozen(catalog, name, val)
        folder = paddle_suite.run_dir(catalog, name)
        state = {"frozen": frozen, "source_code": paddle_suite.code_signature(), "precision": "FP32",
                 "size_bytes": sum((folder / "onnx" / f"{t}.onnx").stat().st_size for t in ("det", "rec"))}
        return (PaddlePipeline(folder, catalog["det_resize"], threads, frozen["postprocess"])
                if create else None), state
    if name in QUANT_NAMES:
        qcfg = quantization.settings()
        source, prepared = quantization.validate_prepared(qcfg)
        variant = QUANT_NAMES[name]
        folder = Path(qcfg["run_dir"]) / variant
        source["benchmark"]["threads"] = threads
        state = {"frozen": prepared["models"][variant], "preparation": prepared["protocol"],
                 "source_code": quantization.code_signature(), "precision": variant.upper(),
                 "scope": prepared["models"][variant]["scope"],
                 "size_bytes": prepared["models"][variant]["total_bytes"]}
        return onnx_pipeline(source, folder) if create else None, state
    from .public_vn_runtime import PublicVNPipeline

    catalog = public_vn.load_catalog()
    frozen = public_vn.validate_frozen(catalog, name, val)
    folder = public_vn.run_dir(catalog, name)
    state = {"frozen": frozen, "source_code": public_vn.code_signature(), "precision": "FP32",
             "size_bytes": (folder / "onnx/model.onnx").stat().st_size}
    return PublicVNPipeline(folder, public_vn.variant(catalog, name), threads) if create else None, state


def baseline_path(name):
    if name in ("VietLPR_v1", "VietLPR_v2"):
        return ROOT / "runs" / name.lower()
    if name in paddle_suite.MODEL_NAMES:
        return ROOT / "runs/paddle_suite" / name
    if name in QUANT_NAMES:
        return Path(quantization.settings()["run_dir"]) / QUANT_NAMES[name]
    return ROOT / "runs/public_vn_ocr" / name


def safe_run_path(path):
    path = Path(path).resolve()
    base = Path(settings()["run_dir"]).resolve()
    if path == base or not path.is_relative_to(base):
        raise ValueError("Lượt đo phải nằm trong runs/ocr_benchmark/<run>")
    return path


def create_run():
    cfg = settings()
    if len(names()) != 16 or len(set(names())) != 16:
        raise ValueError("Cần đúng 16 cấu hình độc lập")
    power = power_context()
    check_power(power, cfg)
    rows, val = (read_split(cfg["dataset"], split, "det") for split in ("test", "val"))
    run = Path(cfg["run_dir"]) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run.mkdir(parents=True, exist_ok=False)
    snapshots = {}
    manifest = {"version": PROTOCOL, "created_utc": now(), "config": cfg,
                "code_signature": code_signature(), "environment": environment(), "power": power,
                "test_signature": split_signature(rows), "val_signature": split_signature(val),
                "models": {}, "orders": [round_order(names(), r, cfg["seed"]) for r in range(cfg["rounds"])],
                "timing_scope": cfg["scope"], "training_performed": False, "calibration_performed": False}
    for name in names():
        _, manifest["models"][name] = load_model(name, cfg, val, create=False)
        folder = baseline_path(name)
        for filename in ("test_result.json", "test_predictions.json"):
            source = folder / filename
            target = run / "previous" / name / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            snapshots[str(source)] = digest(source)
    for source in (ROOT / "notebooks/03_evaluate_models.ipynb", ROOT / "notebooks/06_quantize_vietlpr_v2.ipynb",
                   ROOT / "runs/comparison_ranked.csv", Path(quantization.settings()["run_dir"]) / "comparison.csv"):
        target = run / "previous" / source.name
        shutil.copy2(source, target)
    save_json(run / "baseline_snapshot.json", snapshots)
    save_json(run / "manifest.json", manifest)
    save_json(run / "state.json", {"status": "ready", "completed_passes": 0, "total_passes": 16 * cfg["rounds"]})
    save_json(POINTER, {"run_dir": str(run), "protocol": PROTOCOL})
    print("Lượt đo:", run, flush=True)
    return run


def manifest_for(run):
    run = safe_run_path(run)
    m = read_json(run / "manifest.json")
    if (m["version"] != PROTOCOL or m["config"] != settings()
            or m["code_signature"] != code_signature() or m["environment"] != environment()):
        raise ValueError("Lượt đo khác code/cấu hình/môi trường; không trộn kết quả")
    return m


def round_path(run, name, index):
    if name not in names() or index not in range(settings()["rounds"]):
        raise ValueError("Model/lượt đo không hợp lệ")
    return Path(run) / "passes" / name / f"round_{index + 1:02d}.json"


def validate_pass(record, m, name, index, rows):
    if (record["Model"] != name or record["round"] != index or record["manifest_code"] != m["code_signature"]
            or record["test_signature"] != m["test_signature"] or record["model"] != m["models"][name]):
        raise ValueError("Số đo không thuộc lượt/model đã chốt")
    details = record["details"]
    if [(d["image"], d["GT"]) for d in details] != [(r["relative"], r["text"]) for r in rows]:
        raise ValueError("Danh sách ảnh hoặc nhãn khác test")
    score = text_scores([d["prediction"] for d in details], [d["GT"] for d in details])
    score.update(speed_scores([d["time_ms"] for d in details]))
    if any(not math.isclose(record["scores"][k], v, rel_tol=1e-12, abs_tol=1e-12) for k, v in score.items()):
        raise ValueError("Chỉ số không khớp số đo gốc")
    return score


def worker(run, name, index):
    m = manifest_for(run)
    cfg = m["config"]
    rows, val = (read_split(cfg["dataset"], split, "det") for split in ("test", "val"))
    if split_signature(rows) != m["test_signature"] or split_signature(val) != m["val_signature"]:
        raise ValueError("Dữ liệu thay đổi sau khi chốt lượt đo")
    path = round_path(run, name, index)
    if path.exists():
        validate_pass(read_json(path), m, name, index, rows)
        print(name, index + 1, "đã đo đủ, bỏ qua", flush=True)
        return
    check_power(m["power"], cfg)
    pixels = [image(r["path"]) for r in rows]
    warm = image(val[0]["path"])
    cv2.setNumThreads(cfg["opencv_threads"])
    pipe, state = load_model(name, cfg, val)
    if state != m["models"][name]:
        raise ValueError("Model hoặc ngưỡng khác snapshot đã chốt")
    before = {"utc": now(), "power": check_power(m["power"], cfg),
              "cpu_percent": psutil.cpu_percent(interval=1), "available_memory": psutil.virtual_memory().available}
    print(f"{name}: lượt {index + 1}/{cfg['rounds']}, warm-up {cfg['warmup']}, {len(rows)} ảnh...", flush=True)
    start = time.perf_counter()
    predicted, elapsed = quantization.measure_in_memory(pipe, pixels, warm,
        {"warmup": cfg["warmup"], "repeats": 1}, cfg["seed"] + index)
    duration = time.perf_counter() - start
    after = {"utc": now(), "power": check_power(m["power"], cfg),
             "available_memory": psutil.virtual_memory().available}
    details = [{"image": r["relative"], "GT": r["text"], "prediction": p, "time_ms": t[0]}
               for r, p, t in zip(rows, predicted, elapsed)]
    scores = text_scores(predicted, [r["text"] for r in rows])
    scores.update(speed_scores([t[0] for t in elapsed]))
    record = {"Model": name, "round": index, "manifest_code": m["code_signature"],
              "test_signature": m["test_signature"], "model": state, "scores": scores,
              "details": details, "before": before, "after": after, "pass_wall_seconds": duration}
    validate_pass(record, m, name, index, rows)
    save_json(path, record)
    print(name, f"lượt {index + 1}: Speed={scores['Speed']:.4f} ms, ACC={scores['ACC']:.2f}, CER={scores['CER']:.2f}", flush=True)


def aggregate_passes(records, m, name, rows):
    if len(records) != m["config"]["rounds"]:
        raise ValueError("Chưa đủ lượt, không tạo kết quả cuối")
    for index, r in enumerate(records):
        validate_pass(r, m, name, index, rows)
    predictions = [d["prediction"] for d in records[0]["details"]]
    if any([d["prediction"] for d in r["details"]] != predictions for r in records[1:]):
        raise ValueError("Dự đoán thay đổi giữa các lượt")
    times = [[r["details"][i]["time_ms"] for r in records] for i in range(len(rows))]
    means = [r["scores"]["Speed"] for r in records]
    scores = text_scores(predictions, [r["text"] for r in rows])
    scores.update(speed_scores([v for row in times for v in row]))
    scores["Speed_std"] = statistics.stdev(means)
    return {"Model": name, **scores, "round_mean_ms": means,
            "Size_MB": m["models"][name]["size_bytes"] / 1e6,
            "precision": m["models"][name]["precision"], "test_signature": m["test_signature"],
            "measurement_protocol": PROTOCOL, "manifest_code": m["code_signature"], "details": [
                {"image": r["relative"], "GT": r["text"], "prediction": p, "times_ms": t}
                for r, p, t in zip(rows, predictions, times)]}


def verify_assets(m):
    val = read_split(m["config"]["dataset"], "val", "det")
    for name in m["models"]:
        _, current = load_model(name, m["config"], val, create=False)
        if current != m["models"][name]:
            raise ValueError("Model/ngưỡng thay đổi sau khi đo")


def collect(run):
    m = manifest_for(run)
    rows = read_split(m["config"]["dataset"], "test", "det")
    if split_signature(rows) != m["test_signature"]:
        raise ValueError("Test thay đổi")
    verify_assets(m)
    results = {}
    for name in m["models"]:
        records = [read_json(round_path(run, name, i)) for i in range(m["config"]["rounds"])]
        results[name] = aggregate_passes(records, m, name, rows)
    return m, results


def make_table(results, selected, with_size=False):
    chosen = [results[name] for name in selected]
    slowest = max(r["Speed"] for r in chosen)
    ranked = [{"Rank": i, **{k: r[k] for k in ("Model", "ACC", "CER", "Speed", "Speed_std", "FPS")},
               "Speedup": slowest / r["Speed"], **({"Size_MB": r["Size_MB"]} if with_size else {})}
              for i, r in enumerate(sorted(chosen, key=lambda r: (-r["ACC"], r["CER"], r["Speed"])), 1)]
    columns = ["Rank", "Model", "ACC", "CER", "Speed", "Speed_std", "FPS", "Speedup"] + (["Size_MB"] if with_size else [])
    header = "| Rank | Model | ACC (%) ↑ | CER (%) ↓ | Speed ± SD (ms/ảnh) ↓ | FPS ↑ | Speedup (×) ↑ |" + (" Size (MB) ↓ |" if with_size else "")
    text = header + "\n|---:|---|---:|---:|---:|---:|---:|" + ("---:|" if with_size else "")
    for r in ranked:
        numbers = [str(r["Rank"]), r["Model"], f"{r['ACC']:.2f}", f"{r['CER']:.2f}",
                   f"{r['Speed']:.2f} ± {r['Speed_std']:.2f}", f"{r['FPS']:.2f}", f"{r['Speedup']:.2f}"]
        if with_size:
            numbers.append(f"{r['Size_MB']:.2f}")
        text += "\n| " + " | ".join(numbers) + " |"
    return ranked, columns, text


def save_csv(path, rows, columns):
    import csv

    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def publish(run):
    m, results = collect(run)
    run = Path(run)
    for name, result in results.items():
        folder = run / "models" / name
        details = result.pop("details")
        save_json(folder / "test_result.json", result)
        save_json(folder / "test_predictions.json", details)
    for group, selected in (("ocr", [n for n in names() if n not in QUANT_NAMES]), ("quantization", QUANT_TABLE)):
        rows, columns, text = make_table(results, selected, with_size=group == "quantization")
        save_csv(run / f"comparison_{group}.csv", rows, columns)
        (run / f"comparison_{group}.md").write_text(text + "\n", encoding="utf-8")
    snapshot = read_json(run / "baseline_snapshot.json")
    if any(digest(p) != expected for p, expected in snapshot.items()):
        raise ValueError("Kết quả gốc đã thay đổi trong lúc benchmark")
    save_json(run / "verification.json", {"baseline_files_unchanged": len(snapshot), "model_count": len(results),
              "rounds": m["config"]["rounds"], "samples_per_round": len(read_split(m["config"]["dataset"], "test", "det")),
              "shared_fp32_record": str(run / "models/VietLPR_v2/test_result.json"),
              "std_definition": "sample standard deviation of per-round mean latency (ddof=1)",
              "file_io_in_timed_pass": False, "training_performed": False, "calibration_performed": False})
    refresh_notebooks(run)


def resolve_run(path=None):
    return safe_run_path(path or read_json(POINTER)["run_dir"])


def display_table(group="ocr", run=None):
    run = resolve_run(run)
    state = read_json(run / "state.json")
    if state["status"] != "complete":
        return f"Đang đo: {state.get('completed_passes', 0)}/{state.get('total_passes', 80)} lượt model. " \
               f"Hiện tại: {state.get('model', '—')}, vòng {state.get('round', 0) + 1}. Chưa có bảng cuối."
    _, results = collect(run)
    selected = QUANT_TABLE if group == "quantization" else [n for n in names() if n not in QUANT_NAMES]
    return make_table(results, selected, with_size=group == "quantization")[2]


def refresh_notebooks(run):
    import nbformat

    for file, group in (("03_evaluate_models.ipynb", "ocr"), ("06_quantize_vietlpr_v2.ipynb", "quantization")):
        p = ROOT / "notebooks" / file
        book = nbformat.read(p, as_version=4)
        text = (Path(run) / f"comparison_{group}.md").read_text(encoding="utf-8")
        book.cells[-1].outputs = [nbformat.v4.new_output("display_data", data={"text/markdown": text, "text/plain": text}),
                                  nbformat.v4.new_output("stream", name="stdout", text=f"Run: {run}\n")]
        book.cells[-1].execution_count = 1
        nbformat.validate(book)
        nbformat.write(book, p)


def run_suite(run=None):
    cfg = settings()
    base = Path(cfg["run_dir"])
    with model_lock(base):
        run = resolve_run(run) if run else create_run()
        m = manifest_for(run)
        state = {"status": "running", "started_utc": now(), "pid": os.getpid(),
                 "completed_passes": 0, "total_passes": len(m["models"]) * cfg["rounds"], "run_dir": str(run)}
        try:
            for index, order in enumerate(m["orders"]):
                for name in order:
                    state.update(model=name, round=index)
                    save_json(run / "state.json", state)
                    check_power(m["power"], cfg)
                    command = [sys.executable, "-u", "-B", "-m", "ocr.benchmark", "worker",
                               "--run-dir", str(run), "--model", name, "--round", str(index)]
                    env = os.environ.copy()
                    env.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
                    subprocess.run(command, cwd=ROOT, env=env, check=True)
                    state["completed_passes"] += 1
                    save_json(run / "state.json", state)
            publish(run)
            state.update(status="complete", finished_utc=now())
        except BaseException as error:
            state.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            save_json(run / "state.json", state)
        print("Đã hoàn tất hai bảng:", run, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "worker", "status", "table"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--model", choices=names())
    parser.add_argument("--round", type=int)
    parser.add_argument("--group", choices=("ocr", "quantization"), default="ocr")
    args = parser.parse_args()
    if args.action == "run":
        run_suite(args.run_dir)
    elif args.action == "worker":
        if args.run_dir is None or args.model is None or args.round is None:
            parser.error("worker cần --run-dir, --model, --round")
        worker(args.run_dir, args.model, args.round)
    elif args.action == "status":
        print(json.dumps(read_json(resolve_run(args.run_dir) / "state.json"), ensure_ascii=False, indent=2))
    else:
        print(display_table(args.group, args.run_dir))


if __name__ == "__main__":
    main()

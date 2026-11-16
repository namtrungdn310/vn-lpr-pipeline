"""Đo lại OCR và detector bằng cùng protocol; FP32 dùng chung các notebook."""

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import cv2
import psutil
import onnxruntime as ort

from . import benchmark as baseline
from . import int8_optimization as optimization
from . import int8_refinement as refinement
from . import yolo_int8_refinement as yolo_refinement
from . import cpu_protocol as cpu
from . import cpu_selection
from . import quantization
from .config import ROOT, save_json
from .data import image, read_split, split_signature
from .metrics import speed_scores, text_scores
from .paddle_suite_state import model_lock
from .runtime import onnx_pipeline
from plate_detection import quantization as yq
from plate_detection.data import load_config, read_image, sha256, dataset_signature
from plate_detection.evaluate import split_rows, operating_scores

POINTER = ROOT / "runs/int8_optimization/latest_benchmark.json"
BASE = ROOT / "runs/int8_optimization/benchmark"
NEW_OCR = "VietLPR_v2_INT8_optimized"
YOLO = {"YOLO26n-OBB_FP32": "fp32", "YOLO26n-OBB_INT8_previous": "int8",
        "YOLO26n-OBB_INT4_mixed": "int4", "YOLO26n-OBB_INT8_optimized": "optimized"}
PROTOCOL = "shared_ocr_detection_fixed_physical_cpu_v3"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def names():
    return [*baseline.names(), NEW_OCR, *YOLO]


def groups():
    return {"ocr": [n for n in baseline.names() if n not in baseline.QUANT_NAMES],
            "quantization": [*baseline.QUANT_TABLE, NEW_OCR], "detection": list(YOLO)}


def signature():
    return hashlib.sha256(Path(__file__).read_bytes() + optimization.signature().encode()
                          + refinement.signature().encode() + yolo_refinement.signature().encode()
                          + cpu_selection.signature().encode()
                          + baseline.code_signature().encode()).hexdigest()


def safe_run(path=None):
    run = Path(path or read_json(POINTER)["run_dir"]).resolve()
    if not run.is_relative_to(BASE.resolve()) or run == BASE.resolve():
        raise ValueError("Lượt đo phải nằm trong runs/int8_optimization/benchmark/<run>")
    return run


def assets(name, create=False):
    cfg = baseline.settings()
    if name == NEW_OCR:
        selected = cpu_selection.validate_selected("vietlpr")
        source, _, _ = quantization.source_model(quantization.settings())
        state = {"selection": selected, "precision": "INT8 mixed" if selected["scope"].get("quantized_part") else "INT8",
                 "size_bytes": sum((Path(selected["folder"]) / filename).stat().st_size for filename in selected["artifacts"])}
        return onnx_pipeline(source, selected["folder"]) if create else None, state
    if name in YOLO:
        source = load_config()
        selected = cpu_selection.validate_selected("yolo")
        variant = YOLO[name]
        if variant == "optimized":
            path = Path(selected["folder"]) / "model.onnx"
            freeze = selected
        else:
            prepared = read_json(Path(yq.settings()["run_dir"]) / "prepared.json")
            if (prepared["protocol"]["code_signature"] != yq.code_signature()
                    or prepared["protocol"]["source_config"] != source):
                raise ValueError("Detector lịch sử khác code/cấu hình đã chốt")
            path = Path(yq.settings()["run_dir"]) / variant / "model.onnx"
            freeze = prepared["models"][variant]
            if yq.inventory(path) != freeze["graph"]:
                raise ValueError("Detector lịch sử khác SHA/graph đã chốt")
        precision = ("INT8 mixed" if selected["scope"].get("mixed") or selected["scope"].get("exclude_head")
                     else "INT8") if variant == "optimized" else {"fp32": "FP32", "int8": "INT8", "int4": "INT4 mixed"}[variant]
        state = {"path": str(path), "sha256": sha256(path), "frozen": freeze,
                 "precision": precision, "size_bytes": path.stat().st_size, "source_config": source}
        return yq.detector(path, cfg, source) if create else None, state
    val = read_split(cfg["dataset"], "val", "det")
    return baseline.load_model(name, cfg, val, create=create)


def rows_for(name, split="test"):
    if name in YOLO:
        return split_rows(load_config(), split)
    return read_split(baseline.settings()["dataset"], split, "det")


def rows_signature(name, rows):
    if name in YOLO:
        return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    return split_signature(rows)


def snapshots(run):
    """Lưu SHA và bản sao kết quả cũ trước khi cập nhật notebook."""
    files = set()
    for name in baseline.names():
        directory = baseline.baseline_path(name)
        files.update(directory / f for f in ("test_result.json", "test_predictions.json"))
    directory = Path(load_config()["run_dir"])
    files.update(directory / f for f in ("weights/best.pt", "weights/last.pt", "onnx/model.onnx", "detection_result.json", "detection_predictions.json"))
    for variant in ("fp32", "int8", "int4"):
        folder = Path(yq.settings()["run_dir"]) / variant
        files.add(folder / "model.onnx")
    hashes = {str(p): sha256(p) for p in sorted(files)}
    save_json(run / "baseline_snapshot.json", hashes)
    for p in files:
        if p.suffix == ".json":
            dest = run / "previous" / p.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
    for number in ("03", "04", "06", "07"):
        for path in (ROOT / "notebooks").glob(number + "_*.ipynb"):
            dest = run / "previous" / path.name
            shutil.copy2(path, dest)


def create_run():
    cpu_context = cpu.apply()
    cfg = baseline.settings()
    power = baseline.power_context()
    baseline.check_power(power, cfg)
    run = BASE / datetime.now().strftime("%Y%m%d_%H%M%S")
    run.mkdir(parents=True, exist_ok=False)
    manifest = {"protocol": PROTOCOL, "code_signature": signature(), "config": cfg,
                "environment": baseline.environment(), "power": power, "cpu_protocol": cpu_context, "models": {}, "datasets": {},
                "orders": [baseline.round_order(names(), i, cfg["seed"]) for i in range(cfg["rounds"])],
                "file_io_in_timed_pass": False, "training_performed": False}
    for name in names():
        _, manifest["models"][name] = assets(name)
        manifest["datasets"][name] = {split: rows_signature(name, rows_for(name, split)) for split in ("val", "test")}
    manifest["detector_dataset_signature"] = dataset_signature(load_config()["dataset"])
    snapshots(run)
    save_json(run / "manifest.json", manifest)
    save_json(run / "state.json", {"status": "ready", "completed_passes": 0, "total_passes": len(names()) * cfg["rounds"]})
    # Không đổi bảng đang hiển thị cho tới khi đo hoàn chỉnh.
    save_json(run.parent.parent / "active_benchmark.json", {"run_dir": str(run)})
    return run


def manifest_for(run):
    run = safe_run(run)
    m = read_json(run / "manifest.json")
    if (m["protocol"] != PROTOCOL or m["code_signature"] != signature()
            or m["config"] != baseline.settings() or m["environment"] != baseline.environment()
            or list(m["models"]) != names()):
        raise ValueError("Code/cấu hình/môi trường khác lượt đã chốt")
    return m


def pass_path(run, name, index):
    if name not in names() or index not in range(baseline.settings()["rounds"]):
        raise ValueError("Sai model hoặc lượt")
    return Path(run) / "passes" / name / f"round_{index + 1:02d}.json"


def validate_pass(r, m, name, index):
    if (r["Model"] != name or r["round"] != index or r["manifest_code"] != m["code_signature"]
            or r["model"] != m["models"][name] or r["test_signature"] != m["datasets"][name]["test"]):
        raise ValueError("Số đo khác model/manifest")
    rows = rows_for(name)
    if rows_signature(name, rows) != r["test_signature"]:
        raise ValueError("Dataset thay đổi")
    details = r["details"]
    if [d["image"] for d in details] != [row["image"] if name in YOLO else row["relative"] for row in rows]:
        raise ValueError("Danh sách test không khớp")
    times = [d["time_ms"] for d in details]
    if any(not math.isfinite(t) or t <= 0 for t in times):
        raise ValueError("Thời gian không hợp lệ")
    if name in YOLO:
        actual = yq.ground_truth(rows, [read_image(row["image"]) for row in rows])
        if actual != [d["GT"] for d in details]:
            raise ValueError("Nhãn GT detector khác dataset")
        scores = operating_scores([d["prediction"] for d in details], actual, .5)
    else:
        if [d["GT"] for d in details] != [row["text"] for row in rows]:
            raise ValueError("Nhãn OCR khác dataset")
        scores = text_scores([d["prediction"] for d in details], [row["text"] for row in rows])
    scores.update(speed_scores(times))
    if set(scores) != set(r["scores"]) or any(not math.isclose(scores[k], v, abs_tol=1e-10, rel_tol=1e-10) for k, v in r["scores"].items()):
        raise ValueError("Chỉ số không khớp dữ liệu gốc")
    if r["before"]["power"] != m["power"] or r["after"]["power"] != m["power"]:
        raise ValueError("Điều kiện nguồn đã thay đổi")
    if r.get("cpu_protocol") != m.get("cpu_protocol"):
        raise ValueError("CPU affinity khác manifest")
    return scores


def worker(run, name, index):
    cpu.apply(read_json(Path(run) / "manifest.json")["cpu_protocol"])
    m = manifest_for(run)
    path = pass_path(run, name, index)
    if path.exists():
        validate_pass(read_json(path), m, name, index)
        return
    cfg = m["config"]
    baseline.check_power(m["power"], cfg)
    rows, val = rows_for(name), rows_for(name, "val")
    if any(rows_signature(name, rs) != m["datasets"][name][s] for s, rs in (("test", rows), ("val", val))):
        raise ValueError("Dữ liệu đã đổi")
    cv2.setNumThreads(cfg["opencv_threads"])
    pipe, current = assets(name, create=True)
    if current != m["models"][name]:
        raise ValueError("Model đã đổi")
    pixels = [read_image(r["image"]) for r in rows] if name in YOLO else [image(r["path"]) for r in rows]
    warm = read_image(val[0]["image"]) if name in YOLO else image(val[0]["path"])
    before = {"utc": baseline.now(), "power": baseline.check_power(m["power"], cfg),
              "cpu_percent": psutil.cpu_percent(interval=1), "available_memory": psutil.virtual_memory().available}
    if name in YOLO:
        # Cùng clock và phạm vi pipeline; GT luôn ngoài vùng đo.
        actual = yq.ground_truth(rows, pixels)
        predictions, times = yq.measure_in_memory(pipe, pixels, warm,
            {"warmup": cfg["warmup"], "seed": cfg["seed"]}, index)
        scores = operating_scores(predictions, actual, .5)
        details = [{"image": r["image"], "GT": gt, "prediction": p, "time_ms": t}
                   for r, gt, p, t in zip(rows, actual, predictions, times, strict=True)]
    else:
        predictions, elapsed = quantization.measure_in_memory(pipe, pixels, warm,
            {"warmup": cfg["warmup"], "repeats": 1}, cfg["seed"] + index)
        times = [t[0] for t in elapsed]
        scores = text_scores(predictions, [r["text"] for r in rows])
        details = [{"image": r["relative"], "GT": r["text"], "prediction": p, "time_ms": t}
                   for r, p, t in zip(rows, predictions, times, strict=True)]
    after = {"utc": baseline.now(), "power": baseline.check_power(m["power"], cfg)}
    cpu.check(m["cpu_protocol"])
    scores.update(speed_scores(times))
    record = {"Model": name, "round": index, "model": current, "test_signature": m["datasets"][name]["test"],
              "manifest_code": m["code_signature"], "details": details, "scores": scores, "before": before, "after": after,
              "cpu_protocol": m["cpu_protocol"]}
    validate_pass(record, m, name, index)
    save_json(path, record)
    print(name, f"{index + 1}/{cfg['rounds']}: {scores['Speed']:.4f} ms/ảnh", flush=True)


def aggregate(records, m, name):
    if len(records) != m["config"]["rounds"]:
        raise ValueError("Chưa đủ lượt")
    scores = [validate_pass(r, m, name, i) for i, r in enumerate(records)]
    predictions = [d["prediction"] for d in records[0]["details"]]
    if any([d["prediction"] for d in r["details"]] != predictions for r in records[1:]):
        raise ValueError("Dự đoán thay đổi giữa lượt")
    means = [s["Speed"] for s in scores]
    result = {"Model": name, **scores[0], "Speed": statistics.mean(means), "Speed_std": statistics.stdev(means),
              "FPS": 1000 / statistics.mean(means), "Size_MB": m["models"][name]["size_bytes"] / 1e6,
              "round_mean_ms": means, "measurement_protocol": PROTOCOL, "manifest_code": m["code_signature"],
              "test_signature": m["datasets"][name]["test"], "precision": m["models"][name]["precision"],
              "images": len(records[0]["details"]), "rounds": m["config"]["rounds"],
              "threads": m["config"].get("threads", 4), "opencv_threads": m["config"].get("opencv_threads", 1),
              "provider": "CPUExecutionProvider", "batch_size": 1,
              "timing_scope": "letterbox+ONNX OBB+decode" if name in YOLO else "preprocess+OCR pipeline+final text"}
    return result


def detection_table(results):
    chosen = sorted((results[name] for name in YOLO), key=lambda r: (-r["mAP50-95"], -r["Recall"], r["Speed"]))
    slowest = max(r["Speed"] for r in chosen)
    keys = ["Rank", "Model", "mAP50", "mAP50-95", "Precision", "Recall", "Speed", "Speed_std", "FPS", "Speedup", "Size_MB"]
    rows = [{"Rank": i, **{k: r[k] for k in keys if k not in ("Rank", "Speedup")}, "Speedup": slowest / r["Speed"]}
            for i, r in enumerate(chosen, 1)]
    text = "| Rank | Model | mAP50 (%) ↑ | mAP50-95 (%) ↑ | Precision (%) ↑ | Recall (%) ↑ | Speed ± SD (ms/ảnh) ↓ | FPS ↑ | Speedup (×) ↑ | Size (MB) ↓ |\n|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    for r in rows:
        values = [str(r["Rank"]), r["Model"], *(f"{r[k]:.2f}" for k in ("mAP50", "mAP50-95", "Precision", "Recall")),
                  f"{r['Speed']:.2f} ± {r['Speed_std']:.2f}", *(f"{r[k]:.2f}" for k in ("FPS", "Speedup", "Size_MB"))]
        text += "\n| " + " | ".join(values) + " |"
    return rows, keys, text


def collect(run):
    m = manifest_for(run)
    results = {}
    for name in names():
        _, current = assets(name)
        if current != m["models"][name]:
            raise ValueError("Model khác manifest")
        records = [read_json(pass_path(run, name, i)) for i in range(m["config"]["rounds"])]
        result = aggregate(records, m, name)
        if name in YOLO:
            ap = read_json(Path(run) / "ap" / (name + ".json"))
            if (ap["protocol"]["model_sha256"] != current["sha256"] or ap["protocol"]["split"] != "test"
                    or ap["protocol"]["dataset_signature"] != m["detector_dataset_signature"]
                    or ap["protocol"]["code_signature"] != yq.code_signature()):
                raise ValueError("mAP không khớp model/test")
            if ap.get("evaluation_runtime") != {"cpu_protocol": m["cpu_protocol"], "threads": 4, "inter_threads": 1}:
                raise ValueError("Evaluator mAP khác thiết lập CPU chung")
            result.update({k: ap[k] for k in ("mAP50", "mAP50-95")})
        results[name] = result
    return m, results


def publish(run):
    m, results = collect(run)
    if dataset_signature(load_config()["dataset"]) != m["detector_dataset_signature"]:
        raise ValueError("Dataset detector thay đổi trong lượt đo")
    for name, result in results.items():
        save_json(Path(run) / "models" / name / "test_result.json", result)
    for group, selected in groups().items():
        rows, columns, text = detection_table(results) if group == "detection" else baseline.make_table(results, selected, with_size=group == "quantization")
        text = table_from_results(group, results)
        baseline.save_csv(Path(run) / f"comparison_{group}.csv", rows, columns)
        (Path(run) / f"comparison_{group}.md").write_text(text + "\n", encoding="utf-8")
    (Path(run) / "comparison_detector.md").write_text(table_from_results("detector", results) + "\n", encoding="utf-8")
    snapshot = read_json(Path(run) / "baseline_snapshot.json")
    if any(sha256(path) != expected for path, expected in snapshot.items()):
        raise ValueError("Baseline gốc bị thay đổi")
    save_json(Path(run) / "verification.json", {"baseline_files_unchanged": len(snapshot), "models": len(results),
              "rounds": m["config"]["rounds"], "ocr_images": 124, "detection_images": 258,
              "shared_vietlpr_fp32": str(Path(run) / "models/VietLPR_v2/test_result.json"),
              "shared_yolo_fp32": str(Path(run) / "models/YOLO26n-OBB_FP32/test_result.json"),
              "file_io_in_timed_pass": False, "std_definition": "sample SD of round mean latency (ddof=1)"})
    return results


def refresh_notebooks(run, results=None):
    import nbformat
    if results is None:
        _, results = collect(run)
    for number, group in (("03", "ocr"), ("04", "detector"), ("06", "quantization"), ("07", "detection")):
        path = next((ROOT / "notebooks").glob(number + "_*.ipynb"))
        book = nbformat.read(path, as_version=4)
        text = table_from_results(group, results)
        book.cells[-1].outputs = [nbformat.v4.new_output("display_data", data={"text/markdown": text, "text/plain": text}),
                                  nbformat.v4.new_output("stream", name="stdout", text=f"Run: {run}\n")]
        book.cells[-1].execution_count = None
        nbformat.validate(book)
        nbformat.write(book, path)


def display_table(group="ocr", run=None):
    active = BASE.parent / "active_benchmark.json"
    if run is None and active.is_file():
        pending = safe_run(read_json(active)["run_dir"])
        if read_json(pending / "state.json")["status"] != "complete":
            run = pending
    if run is None and not POINTER.is_file():
        progress = []
        for task in ("vietlpr", "yolo"):
            root = Path(optimization.settings()["run_dir"]) / task
            complete = (root / "refinement/selected.json").is_file()
            count = len(read_json(root / "trials.json")) if (root / "trials.json").is_file() else 0
            progress.append(f"{task}: {'đã chọn model' if complete else str(count) + ' ứng viên đã chấm val'}")
        return "Đang chuẩn bị INT8; chưa có bảng test mới. " + "; ".join(progress) + "."
    run = safe_run(run)
    state = read_json(run / "state.json")
    if state["status"] != "complete":
        return f"Đang đo {state.get('completed_passes', 0)}/{state.get('total_passes', 105)} lượt; chưa có bảng cuối."
    _, results = collect(run)
    return table_from_results(group, results)


def table_from_results(group, results):
    if group == "detector":
        only = {k: v for k, v in results.items() if k == "YOLO26n-OBB_FP32"}
        # Cùng bản ghi FP32 như bảng lượng tử hoá; Speedup của bảng một model bằng 1.
        rows = list(only.values())
        r = rows[0]
        return ("| Model | mAP50 (%) | mAP50-95 (%) | Precision (%) | Recall (%) | Speed ± SD (ms/ảnh) | FPS |\n"
                "|---|---:|---:|---:|---:|---:|---:|\n"
                f"| {r['Model']} | {r['mAP50']:.2f} | {r['mAP50-95']:.2f} | {r['Precision']:.2f} | {r['Recall']:.2f} | {r['Speed']:.2f} ± {r['Speed_std']:.2f} | {r['FPS']:.2f} |")
    if group == "detection":
        return detection_table(results)[2] + "\n\nINT8 optimized giữ các phần nhạy cảm ở FP32 nếu cấu hình được chọn là mixed; phạm vi lưu trong manifest."
    text = baseline.make_table(results, groups()[group], with_size=group == "quantization")[2]
    if group == "quantization":
        text += "\n\nINT8 optimized có thể là INT8 mixed (DET hoặc một phần REC giữ FP32); phạm vi chính xác lưu trong manifest."
    return text


def run_suite(run=None):
    cpu.apply(read_json(Path(run) / "manifest.json")["cpu_protocol"] if run else None)
    with model_lock(ROOT / "runs/ocr_benchmark"), model_lock(BASE), model_lock(Path(yq.settings()["run_dir"])):
        run = safe_run(run) if run else create_run()
        m = manifest_for(run)
        state = {"status": "running", "pid": os.getpid(), "started_utc": baseline.now(), "completed_passes": 0,
                 "total_passes": len(names()) * m["config"]["rounds"]}
        save_json(Path(run) / "state.json", state)
        try:
            # Chỉ chấm test sau khi cả hai model đã chọn xong trên val.
            for name in YOLO:
                output = Path(run) / "ap" / (name + ".json")
                subprocess.run([sys.executable, "-u", "-B", "-m", "ocr.optimized_benchmark", "ap",
                                "--model", name, "--run-dir", str(run)],
                               cwd=ROOT, check=True)
            for index, order in enumerate(m["orders"]):
                for name in order:
                    baseline.check_power(m["power"], m["config"])
                    state.update(model=name, round=index + 1, updated_utc=baseline.now())
                    save_json(Path(run) / "state.json", state)
                    subprocess.run([sys.executable, "-u", "-B", "-m", "ocr.optimized_benchmark", "worker",
                                    "--run-dir", str(run), "--model", name, "--round", str(index)], cwd=ROOT, check=True,
                                   env={**os.environ, "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "4"})
                    state["completed_passes"] += 1
                    save_json(Path(run) / "state.json", state)
            results = publish(run)
            state.update(status="complete", finished_utc=baseline.now())
            save_json(Path(run) / "state.json", state)
            save_json(POINTER, {"run_dir": str(run), "protocol": PROTOCOL})
            refresh_notebooks(run, results)
        except BaseException as error:
            state.update(status="failed", error=str(error), updated_utc=baseline.now())
            save_json(Path(run) / "state.json", state)
            raise
        print("Hoàn tất benchmark chung:", run, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("run", "all", "worker", "table", "ap"))
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--model", choices=names())
    p.add_argument("--round", type=int)
    p.add_argument("--group", choices=("ocr", "quantization", "detection", "detector"), default="ocr")
    args = p.parse_args()
    if args.action == "ap":
        m = manifest_for(args.run_dir)
        cpu.apply(m["cpu_protocol"])
        original = ort.SessionOptions
        def ap_options():
            options = original()
            options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
            return options
        ort.SessionOptions = ap_options
        original_session = ort.InferenceSession
        # AutoBackend của Ultralytics bỏ qua SessionOptions nếu không truyền rõ.
        def ap_session(path, sess_options=None, *positional, **keywords):
            return original_session(path, sess_options if sess_options is not None else ap_options(), *positional, **keywords)
        ort.InferenceSession = ap_session
        if args.model not in YOLO:
            raise ValueError("mAP chỉ dùng cho detector")
        output = Path(args.run_dir) / "ap" / (args.model + ".json")
        result = yq.native_ap(Path(m["models"][args.model]["path"]), "test", output)
        result["evaluation_runtime"] = {"cpu_protocol": m["cpu_protocol"], "threads": 4, "inter_threads": 1}
        save_json(output, result)
    elif args.action == "worker":
        worker(args.run_dir, args.model, args.round)
    elif args.action == "table":
        print(display_table(args.group, args.run_dir))
    else:
        if args.action == "all":
            for task in ("vietlpr", "yolo"):
                optimization.execute("prepare", "--task", task)
            subprocess.run([sys.executable, "-u", "-B", "-m", "ocr.int8_refinement"], cwd=ROOT, check=True)
            subprocess.run([sys.executable, "-u", "-B", "-m", "ocr.yolo_int8_refinement"], cwd=ROOT, check=True)
            for task in ("vietlpr", "yolo"):
                subprocess.run([sys.executable, "-u", "-B", "-m", "ocr.cpu_selection", "prepare", "--task", task], cwd=ROOT, check=True)
        run_suite(args.run_dir)


if __name__ == "__main__":
    main()

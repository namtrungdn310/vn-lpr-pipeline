"""Chuẩn bị, chọn ngưỡng trên val và đo bảy bản PaddleOCR trên CPU."""

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import platform
import random
import time
from urllib.request import urlopen

import cv2
import numpy as np

from .config import ROOT, save_json
from .data import image, read_split, split_signature
from .metrics import speed_scores, text_scores
from .paddle_suite_runtime import PaddlePipeline, Preprocessor, session
from .paddle_suite_state import BenchmarkState, model_lock

CATALOG = ROOT / "configs/ocr/paddle_suite.json"
MODEL_NAMES = (
    "PP-OCRv4_server", "PP-OCRv4_mobile", "PP-OCRv5_server", "PP-OCRv5_mobile",
    "PP-OCRv6_medium", "PP-OCRv6_small", "PP-OCRv6_tiny",
)
TIMING_SCOPE = "preprocess+DET+postprocess+crop+REC+decode; excludes disk IO and model load"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_catalog():
    catalog = read_json(CATALOG)
    if tuple(v["name"] for v in catalog["variants"]) != MODEL_NAMES:
        raise ValueError("Danh sách cần đúng bảy bản chính thức đã chốt")
    for key in ("dataset", "run_dir"):
        catalog[key] = str((ROOT / catalog[key]).resolve())
    if catalog["baseline"] not in MODEL_NAMES:
        raise ValueError("Baseline không nằm trong bộ so sánh")
    return catalog


def run_dir(catalog, name):
    if name not in MODEL_NAMES:
        raise ValueError("Tên model không hợp lệ")
    return Path(catalog["run_dir"]) / name


def profile(catalog, name):
    return {
        "variant": next(v for v in catalog["variants"] if v["name"] == name),
        **{key: catalog[key] for key in ("dataset", "det_resize", "benchmark")},
    }


def download(url, target, expected):
    target = Path(target)
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        pending = target.with_suffix(target.suffix + ".part")
        print("Tải:", url, flush=True)
        with urlopen(url, timeout=90) as response, pending.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if digest(pending) != expected:
            raise ValueError(f"Checksum tải về không khớp: {target.name}")
        pending.replace(target)
    if digest(target) != expected:
        raise ValueError(f"File đã thay đổi: {target}")


def artifact_hashes(directory):
    return {
        relative: digest(Path(directory) / relative)
        for relative in ("onnx/det.onnx", "onnx/rec.onnx", "source/det/inference.yml", "source/rec/inference.yml")
    }


def check_artifacts(directory, expected):
    if artifact_hashes(directory) != expected:
        raise ValueError("Model hoặc dictionary đã đổi; cần chuẩn bị và chọn lại trên val")


def code_signature():
    paths = [Path(__file__).parent / n for n in (
        "paddle_suite.py", "paddle_suite_runtime.py", "paddle_suite_state.py", "data.py", "metrics.py",
    )]
    paths += [ROOT / "third_party/PaddleOCR_suite" / n for n in ("processors.py", "source_manifest.json")]
    return hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()


def prepared_state(catalog, name):
    directory = run_dir(catalog, name)
    state = read_json(directory / "prepared.json")
    if not state.get("ready") or state["profile"] != profile(catalog, name):
        raise ValueError(f"Cấu hình {name} đã đổi; dùng thư mục lượt chạy mới")
    check_artifacts(directory, state["artifacts"])
    for task, model in profile(catalog, name)["variant"]["models"].items():
        if digest(directory / "source" / task / "inference.yml") != model["config"]["sha256"]:
            raise ValueError("YAML không khớp bản chính thức đã chốt")
        if model["format"] == "onnx" and digest(directory / "onnx" / f"{task}.onnx") != model["onnx"]["sha256"]:
            raise ValueError("ONNX không khớp bản chính thức đã chốt")
    return state


def prepare(catalog, name, paddle_python=None):
    from .paddle_pretrained import extract_inference
    from .notebook import run

    directory = run_dir(catalog, name)
    if (directory / "prepared.json").is_file():
        prepared_state(catalog, name)
        print(name, "đã sẵn sàng, dùng lại model đã kiểm tra.", flush=True)
        return
    variant = profile(catalog, name)["variant"]
    for task, model in variant["models"].items():
        source = directory / "source" / task
        download(model["config"]["url"], source / "inference.yml", model["config"]["sha256"])
        if model["format"] == "onnx":
            download(model["onnx"]["url"], directory / "onnx" / f"{task}.onnx", model["onnx"]["sha256"])
        else:
            archive = source / "model.tar"
            download(model["archive_url"], archive, model["archive_sha256"])
            extract_inference(archive, source)
            check_file = directory / "onnx" / f"{task}_conversion.json"
            output = directory / "onnx" / f"{task}.onnx"
            checked = check_file.is_file() and output.is_file() and read_json(check_file)["sha256"] == digest(output)
            if not checked:
                python = paddle_python or ROOT / ".venv-ocr-paddle/Scripts/python.exe"
                run(python, "-m", "ocr.paddle_suite_convert", "--model", name, "--task", task)
    pre = Preprocessor(directory, catalog["det_resize"])
    checks = {}
    cv2.setNumThreads(1)
    for task, model in variant["models"].items():
        engine = session(directory / "onnx" / f"{task}.onnx", catalog["benchmark"]["threads"])
        shapes = []
        for row in read_split(catalog["dataset"], "val", task)[:3]:
            pixels = image(row["path"])
            inputs = pre.det_input(pixels)[0] if task == "det" else pre.rec_input(pixels)
            actual = engine.run(None, {engine.get_inputs()[0].name: inputs})[0]
            if not np.isfinite(actual).all():
                raise ValueError("Output ONNX chứa NaN/Inf")
            if task == "rec" and (actual.ndim != 3 or actual.shape[-1] != len(pre.decoder.character)):
                raise ValueError("Dictionary không khớp trọng số REC")
            if task == "det" and (actual.ndim != 4 or actual.shape[:2] != (1, 1)):
                raise ValueError("Output DET không hợp lệ")
            shapes.append({"input": list(inputs.shape), "output": list(actual.shape)})
        checks[task] = {
            "verification": "official_onnx_sha256_and_val_smoke" if model["format"] == "onnx" else "local_paddle_vs_onnx",
            "shapes": shapes,
        }
        del engine
    pipe = PaddlePipeline(directory, catalog["det_resize"], catalog["benchmark"]["threads"])
    sample = read_split(catalog["dataset"], "val", "det")[0]
    smoke = pipe(image(sample["path"]))
    save_json(directory / "prepared.json", {
        "ready": True, "profile": profile(catalog, name), "artifacts": artifact_hashes(directory),
        "checks": checks, "val_smoke": {"image": sample["relative"], "prediction": smoke["text"]},
        "weights_source": "official_pretrained", "trained_on_dataset": False,
    })
    print(name, "ONNX FP32 sẵn sàng.", flush=True)


def candidates(default):
    result = [copy.deepcopy(default)]
    for threshold in (0.25, 0.35, 0.45):
        for unclip in (1.2, 1.5, 1.8):
            row = {**default, "thresh": threshold, "unclip_ratio": unclip}
            if row not in result:
                result.append(row)
    return result


def progress(name, phase, index, total, last):
    now = time.perf_counter()
    if now - last >= 15 or index == total:
        print(f"{name} {phase}: {index}/{total} ảnh", flush=True)
        return now
    return last


def tune(catalog, name, search=False):
    prepared_state(catalog, name)
    directory = run_dir(catalog, name)
    rows = read_split(catalog["dataset"], "val", "det")
    signature = split_signature(rows)
    pixels = [image(row["path"]) for row in rows]
    cv2.setNumThreads(1)
    pipe = PaddlePipeline(directory, catalog["det_resize"], catalog["benchmark"]["threads"])
    # Chỉ cache DET khi chọn ngưỡng; lúc benchmark vẫn chạy đủ luồng.
    raw, last = [], time.perf_counter()
    for index, pixels_one in enumerate(pixels, 1):
        raw.append(pipe.detect_raw(pixels_one))
        last = progress(name, "val DET", index, len(pixels), last)
    trials, selected, best = [], None, (-1, -float("inf"))
    grid = candidates(pipe.postprocess) if search else [pipe.postprocess]
    for index, params in enumerate(grid):
        pipe.postprocess = params
        predictions, last = [], time.perf_counter()
        for count, (p, det) in enumerate(zip(pixels, raw), 1):
            predictions.append(pipe.recognize(p, det)["text"])
            last = progress(name, f"val REC {index + 1}/{len(grid)}", count, len(pixels), last)
        metrics = text_scores(predictions, [r["text"] for r in rows])
        trials.append({"postprocess": params, **metrics})
        score = metrics["ACC"], -metrics["CER"]
        if score > best:
            best, selected = score, copy.deepcopy(params)
        print(f"{name} val {index + 1}/{len(grid)}: ACC={metrics['ACC']:.2f}, CER={metrics['CER']:.2f}", flush=True)
    save_json(directory / "val_selection.json", {"split": "val", "signature": signature, "trials": trials, "selected": selected, "search": search})
    frozen = {"profile": profile(catalog, name), "artifacts": artifact_hashes(directory),
              "val_signature": signature, "code_signature": code_signature(), "postprocess": selected,
              "selection": {"ACC": best[0], "CER": -best[1]}, "search": search}
    save_json(directory / "onnx/frozen.json", frozen)
    return frozen["selection"]


def validate_frozen(catalog, name, val):
    prepared_state(catalog, name)
    directory = run_dir(catalog, name)
    frozen = read_json(directory / "onnx/frozen.json")
    if frozen["profile"] != profile(catalog, name) or frozen["code_signature"] != code_signature():
        raise ValueError("Cấu hình hoặc code đã đổi; chọn lại trên val trước khi test")
    check_artifacts(directory, frozen["artifacts"])
    if frozen["artifacts"] != read_json(directory / "prepared.json")["artifacts"]:
        raise ValueError("Trọng số đã chuẩn bị khác bản chọn trên val")
    if frozen["val_signature"] != split_signature(val):
        raise ValueError("Val đã đổi sau khi khoá model")
    return frozen


def benchmark(catalog, name):
    with model_lock(run_dir(catalog, name)):
        return benchmark_locked(catalog, name)


def benchmark_locked(catalog, name):
    import onnxruntime as ort

    val = read_split(catalog["dataset"], "val", "det")
    frozen = validate_frozen(catalog, name, val)
    rows = read_split(catalog["dataset"], "test", "det")
    pixels = [image(row["path"]) for row in rows]
    params = catalog["benchmark"]
    if min(params["warmup"], params["repeats"], params["threads"]) < 1:
        raise ValueError("Số luồng, warm-up và số lượt đo cần dương")
    test_signature = split_signature(rows)
    protocol = {
        "name": name, "profile": profile(catalog, name), "test_signature": test_signature,
        "val_signature": frozen["val_signature"], "artifacts": frozen["artifacts"],
        "postprocess": frozen["postprocess"], "code_signature": code_signature(),
        "runtime": ort.__version__, "platform": platform.platform(), "processor": platform.processor(),
    }
    directory = run_dir(catalog, name)
    state = BenchmarkState(directory / "benchmark_progress.json", protocol, len(rows), params["repeats"])
    state.save()
    print(f"{name}: đã lưu {state.completed}/{len(rows) * params['repeats']} số đo; tiếp tục phần còn thiếu.", flush=True)
    cv2.setNumThreads(1)
    pipe = PaddlePipeline(run_dir(catalog, name), catalog["det_resize"], params["threads"], frozen["postprocess"])
    warm = image(val[0]["path"])
    last = time.perf_counter()
    for index in range(params["warmup"] if state.completed < len(rows) * params["repeats"] else 0):
        pipe(warm)
        last = progress(name, "warm-up", index + 1, params["warmup"], last)
    times, predictions = state.times, state.predictions
    for repeat in range(params["repeats"]):
        order = list(range(len(rows)))
        random.Random(params["seed"] + repeat).shuffle(order)
        last = time.perf_counter()
        for count, i in enumerate(order, 1):
            if len(times[i]) > repeat:
                continue
            start = time.perf_counter_ns()
            prediction = pipe(pixels[i])["text"]
            duration = (time.perf_counter_ns() - start) / 1_000_000
            # Ghi file sau khi dừng đồng hồ; không cộng I/O vào Speed.
            state.record(i, repeat, prediction, duration)
            last = progress(name, f"test lượt {repeat + 1}/{params['repeats']}", count, len(rows), last)
        print(f"{name} test: lượt {repeat + 1}/{params['repeats']}", flush=True)
    result = {
        "Model": name, **text_scores(predictions, [r["text"] for r in rows]),
        **speed_scores([t for item in times for t in item]),
        "test_signature": test_signature, "val_signature": frozen["val_signature"],
        "threads": params["threads"], "opencv_threads": 1, "batch_size": 1, "batch_unit": "plate",
        "rec_batching": "sequential", "precision": "FP32", "runtime": ort.__version__,
        "provider": "CPUExecutionProvider", "platform": platform.platform(), "processor": platform.processor(),
        "warmup": params["warmup"], "repeats": params["repeats"], "timing_scope": TIMING_SCOPE,
        "det_resize": catalog["det_resize"], "code_signature": code_signature(),
        "artifact_sha256": frozen["artifacts"], "postprocess": frozen["postprocess"],
        "weights_source": "official_pretrained", "trained_on_dataset": False,
    }
    directory = run_dir(catalog, name)
    save_json(directory / "test_predictions.json", [
        {"image": r["relative"], "GT": r["text"], "prediction": p, "correct": r["text"] == p, "times_ms": t}
        for r, p, t in zip(rows, predictions, times)
    ])
    save_json(directory / "test_result.json", result)
    return result


def compare(catalog, results):
    if len(results) != len(MODEL_NAMES) or {r["Model"] for r in results} != set(MODEL_NAMES):
        raise ValueError("Cần kết quả thật của đủ bảy bản")
    fields = ("test_signature", "val_signature", "threads", "opencv_threads", "batch_size", "batch_unit",
              "rec_batching", "precision", "runtime", "provider", "platform", "processor", "warmup",
              "repeats", "timing_scope", "det_resize", "code_signature")
    for key in fields:
        if len({json.dumps(r[key], sort_keys=True) for r in results}) != 1:
            raise ValueError(f"Điều kiện đo khác nhau: {key}")
    if results[0]["code_signature"] != code_signature():
        raise ValueError("Bảng chứa kết quả từ code cũ; chọn val/test lại bộ Paddle")
    baseline = next(r["Speed"] for r in results if r["Model"] == catalog["baseline"])
    if any(r["Speed"] <= 0 for r in results):
        raise ValueError("Speed cần lớn hơn 0")
    indexed = {r["Model"]: r for r in results}
    table = [{key: indexed[name][key] for key in ("Model", "ACC", "CER", "Speed", "FPS")}
             | {"Speedup": baseline / indexed[name]["Speed"]} for name in MODEL_NAMES]
    target = Path(catalog["run_dir"]) / "comparison.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("Model", "ACC", "CER", "Speed", "FPS", "Speedup"))
        writer.writeheader()
        writer.writerows(table)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "tune", "test", "compare"))
    parser.add_argument("--model", choices=MODEL_NAMES, nargs="+")
    parser.add_argument("--paddle-python")
    parser.add_argument("--search", action="store_true", help="Dò lưới ngưỡng trên val; mặc định giữ ngưỡng nguồn")
    args = parser.parse_args()
    catalog = load_catalog()
    if args.action == "compare":
        val = read_split(catalog["dataset"], "val", "det")
        test_sig = split_signature(read_split(catalog["dataset"], "test", "det"))
        results = []
        for name in MODEL_NAMES:
            frozen = validate_frozen(catalog, name, val)
            result = read_json(run_dir(catalog, name) / "test_result.json")
            if result["artifact_sha256"] != frozen["artifacts"] or result["postprocess"] != frozen["postprocess"] or result["test_signature"] != test_sig:
                raise ValueError("Kết quả không khớp model/ngưỡng/dữ liệu hiện tại")
            results.append(result)
        print(json.dumps(compare(catalog, results), ensure_ascii=False, indent=2))
        return
    for name in args.model or MODEL_NAMES:
        print("\n===", name, args.action, "===", flush=True)
        if args.action == "prepare":
            prepare(catalog, name, args.paddle_python)
        elif args.action == "tune":
            print(tune(catalog, name, args.search), flush=True)
        else:
            result = benchmark(catalog, name)
            print({k: result[k] for k in ("Model", "ACC", "CER", "Speed", "FPS")}, flush=True)


if __name__ == "__main__":
    main()

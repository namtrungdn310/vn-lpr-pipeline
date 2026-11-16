"""Benchmark FP32 chung: năm vòng, bốn lõi P, không I/O trong vùng đo."""

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

import cv2
import onnxruntime as ort
import psutil

from ocr.config import ROOT, save_json
from ocr.data import image, split_signature
from ocr.metrics import text_scores
from ocr.paddle_suite_state import model_lock
from ocr import cpu_protocol as cpu
from .common import settings, directory, read_json, code_signature, validate_frozen
from .data import rows, verify_dataset
from .runtime import pipeline

PROTOCOL = "adapted_seven_ocr_fp32_fixed_cpu_v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def power():
    battery = psutil.sensors_battery()
    scheme = (
        subprocess.check_output(["powercfg", "/getactivescheme"], text=True).strip()
        if os.name == "nt"
        else None
    )
    return {"ac": None if battery is None else bool(battery.power_plugged), "power_scheme": scheme}


def check_power(expected):
    actual = power()
    if actual["ac"] is False or actual != expected:
        raise RuntimeError("Nguồn/chế độ nguồn khác manifest")
    return actual


def environment():
    import winreg

    with winreg.OpenKey(
        winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
    ) as key:
        name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return {
        "cpu": name,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "opencv": cv2.__version__,
        "memory_bytes": psutil.virtual_memory().total,
        "provider": "CPUExecutionProvider",
    }


def safe_run(path=None):
    pointer = directory() / "latest_benchmark.json"
    run = Path(path or read_json(pointer)["run_dir"]).resolve()
    if not run.is_relative_to((directory() / "benchmark").resolve()):
        raise ValueError("Sai thư mục benchmark")
    return run


def create():
    verify_dataset()
    cfg = settings()
    context = cpu.apply()
    env = environment()
    if "14600KF" not in env["cpu"]:
        raise RuntimeError("Lượt này chốt CPU i5-14600KF; không đo thay trên ThinkPad")
    actual_power = power()
    check_power(actual_power)
    names = [m["name"] for m in cfg["models"]]
    models = {name: validate_frozen(name) for name in names}
    run = directory() / "benchmark" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run.mkdir(parents=True, exist_ok=False)
    orders = []
    for i in range(cfg["benchmark"]["rounds"]):
        order = names.copy()
        random.Random(cfg["seed"] + i).shuffle(order)
        orders.append(order)
    manifest = {
        "protocol": PROTOCOL,
        "code_signature": code_signature(),
        "settings": cfg,
        "environment": env,
        "power": actual_power,
        "cpu_protocol": context,
        "models": models,
        "orders": orders,
        "test_signature": split_signature(rows("test")),
        "val_signature": split_signature(rows("val")),
        "created_utc": now(),
        "disk_io_in_timed_loop": False,
    }
    if any(s["val_signature"] != manifest["val_signature"] for s in models.values()):
        raise ValueError("Val model khác val benchmark")
    save_json(run / "manifest.json", manifest)
    save_json(directory() / "active_benchmark.json", {"run_dir": str(run)})
    return run


def manifest(run):
    value = read_json(run / "manifest.json")
    if (
        value["protocol"] != PROTOCOL
        or value["code_signature"] != code_signature()
        or value["settings"] != settings()
        or value["environment"] != environment()
    ):
        raise ValueError("Code/config/máy đo khác lượt đã chốt")
    return value


def worker(run, name, index):
    m = manifest(run)
    cpu.apply(m["cpu_protocol"])
    check_power(m["power"])
    path = run / "passes" / name / f"round_{index+1:02d}.json"
    if path.exists():
        validate_record(read_json(path), m, name, index)
        return
    if validate_frozen(name) != m["models"][name]:
        raise ValueError("Model đã đổi")
    cfg = m["settings"]["benchmark"]
    cv2.setNumThreads(cfg["opencv_threads"])
    test = rows("test")
    val = rows("val")
    if split_signature(test) != m["test_signature"] or split_signature(val) != m["val_signature"]:
        raise ValueError("Split đã đổi")
    pixels = [image(row["path"]) for row in test]
    warm = image(val[0]["path"])
    infer = pipeline(name, cfg["threads"])
    for _ in range(cfg["warmup"]):
        infer(warm)
    predictions = [None] * len(test)
    times = [None] * len(test)
    order = list(range(len(test)))
    random.Random(settings()["seed"] + index).shuffle(order)
    before = {
        "utc": now(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "power": check_power(m["power"]),
    }
    for i in order:
        start = time.perf_counter_ns()
        out = infer(pixels[i])
        elapsed = (time.perf_counter_ns() - start) / 1e6
        predictions[i] = out["text"]
        times[i] = elapsed
    after = {"utc": now(), "power": check_power(m["power"])}
    cpu.check(m["cpu_protocol"])
    record = {
        "name": name,
        "round": index,
        "manifest_signature": m["code_signature"],
        "model": m["models"][name],
        "cpu_protocol": m["cpu_protocol"],
        "before": before,
        "after": after,
        "details": [
            {"image": r["relative"], "GT": r["text"], "prediction": p, "time_ms": t}
            for r, p, t in zip(test, predictions, times)
        ],
    }
    validate_record(record, m, name, index)
    save_json(path, record)
    print(name, f"{index+1}/{cfg['rounds']}: {statistics.mean(times):.4f} ms/image", flush=True)


def validate_record(record, m, name, index):
    if (
        record["name"] != name
        or record["round"] != index
        or record["manifest_signature"] != m["code_signature"]
        or record["model"] != m["models"][name]
    ):
        raise ValueError("Pass khác model/lượt/code")
    if (
        record["cpu_protocol"] != m["cpu_protocol"]
        or record["before"]["power"] != m["power"]
        or record["after"]["power"] != m["power"]
    ):
        raise ValueError("Pass khác CPU/nguồn")
    test = rows("test")
    details = record["details"]
    if [(d["image"], d["GT"]) for d in details] != [(r["relative"], r["text"]) for r in test]:
        raise ValueError("Pass khác test")
    if any(not math.isfinite(d["time_ms"]) or d["time_ms"] <= 0 for d in details):
        raise ValueError("Thời gian không hợp lệ")


def collect(run):
    m = manifest(run)
    if split_signature(rows("test")) != m["test_signature"]:
        raise ValueError("Test đã đổi")
    result = []
    for name in m["models"]:
        if validate_frozen(name) != m["models"][name]:
            raise ValueError("Model đã đổi")
        records = [
            read_json(run / "passes" / name / f"round_{i+1:02d}.json")
            for i in range(m["settings"]["benchmark"]["rounds"])
        ]
        for i, r in enumerate(records):
            validate_record(r, m, name, i)
        predicted = [d["prediction"] for d in records[0]["details"]]
        if any([d["prediction"] for d in r["details"]] != predicted for r in records[1:]):
            raise ValueError("Dự đoán khác giữa vòng")
        means = [statistics.mean(d["time_ms"] for d in r["details"]) for r in records]
        result.append(
            {
                "Model": name,
                **text_scores(predicted, [d["GT"] for d in records[0]["details"]]),
                "Speed": statistics.mean(means),
                "Speed_std": statistics.stdev(means),
                "FPS": 1000 / statistics.mean(means),
                "round_mean_ms": means,
            }
        )
    return result


def render(results):
    slowest = max(r["Speed"] for r in results)
    ordered = sorted(results, key=lambda r: (-r["ACC"], r["CER"], r["Speed"]))
    table = "| Rank | Model | ACC (%) ↑ | CER (%) ↓ | Speed ± SD (ms/image) ↓ | FPS ↑ | Speedup (×) ↑ |\n|---:|---|---:|---:|---:|---:|---:|"
    for rank, r in enumerate(ordered, 1):
        r = {**r, "Rank": rank, "Speedup": slowest / r["Speed"]}
        table += f"\n| {rank} | {r['Model']} | {r['ACC']:.2f} | {r['CER']:.2f} | {r['Speed']:.2f} ± {r['Speed_std']:.2f} | {r['FPS']:.2f} | {r['Speedup']:.2f} |"
    return table


def table():
    pointer = directory() / "latest_benchmark.json"
    if not pointer.exists():
        return (
            "Chưa có benchmark FP32 của bảy OCR sau train. Không dùng bảng pretrained cũ thay thế."
        )
    return render(collect(safe_run()))


def run_suite(run=None):
    import csv

    with model_lock(directory() / "benchmark"):
        run = safe_run(run) if run else create()
        m = manifest(run)
        total = sum(map(len, m["orders"]))
        completed = 0
        try:
            for i, order in enumerate(m["orders"]):
                for name in order:
                    save_json(
                        run / "state.json",
                        {
                            "status": "running",
                            "model": name,
                            "round": i + 1,
                            "completed_passes": completed,
                            "total_passes": total,
                        },
                    )
                    subprocess.run(
                        [
                            sys.executable,
                            "-u",
                            "-B",
                            "-m",
                            "ocr.adaptation.benchmark",
                            "worker",
                            "--run-dir",
                            str(run),
                            "--model",
                            name,
                            "--round",
                            str(i),
                        ],
                        cwd=ROOT,
                        check=True,
                    )
                    completed += 1
            results = collect(run)
            save_json(run / "results.json", results)
            text = render(results)
            (run / "comparison.md").write_text(text + "\n", encoding="utf-8")
            with (run / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                fields = ["Rank", "Model", "ACC", "CER", "Speed", "Speed_std", "FPS", "Speedup"]
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                slowest = max(r["Speed"] for r in results)
                for rank, r in enumerate(
                    sorted(results, key=lambda r: (-r["ACC"], r["CER"], r["Speed"])), 1
                ):
                    writer.writerow(
                        {
                            **{k: r[k] for k in fields if k not in ("Rank", "Speedup")},
                            "Rank": rank,
                            "Speedup": slowest / r["Speed"],
                        }
                    )
            save_json(
                run / "state.json",
                {
                    "status": "complete",
                    "completed_passes": completed,
                    "total_passes": total,
                    "finished_utc": now(),
                },
            )
            save_json(directory() / "latest_benchmark.json", {"run_dir": str(run)})
            print(text)
        except BaseException as error:
            save_json(
                run / "state.json",
                {
                    "status": "interrupted",
                    "completed_passes": completed,
                    "total_passes": total,
                    "error": str(error),
                },
            )
            raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("run", "worker", "table"))
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--model")
    p.add_argument("--round", type=int)
    a = p.parse_args()
    if a.action == "table":
        print(table())
    elif a.action == "worker":
        worker(safe_run(a.run_dir), a.model, a.round)
    else:
        run_suite(a.run_dir)

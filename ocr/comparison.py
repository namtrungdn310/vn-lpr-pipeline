"""Ghép kết quả đã đo, kiểm điều kiện và xếp hạng chín hệ thống OCR."""

import csv
import math
from pathlib import Path

from .config import ROOT
from .data import read_split, split_signature
from .metrics import speed_scores, text_scores
from . import paddle_suite
from .evaluate import code_signature, deployed_config, validate_export
from .runtime import inference_signature

CONDITIONS = (
    "test_signature", "samples", "reference_characters", "threads", "opencv_threads",
    "batch_size", "batch_unit", "precision", "runtime", "provider", "platform",
    "processor", "warmup", "repeats", "timing_scope",
)


def check_conditions(results):
    if not results:
        raise ValueError("Chưa có kết quả để so sánh")
    for key in CONDITIONS:
        if len({str(result[key]) for result in results}) != 1:
            raise ValueError(f"Điều kiện đo khác nhau: {key}")


def collect_results():
    catalog = paddle_suite.load_catalog()
    val = read_split(catalog["dataset"], "val", "det")
    test = read_split(catalog["dataset"], "test", "det")
    test_signature = split_signature(test)
    results = []
    sources = [(name, paddle_suite.run_dir(catalog, name), True) for name in paddle_suite.MODEL_NAMES]
    sources += [(name, ROOT / "runs" / name.lower(), False) for name in ("VietLPR_v1", "VietLPR_v2")]
    for name, directory, is_paddle in sources:
        result = paddle_suite.read_json(directory / "test_result.json")
        if result["Model"] != name or result["test_signature"] != test_signature:
            raise ValueError(f"Tên model hoặc tập test không khớp: {name}")
        if is_paddle:
            frozen = paddle_suite.validate_frozen(catalog, name, val)
            expected = frozen["artifacts"]
            if result["code_signature"] != paddle_suite.code_signature() or result["postprocess"] != frozen["postprocess"]:
                raise ValueError(f"Kết quả khác code hoặc ngưỡng đã chốt: {name}")
        else:
            cfg = deployed_config(name.lower())
            validate_export(cfg)
            frozen = paddle_suite.read_json(directory / "onnx/frozen.json")
            expected = frozen["files"]
            if frozen["inference_signature"] != inference_signature() or frozen["val_signature"] != split_signature(val):
                raise ValueError(f"Code hoặc val của {name} đã đổi")
            if result["code_signature"] != code_signature():
                raise ValueError(f"Code đo VietLPR đã đổi: {name}")
            if any(paddle_suite.digest(directory / "onnx" / relative) != digest for relative, digest in expected.items()):
                raise ValueError(f"Model đã đổi sau evaluation: {name}")
        if result["artifact_sha256"] != expected:
            raise ValueError(f"Kết quả không thuộc model đã chốt: {name}")
        details = paddle_suite.read_json(directory / "test_predictions.json")
        if [(row["image"], row["GT"]) for row in details] != [(row["relative"], row["text"]) for row in test]:
            raise ValueError(f"Danh sách ảnh/nhãn test không khớp: {name}")
        if any(len(row["times_ms"]) != result["repeats"] for row in details):
            raise ValueError(f"Chưa đủ số lượt đo: {name}")
        scores = text_scores([row["prediction"] for row in details], [row["GT"] for row in details])
        scores.update(speed_scores([time for row in details for time in row["times_ms"]]))
        if any(not math.isclose(value, result[key], rel_tol=1e-10, abs_tol=1e-10) for key, value in scores.items()):
            raise ValueError(f"Chỉ số khác dự đoán hoặc số đo gốc: {name}")
        results.append(result)
    check_conditions(results)
    return results


def rank_results(results):
    check_conditions(results)
    baseline = max(result["Speed"] for result in results)
    ordered = sorted(results, key=lambda result: (-result["ACC"], result["CER"], result["Speed"], result["Model"]))
    return [
        {"Rank": rank, **{key: result[key] for key in ("Model", "ACC", "CER", "Speed", "FPS")},
         "Speedup": baseline / result["Speed"]}
        for rank, result in enumerate(ordered, 1)
    ]


def save_table(rows, destination=None):
    path = Path(destination or ROOT / "runs/comparison_ranked.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("Rank", "Model", "ACC", "CER", "Speed", "FPS", "Speedup"))
        writer.writeheader()
        writer.writerows(rows)
    return path


def markdown_table(rows):
    lines = ["| Rank | Model | ACC (%) ↑ | CER (%) ↓ | Speed (ms/ảnh) ↓ | FPS ↑ | Speedup (×) ↑ |",
             "|---:|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        values = [str(row["Rank"]), row["Model"]] + [f"{row[key]:.2f}" for key in ("ACC", "CER", "Speed", "FPS", "Speedup")]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    rows = rank_results(collect_results())
    print(save_table(rows))
    print(markdown_table(rows))

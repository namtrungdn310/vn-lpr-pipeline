"""Tên hiển thị và bảng hiện tại; không thay mã định danh/số đo đã chốt."""

import argparse
from pathlib import Path

from . import optimized_benchmark as measured
from .config import ROOT

LABELS = {
    "VietLPR_v2": "VietLPR_v1_FP32",
    "VietLPR_v2_INT4": "VietLPR_v1_INT4_mixed",
    measured.NEW_OCR: "VietLPR_v1_INT8_mixed",
    "YOLO26n-OBB_INT8_optimized": "YOLO26n-OBB_INT8_mixed",
}
GROUPS = {
    "ocr": measured.groups()["ocr"],
    "quantization": ["VietLPR_v2", measured.NEW_OCR, "VietLPR_v2_INT4"],
    "detection": ["YOLO26n-OBB_FP32", "YOLO26n-OBB_INT8_optimized", "YOLO26n-OBB_INT4_mixed"],
}


def render(group, results):
    renamed = {
        key: {**value, "Model": LABELS.get(key, value["Model"])} for key, value in results.items()
    }
    selected = GROUPS[group]
    if group != "detection":
        rows, columns, text = measured.baseline.make_table(
            renamed, selected, with_size=group == "quantization"
        )
        if group == "quantization":
            rows.sort(key=lambda row: -row["FPS"])
            text = "| Rank | Model | ACC (%) ↑ | CER (%) ↓ | Speed ± SD (ms/ảnh) ↓ | FPS ↑ | Speedup (×) ↑ | Size (MB) ↓ |\n|---:|---|---:|---:|---:|---:|---:|---:|"
            for rank, row in enumerate(rows, 1):
                row["Rank"] = rank
                values = [
                    str(rank),
                    row["Model"],
                    f"{row['ACC']:.2f}",
                    f"{row['CER']:.2f}",
                    f"{row['Speed']:.2f} ± {row['Speed_std']:.2f}",
                    f"{row['FPS']:.2f}",
                    f"{row['Speedup']:.2f}",
                    f"{row['Size_MB']:.2f}",
                ]
                text += "\n| " + " | ".join(values) + " |"
            text += "\n\nRank theo FPS giảm dần; model chậm nhất trong bảng làm Speedup 1×."
            text += "\n\nINT8 dùng lượng tử hoá hỗn hợp; DET/Conv giữ FP32. INT4 cũng là bản mixed."
        return rows, columns, text
    chosen = sorted((renamed[key] for key in selected), key=lambda r: -r["FPS"])
    slowest = max(r["Speed"] for r in chosen)
    columns = [
        "Rank",
        "Model",
        "mAP50",
        "mAP50-95",
        "Precision",
        "Recall",
        "Speed",
        "Speed_std",
        "FPS",
        "Speedup",
        "Size_MB",
    ]
    rows = [
        {
            "Rank": i,
            **{k: r[k] for k in columns if k not in ("Rank", "Speedup")},
            "Speedup": slowest / r["Speed"],
        }
        for i, r in enumerate(chosen, 1)
    ]
    text = "| Rank | Model | mAP50 (%) ↑ | mAP50-95 (%) ↑ | Precision (%) ↑ | Recall (%) ↑ | Speed ± SD (ms/ảnh) ↓ | FPS ↑ | Speedup (×) ↑ | Size (MB) ↓ |\n|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    for row in rows:
        values = [
            str(row["Rank"]),
            row["Model"],
            *(f"{row[k]:.2f}" for k in ("mAP50", "mAP50-95", "Precision", "Recall")),
            f"{row['Speed']:.2f} ± {row['Speed_std']:.2f}",
            *(f"{row[k]:.2f}" for k in ("FPS", "Speedup", "Size_MB")),
        ]
        text += "\n| " + " | ".join(values) + " |"
    return (
        rows,
        columns,
        text
        + "\n\nRank theo FPS giảm dần; model chậm nhất trong bảng làm Speedup 1×.\n\nINT8 lượng tử hoá backbone, giữ neck/head FP32; INT4 là bản mixed.",
    )


def display_table(group="ocr", run=None):
    if group == "ocr":
        from .adaptation.benchmark import table

        return table()
    if run is None and not measured.POINTER.is_file():
        return measured.display_table(group)
    run = measured.safe_run(run)
    if measured.read_json(run / "state.json")["status"] != "complete":
        return measured.display_table(group, run)
    results = collect_archive(run, GROUPS[group])
    return render(group, results)[2]


def collect_archive(run, selected):
    """Chỉ kiểm các model lượng tử còn dùng, không yêu cầu 14 baseline đã bỏ."""
    m = measured.manifest_for(run)
    results = {}
    for name in selected:
        _, current = measured.assets(name)
        if current != m["models"][name]:
            raise ValueError("Model lượng tử khác manifest")
        records = [
            measured.read_json(measured.pass_path(run, name, i))
            for i in range(m["config"]["rounds"])
        ]
        result = measured.aggregate(records, m, name)
        if name in measured.YOLO:
            ap = measured.read_json(Path(run) / "ap" / (name + ".json"))
            if (
                ap["protocol"]["model_sha256"] != current["sha256"]
                or ap["protocol"]["split"] != "test"
                or ap["protocol"]["dataset_signature"] != m["detector_dataset_signature"]
                or ap["protocol"]["code_signature"] != measured.yq.code_signature()
            ):
                raise ValueError("mAP khác model/test đã chốt")
            result.update({key: ap[key] for key in ("mAP50", "mAP50-95")})
        results[name] = result
    return results


def refresh_tables(run=None):
    """Chỉ xuất lại bảng, không chạy inference hoặc sửa các pass gốc."""
    import nbformat

    run = measured.safe_run(run)
    if measured.read_json(run / "state.json")["status"] != "complete":
        raise ValueError("Chưa có lượt đo hoàn chỉnh")
    selected = list(dict.fromkeys([*GROUPS["quantization"], *GROUPS["detection"]]))
    results = collect_archive(run, selected)
    for group, number in (("quantization", "06"), ("detection", "07")):
        rows, columns, text = render(group, results)
        measured.baseline.save_csv(run / f"comparison_{group}.csv", rows, columns)
        (run / f"comparison_{group}.md").write_text(text + "\n", encoding="utf-8")
        path = next((ROOT / "notebooks").glob(number + "_*.ipynb"))
        book = nbformat.read(path, as_version=4)
        book.cells[-1].outputs = [
            nbformat.v4.new_output(
                "display_data", data={"text/markdown": text, "text/plain": text}
            ),
            nbformat.v4.new_output("stream", name="stdout", text=f"Run: {run}\n"),
        ]
        book.cells[-1].execution_count = None
        nbformat.validate(book)
        nbformat.write(book, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("table", "refresh", "run", "all"))
    parser.add_argument("--group", choices=tuple(GROUPS), default="ocr")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.action in ("run", "all"):
        raise ValueError(
            "Lượt cũ chỉ đọc. Train/benchmark FP32 mới dùng ocr.adaptation.suite và ocr.adaptation.benchmark"
        )
    elif args.action == "refresh":
        refresh_tables(args.run_dir)
    else:
        print(display_table(args.group, args.run_dir))


if __name__ == "__main__":
    main()

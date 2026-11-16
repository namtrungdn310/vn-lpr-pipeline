"""Xuất bảng toàn cảnh; lệnh trực tiếp đọc benchmark OCR hiện tại."""

import csv
from pathlib import Path


def save_table(rows, destination):
    path = Path(destination)
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
    from .adaptation.benchmark import table

    print(table())

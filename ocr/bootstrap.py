"""Chuẩn bị mã nguồn đúng phiên bản; không gọi train."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {}  # Không tải lại kiến trúc VietLPR cũ đã bỏ.
NOTEBOOK_SHA256 = "aafe7b2f63c0c6346ac00f807c96ee249de186e37021a0d0c81daf915f2adbed"


def extract_v2():
    required = ("models.py", "source.py", "inference.py", "source_manifest.json")
    target = ROOT / "third_party/VietLPR_v2"
    if not all((target / name).is_file() for name in required):
        raise FileNotFoundError(
            "Thiếu nguồn V2 đã tách; lấy đủ third_party/VietLPR_v2 từ repository"
        )
    manifest = json.loads((target / "source_manifest.json").read_text(encoding="utf-8"))
    if manifest["sha256"] != NOTEBOOK_SHA256:
        raise ValueError("Nguồn notebook V2 khác bản đã chốt")
    return manifest["sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("all", *SOURCES, "VietLPR_v2", "v2"), default="all")
    args = parser.parse_args()
    if args.source in ("all", "VietLPR_v2", "v2"):
        print("Notebook SHA256:", extract_v2())


if __name__ == "__main__":
    main()

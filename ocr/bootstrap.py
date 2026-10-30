"""Chuẩn bị mã nguồn đúng phiên bản; không gọi train."""

import argparse
import ast
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "VietLPR_v1": {
        "url": "https://github.com/vietdai-bk/VietLPR_v2.git",
        "ref": "main",
        "sha": "ff3a6b3d65b64463ae61b251abb9e117c3e0ea56",
    },
}
NOTEBOOK_SHA256 = "aafe7b2f63c0c6346ac00f807c96ee249de186e37021a0d0c81daf915f2adbed"


def checkout(name):
    source = SOURCES[name]
    target = ROOT / "third_party" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", source["ref"], source["url"], str(target)],
            check=True,
        )
    actual = subprocess.check_output(
        ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != source["sha"]:
        subprocess.run(
            ["git", "-C", str(target), "fetch", "--depth", "1", "origin", source["sha"]], check=True
        )
        subprocess.run(
            ["git", "-C", str(target), "checkout", "--detach", source["sha"]], check=True
        )
    return target


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


def extract_ops():
    """Chỉ lấy hàm hình học/hậu xử lý; không import framework khi chạy ONNX."""
    specs = {
        "ops_v1.py": [
            (
                ROOT / "third_party/VietLPR_v1/vietlpr/utils.py",
                {"order_points", "warp_crop", "letterbox_resize"},
            ),
            (
                ROOT / "third_party/VietLPR_v1/vietlpr/infer.py",
                {
                    "get_box_height",
                    "build_box_info_from_mask",
                    "split_tall_component_by_projection",
                    "get_boxes_from_overlay_mask",
                    "pad_rotated_box",
                },
            ),
        ],
    }
    for name, inputs in specs.items():
        functions = []
        for path, selected in inputs:
            source = path.read_text(encoding="utf-8")
            functions.extend(
                node
                for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name in selected
            )
        header = '"""Hình học/hậu xử lý v1 tách từ upstream, không cần torch khi inference."""\nfrom typing import Tuple\nimport cv2\nimport numpy as np\n\n'
        (ROOT / "ocr" / name).write_text(
            header + "\n\n".join(ast.unparse(node) for node in functions) + "\n",
            encoding="utf-8", newline="\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("all", *SOURCES, "VietLPR_v2", "v2"), default="all")
    args = parser.parse_args()
    for name in SOURCES:
        if args.source in ("all", name):
            print(checkout(name))
    if args.source in ("all", "VietLPR_v2", "v2"):
        print("Notebook SHA256:", extract_v2())
        if (ROOT / "third_party/VietLPR_v1/vietlpr/infer.py").is_file() and not (
            ROOT / "ocr/ops_v1.py"
        ).is_file():
            extract_ops()


if __name__ == "__main__":
    main()

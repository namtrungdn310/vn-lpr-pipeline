"""Tải đúng phiên bản mã nguồn và tách kiến trúc v2; không chạy train."""

import argparse
import ast
import hashlib
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
    "PaddleOCR": {
        "url": "https://github.com/PaddlePaddle/PaddleOCR.git",
        "ref": "v2.9.1",
        "sha": "07603421c20a96bb94bb87d0c4211032527ae706",
    },
}
NOTEBOOK_SHA256 = "aafe7b2f63c0c6346ac00f807c96ee249de186e37021a0d0c81daf915f2adbed"


def notebook_content():
    path = ROOT / "VietLPR_v2.ipynb"
    if hashlib.sha256(path.read_bytes()).hexdigest() != NOTEBOOK_SHA256:
        raise ValueError("Notebook nguồn đã đổi; kiểm kiến trúc trước khi cập nhật SHA đã chốt")
    return json.loads(path.read_text(encoding="utf-8"))


def render_definitions(nodes):
    """Giữ thuật toán gốc, rút gọn chú thích bằng tiếng Việt."""
    descriptions = {
        "RepConv": "Nhiều nhánh khi train; gộp thành một convolution tương đương khi xuất model.",
        "MOBlock": "Khối MobileOne gồm convolution theo kênh rồi convolution 1×1.",
        "FastDBNet": "Nhận ảnh BGR (B,3,S,S); trả logits vùng lõi chữ (B,1,S/2,S/2).",
        "FastCRNN": "Nhận ảnh BGR (B,3,H,W); trả logits CTC (B,W/4,số lớp).",
        "MHSA": "Lớp attention tự viết, hỗ trợ xuất ONNX.",
        "ConvSeqBlock": "Khối convolution cho chuỗi, có nhánh cộng tắt.",
        "fit_quad": "Fit cạnh trên/dưới riêng; nới bốn cạnh và hai đầu để bao vùng chữ.",
        "boxes_from_bitmap": "Lấy các tứ giác từ bản đồ xác suất và mặt nạ nhị phân.",
        "box_score_fast": "Tính điểm trung bình trong tứ giác.",
        "box_score_slow": "Tính điểm trung bình trong đường bao.",
    }
    for node in nodes:
        for child in ast.walk(node):
            if (
                isinstance(child, (ast.FunctionDef, ast.ClassDef))
                and child.name in descriptions
                and ast.get_docstring(child)
            ):
                child.body[0].value = ast.Constant(descriptions[child.name])
    return "\n\n".join(ast.unparse(node) for node in nodes)


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
    notebook = ROOT / "VietLPR_v2.ipynb"
    target = ROOT / "third_party/VietLPR_v2/models.py"
    if not notebook.is_file():
        if target.is_file():
            return "Dùng kiến trúc v2 đã tách và lưu trong repository"
        raise FileNotFoundError("Thiếu notebook gốc và kiến trúc v2 đã tách")
    content = notebook_content()
    definitions = []
    for cell in content["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        if not any(
            isinstance(node, ast.ClassDef) and node.name == "FastDBNet" for node in tree.body
        ):
            continue
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name not in (
                "count_macs",
                "count_params",
            ):
                definitions.append(node)
    if not definitions:
        raise ValueError("Không tìm thấy FastDBNet/FastCRNN trong notebook")
    target.parent.mkdir(parents=True, exist_ok=True)
    header = '"""Kiến trúc tách từ VietLPR_v2.ipynb; không chạy cell train hoặc video."""\n\nimport copy\nimport math\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n'
    target.write_text(
        header + render_definitions(definitions) + "\n", encoding="utf-8", newline="\n"
    )
    return hashlib.sha256(notebook.read_bytes()).hexdigest()


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
        (ROOT / "experiments" / name).write_text(
            header + render_definitions(functions) + "\n", encoding="utf-8", newline="\n"
        )
    notebook = notebook_content()
    functions = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        functions.extend(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in ("_fit_line", "fit_quad")
        )
    (ROOT / "experiments/ops_v2.py").write_text(
        '"""Fit bốn cạnh v2 tách từ notebook, không chạy cell train."""\nimport math\nimport cv2\nimport numpy as np\n\n'
        + render_definitions(functions)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def extract_paddle_postprocess():
    source_path = ROOT / "third_party/PaddleOCR/ppocr/postprocess/db_postprocess.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DBPostProcess"
    )
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in (
            "__init__",
            "boxes_from_bitmap",
            "unclip",
            "get_mini_boxes",
            "box_score_fast",
            "box_score_slow",
        )
    ]
    header_lines = []
    for line in source.splitlines():
        if not line.startswith("#"):
            break
        header_lines.append(line)
    header = "\n".join(header_lines)
    text = (
        header
        + '\n\n"""Hậu xử lý DB chính thức, chỉ giữ nhánh quad dùng NumPy để chạy không cần Paddle."""\nimport cv2\nimport numpy as np\nimport pyclipper\nfrom shapely.geometry import Polygon\n\n'
    )
    (ROOT / "experiments/paddle_postprocess.py").write_text(
        text + render_definitions([cls]) + "\n", encoding="utf-8", newline="\n"
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
        if (ROOT / "third_party/VietLPR_v1/vietlpr/infer.py").is_file() and (
            ROOT / "VietLPR_v2.ipynb"
        ).is_file():
            extract_ops()
    if args.source in ("all", "PaddleOCR"):
        extract_paddle_postprocess()


if __name__ == "__main__":
    main()

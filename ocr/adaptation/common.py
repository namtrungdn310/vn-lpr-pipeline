"""Cấu hình portable và checksum cho lượt so sánh mới."""

import hashlib
import json
from pathlib import Path

from ocr.config import ROOT, save_json

CONFIG = ROOT / "configs/ocr/adaptation.json"


def settings():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if cfg["precision"] != "FP32":
        raise ValueError("Lượt này chỉ dùng FP32")
    return cfg


def entry(name):
    return next(m for m in settings()["models"] if m["name"] == name)


def directory(name=None):
    root = ROOT / settings()["run_dir"]
    return root / name if name else root


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def code_signature():
    paths = sorted(Path(__file__).parent.glob("*.py"))
    paths += [
        ROOT / "ocr" / name
        for name in (
            "data.py",
            "metrics.py",
            "cpu_protocol.py",
            "paddle_suite_runtime.py",
            "public_vn_runtime.py",
        )
    ]
    return hashlib.sha256(CONFIG.read_bytes() + b"".join(p.read_bytes() for p in paths)).hexdigest()


def check_fp32(path):
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer):
        raise ValueError("ONNX phải gói trọng số trong một file")
    if any(
        t.data_type in (onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16)
        for t in model.graph.initializer
    ):
        raise ValueError("Model còn trọng số FP16/BF16")
    forbidden = {
        "QuantizeLinear",
        "DequantizeLinear",
        "MatMulNBits",
        "DynamicQuantizeLinear",
        "QLinearConv",
        "QLinearMatMul",
        "ConvInteger",
        "MatMulInteger",
    }
    if any(n.op_type in forbidden for n in model.graph.node):
        raise ValueError("Không nhận graph lượng tử hoá")
    onnx.checker.check_model(model)
    return digest(path)


def freeze(name, details):
    from .data import verify_dataset
    from ocr.data import read_split, split_signature

    cfg = settings()
    verify_dataset()
    base = directory(name)
    files = {
        p.relative_to(base).as_posix(): digest(p)
        for p in sorted((base / "onnx").glob("*"))
        if p.is_file()
    }
    if not files:
        raise ValueError("Chưa có model ONNX")
    for p in (base / "onnx").glob("*.onnx"):
        check_fp32(p)
    state = {
        "name": name,
        "precision": "FP32",
        "files": files,
        "dataset": cfg["dataset_fingerprint"],
        "val_signature": split_signature(read_split(ROOT / cfg["dataset"], "val", "det")),
        "training_config": cfg,
        "export_code_signature": code_signature(),
        "selection_split": "val",
        "details": details,
    }
    save_json(base / "frozen.json", state)
    return state


def validate_frozen(name):
    base = directory(name)
    state = read_json(base / "frozen.json")
    if (
        state["name"] != name
        or state["precision"] != "FP32"
        or state["dataset"] != settings()["dataset_fingerprint"]
        or state["training_config"] != settings()
    ):
        raise ValueError("Model khác tên, precision, dataset hoặc cấu hình train")
    if any(digest(base / p) != sha for p, sha in state["files"].items()):
        raise ValueError("Model hoặc cấu hình suy luận đã thay đổi")
    return state

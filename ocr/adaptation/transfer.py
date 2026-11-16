"""Đóng gói model đã chốt để chép PC riêng; không upload trọng số."""

import argparse
import json
from pathlib import Path
import shutil
import zipfile

from ocr.config import ROOT, save_json
from .common import directory, digest, freeze, read_json
from .data import verify_dataset


def create():
    verify_dataset()
    original = ROOT / "runs/vietlpr_v2/onnx"
    target = directory("VietLPR_v1") / "onnx"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("det.onnx", "rec.onnx", "lpr_config.json"):
        shutil.copy2(original / name, target / name)
        if digest(original / name) != digest(target / name):
            raise ValueError("Sao chép làm đổi model đã chốt")
    save_json(target / "runtime.json", {"family": "vietlpr"})
    original_freeze = read_json(original / "frozen.json")
    freeze(
        "VietLPR_v1",
        {
            "training_performed": False,
            "renamed_from": "VietLPR_v2",
            "original_frozen": original_freeze,
            "original_sha256": {n: digest(original / n) for n in ("det.onnx", "rec.onnx")},
        },
    )
    files = [*target.glob("*"), directory("VietLPR_v1") / "frozen.json"]
    # Giữ cả checkpoint và graph detector đã chốt, không train/export lại YOLO.
    yolo = ROOT / "runs/plate_detection/yolo26n_obb"
    for relative in (
        "weights/best.pt",
        "weights/last.pt",
        "onnx/model.onnx",
        "onnx/verified.json",
        "training_protocol.json",
    ):
        p = yolo / relative
        if p.exists():
            files.append(p)
    manifest = {p.relative_to(ROOT).as_posix(): digest(p) for p in files}
    path = ROOT / "runs/pc_transfer/current_baselines.zip"
    path.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in files:
            archive.write(p, p.relative_to(ROOT).as_posix())
        archive.writestr("transfer_manifest.json", json.dumps(manifest, indent=2))
    print("Chép riêng ZIP này và vn-plate-dataset_v2.zip sang PC:", path)
    return path


def install(path):
    import hashlib

    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("transfer_manifest.json"))
        if set(archive.namelist()) != {*manifest, "transfer_manifest.json"}:
            raise ValueError("ZIP có file không nằm trong manifest")
        prepared = []
        for relative, expected in manifest.items():
            target = (ROOT / relative).resolve()
            allowed = target.is_relative_to(
                directory("VietLPR_v1").resolve()
            ) or target.is_relative_to((ROOT / "runs/plate_detection/yolo26n_obb").resolve())
            if not allowed or target.is_symlink():
                raise ValueError("Đường dẫn transfer sai phạm vi")
            content = archive.read(relative)
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError("Transfer sai SHA")
            if target.exists() and digest(target) != expected:
                raise ValueError("Model tại đích khác; không ghi đè")
            prepared.append((target, content))
        for target, content in prepared:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    print("Đã nạp model giữ nguyên SHA; VietLPR_v1 và YOLO không cần train lại.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("create", "install"))
    parser.add_argument("path", nargs="?", type=Path)
    args = parser.parse_args()
    create() if args.action == "create" else install(args.path)

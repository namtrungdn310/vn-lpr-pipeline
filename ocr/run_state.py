"""Tách lượt train khi dữ liệu đổi, giữ nguyên trọng số và kết quả cũ."""

from datetime import datetime
from pathlib import Path
import stat

from .config import ROOT, save_json, same_training_config
from .data import read_split, split_signature


def _check_run_path(path):
    root = (ROOT / "runs").absolute()
    path = Path(path).absolute()
    if path == root or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Chỉ lưu trữ lượt train nằm bên trong runs/")
    for parent in (path, *path.parents):
        if parent.exists():
            attributes = getattr(parent.lstat(), "st_file_attributes", 0)
            if parent.is_symlink() or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
                raise ValueError(f"Không di chuyển thư mục liên kết: {parent}")
        if parent == root:
            break
    return path.resolve()


def prepare_vietlpr_run(cfg):
    """Resume cùng dữ liệu; lưu riêng cả DET/REC nếu train hoặc val đã đổi."""
    run = _check_run_path(cfg["run_dir"])
    paths = [
        (task, run / f"{task}_{kind}.pth")
        for task in ("det", "rec")
        for kind in ("last", "best")
        if (run / f"{task}_{kind}.pth").is_file()
    ]
    protocol = run / "source_protocol.json"
    state = None
    if protocol.exists():
        import json

        state = json.loads(protocol.read_text(encoding="utf-8"))
    if not paths and state is None:
        return None

    import torch

    signatures, changed, protocol_changed = {}, [], False
    if state is not None:
        if not same_training_config(state["cfg"], cfg):
            if state["cfg"].get("training_revision") != cfg.get("training_revision"):
                protocol_changed = True
                changed.append("source_protocol.json")
            else:
                raise ValueError("Cấu hình khác lượt native; dùng run_dir mới")
        for task, entry in state["tasks"].items():
            signatures[task] = {
                split: split_signature(read_split(cfg["dataset"], split, task))
                for split in ("train", "val")
            }
            if entry["smoke_only"]:
                raise ValueError("Không resume lượt smoke thành lượt chính")
            if entry["signatures"] != signatures[task]:
                changed.append(f"native/{task}_last.pth")
    for task, path in paths:
        if task not in signatures:
            signatures[task] = {
                split: split_signature(read_split(cfg["dataset"], split, task))
                for split in ("train", "val")
            }
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not same_training_config(checkpoint.get("cfg"), cfg):
            previous = checkpoint.get("cfg", {})
            if (
                cfg.get("training_revision")
                and previous.get("training_revision") != cfg["training_revision"]
            ):
                protocol_changed = True
                changed.append(path.name)
            else:
                raise ValueError(
                    f"Cấu hình khác checkpoint {path.name}. Khôi phục config của lượt cũ "
                    "hoặc chọn run_dir mới trước khi train."
                )
        if checkpoint.get("task") != task or checkpoint.get("smoke_only", False):
            raise ValueError(f"Checkpoint không đúng nhánh hoặc chỉ dùng smoke: {path}")
        if not isinstance(checkpoint.get("signatures"), dict):
            raise ValueError(f"Checkpoint thiếu thông tin dữ liệu: {path}")
        if checkpoint["signatures"] != signatures[task]:
            changed.append(path.name)
        del checkpoint
    if not changed:
        return None

    archive = _check_run_path(
        ROOT / "runs/archive" / f"{run.name}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    if archive.exists():
        raise FileExistsError(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    # Đổi tên trong cùng runs/ để giữ đủ checkpoint, ONNX, log và kết quả.
    run.rename(archive)
    comparison = ROOT / "runs/comparison_ranked.csv"
    if comparison.is_file() and not comparison.is_symlink():
        comparison = _check_run_path(comparison)
        target = archive / "previous_comparison_ranked.csv"
        if not target.exists():
            comparison.rename(target)
    save_json(
        archive / "archive_reason.json",
        {
            "reason": (
                "training_protocol_changed" if protocol_changed else "train_or_val_data_changed"
            ),
            "previous_run_dir": str(run),
            "changed_checkpoints": changed,
            "new_train_val_signatures": signatures,
        },
    )
    reason = "quy trình train đã đổi" if protocol_changed else "dữ liệu đã đổi"
    print(f"{cfg['name']}: {reason}; lưu lượt cũ tại {archive}", flush=True)
    print(f"Bắt đầu lượt mới tại {run}; không dùng trọng số cũ để resume.", flush=True)
    return archive

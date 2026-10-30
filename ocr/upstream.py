"""Nạp nguồn đã chốt; chỉ nối cấu hình, đường dẫn và split hiện có."""

import importlib
import importlib.util
from pathlib import Path
import sys

from .config import ROOT


def v1_modules():
    root = str(ROOT / "third_party/VietLPR_v1")
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("vietlpr.datasets"), importlib.import_module("vietlpr.train")


def v2_config(cfg):
    p, m = cfg["postprocess"], cfg["model"]
    return {
        "det_size": m["det_size"],
        "det_thresh": p["threshold"],
        "det_box_thresh": p["box_threshold"],
        "det_unclip": p["unclip"],
        "det_pad_x": p.get("pad_x", 0.0),
        "rec_h": m["rec_h"],
        "rec_w": m["rec_w"],
        "rec_resize": m["rec_resize"],
        "chars": cfg["chars"],
    }


def v2_source(cfg, device="cpu"):
    """Mỗi lượt có namespace riêng để cấu hình gốc không ghi đè lẫn nhau."""
    import torch

    path = ROOT / "third_party/VietLPR_v2/source.py"
    spec = importlib.util.spec_from_file_location("_vietlpr_v2_source", path)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    m, p, t = cfg["model"], cfg["postprocess"], cfg["train"]
    source.DET_SIZE, source.DET_STRIDE = m["det_size"], 2
    source.DET_M, source._SUB = m["det_size"] // 2, 16
    source.REC_H, source.REC_W, source.REC_RESIZE = m["rec_h"], m["rec_w"], m["rec_resize"]
    source.DET_SHRINK = p["shrink"]
    source.DET_THRESH, source.DET_BOX_TH = p["threshold"], p["box_threshold"]
    source.DET_UNCLIP, source.DET_PAD_X = p["unclip"], p.get("pad_x", 0.0)
    source.CHARS = cfg["chars"]
    source.CHAR2IDX = {c: i + 1 for i, c in enumerate(cfg["chars"])}
    source.DEVICE = torch.device(device)
    source.AMP = t["amp"] and source.DEVICE.type == "cuda"
    source.WEIGHT_DECAY, source.WARMUP_EPOCHS = t["weight_decay"], t["warmup_epochs"]
    source.PATIENCE, source.RESUME = t["patience"], False
    source.OUT_DIR = str(Path(cfg["run_dir"]) / "native")
    return source


def v2_items(rows, task):
    if task == "rec":
        return [(str(r["path"]), r["text"]) for r in rows]
    return [
        (
            str(r["path"]),
            [
                {"poly": q.copy(), "text": text, "ignore": False}
                for q, text in zip(r["boxes"], r["texts"])
            ],
        )
        for r in rows
    ]

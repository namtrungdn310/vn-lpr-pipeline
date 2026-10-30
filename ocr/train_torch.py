"""Train theo từng nguồn; chỉ nối split cố định và checkpoint/resume trên máy."""

import argparse
import json
import math
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from .config import load_config, save_json, same_training_config
from .data import read_split, split_signature
from .torch_models import build_model
from .upstream import v1_modules, v2_source, v2_items


def task_rows(cfg, task, limit=None):
    rows = {s: read_split(cfg["dataset"], s, task) for s in ("train", "val")}
    if limit is not None:
        if limit < 2:
            raise ValueError("Smoke cần ít nhất hai mẫu")
        rows = {s: r[:limit] for s, r in rows.items()}
    if any(set(r["text"]) - set(cfg["chars"]) for values in rows.values() for r in values):
        raise ValueError("Nhãn chứa ký tự ngoài charset nguồn; không tự bỏ ký tự")
    return rows, {s: split_signature(r) for s, r in rows.items()}


def protocol_file(cfg, task, signatures, smoke, epochs):
    path = Path(cfg["run_dir"]) / "source_protocol.json"
    record = {"cfg": cfg, "tasks": {}}
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if not same_training_config(record["cfg"], cfg):
            raise ValueError("Lượt native cũ khác cấu hình; dùng run_dir mới")
    entry = {"signatures": signatures, "smoke_only": smoke, "epochs": epochs}
    if task in record["tasks"] and record["tasks"][task] != entry:
        raise ValueError("Dữ liệu hoặc số epoch lượt native đã đổi; dùng run_dir mới")
    record["tasks"][task] = entry
    save_json(path, record)


def v1_dataset(cfg, split, task, limit):
    datasets, _ = v1_modules()
    root = Path(cfg["dataset"]) / split / task
    if task == "det":
        ds = datasets.DetDataset(
            str(root / "det_labels.txt"),
            str(root / "images"),
            img_size=cfg["model"]["det_size"],
            augment=split == "train",
        )
    else:
        ds = datasets.RecDataset(
            str(root / "rec_labels.txt"),
            str(root / "images"),
            img_size=(cfg["model"]["rec_w"], cfg["model"]["rec_h"]),
        )
    return Subset(ds, range(min(limit, len(ds)))) if limit is not None else ds


def train_v1(cfg, task, rows, signatures, device, epochs, limit, resume):
    datasets, source = v1_modules()
    if cfg["chars"] != "".join(datasets.CHAR2LABEL):
        raise ValueError("V1 dùng đúng charset và thứ tự trong constants.py gốc")
    model = build_model(cfg, task).to(device)
    criterion = source.DetLoss() if task == "det" else torch.nn.CTCLoss(blank=0, zero_infinity=True)
    loaders = {}
    for split in ("train", "val"):
        ds = v1_dataset(cfg, split, task, limit)
        loaders[split] = DataLoader(
            ds,
            batch_size=min(cfg["train"][f"{task}_batch"], len(ds)),
            shuffle=split == "train",
            num_workers=0,
            pin_memory=device.type == "cuda",
            collate_fn=datasets.collate_rec if task == "rec" else None,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"][f"{task}_lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"] and device.type == "cuda")
    run = Path(cfg["run_dir"])
    last_path = run / f"{task}_last.pth"
    start, best, history = (
        0,
        (0.0, float("inf")) if task == "det" else (float("inf"), float("inf")),
        [],
    )
    if resume:
        ck = torch.load(last_path, map_location=device, weights_only=False)
        if not same_training_config(ck["cfg"], cfg) or ck["signatures"] != signatures:
            raise ValueError("Resume khác cấu hình hoặc dữ liệu")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start, best, history = ck["epoch"], tuple(ck["best"]), ck["history"]
        if "rng" in ck:
            random.setstate(ck["rng"]["python"])
            np.random.set_state(ck["rng"]["numpy"])
            torch.set_rng_state(ck["rng"]["torch"].cpu())
            if device.type == "cuda":
                torch.cuda.set_rng_state_all([r.cpu() for r in ck["rng"]["cuda"]])
    elif last_path.exists():
        raise FileExistsError("Đã có checkpoint; dùng --resume hoặc run_dir mới")
    for epoch in range(start, epochs):
        lr = source.warmup_cosine_lr(
            optimizer, epoch, cfg["train"]["warmup_epochs"], epochs, cfg["train"][f"{task}_lr"]
        )
        model.train()
        total_loss = 0.0
        for batch in loaders["train"]:
            x = batch[0].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                output = model(x)
                if task == "det":
                    target = batch[1].to(device)
                    if output.shape[-2:] != target.shape[-2:]:
                        target = torch.nn.functional.interpolate(
                            target, size=output.shape[-2:], mode="nearest"
                        )
                    loss = criterion(output, target)
                else:
                    logp = output.permute(1, 0, 2).log_softmax(2)
                    lengths = torch.full(
                        (len(x),), output.shape[1], dtype=torch.long, device=device
                    )
                    loss = criterion(logp, batch[1].to(device), lengths, batch[2].to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Loss không hữu hạn; dừng trước khi ghi model")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        if task == "det":
            metrics = source.validate_det_model(model, loaders["val"], criterion, device)
            improved = metrics["iou"] > best[0] or (
                math.isclose(metrics["iou"], best[0]) and metrics["loss"] < best[1]
            )
            candidate = metrics["iou"], metrics["loss"]
        else:
            metrics = source.validate_rec_model(model, loaders["val"], criterion, device)
            improved = metrics["cer"] < best[0] or (
                math.isclose(metrics["cer"], best[0]) and metrics["loss"] < best[1]
            )
            candidate = metrics["cer"], metrics["loss"]
        if improved:
            best = candidate
        record = {
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": total_loss / len(loaders["train"]),
            **metrics,
        }
        history.append(record)
        ck = {
            "cfg": cfg,
            "task": task,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch + 1,
            "best": best,
            "history": history,
            "signatures": signatures,
            "smoke_only": limit is not None,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            },
        }
        torch.save(ck, last_path)
        if improved:
            torch.save(ck, run / f"{task}_best.pth")
        save_json(run / f"{task}_history.json", history)
        print(json.dumps(record), "best" if improved else "", flush=True)
    return history


def train_v2(cfg, task, rows, signatures, device, epochs, limit, resume):
    source = v2_source(cfg, str(device))
    Path(source.OUT_DIR).mkdir(parents=True, exist_ok=True)
    source.RESUME = resume
    items = {s: v2_items(r, task) for s, r in rows.items()}
    if task == "det":
        train_ds = source.DetDataset(items["train"], True)
        source_cfg = {
            "arch": "FastDBNet",
            "width": cfg["model"]["det_width"],
            "fpn_ch": cfg["model"]["det_fpn"],
            "size": cfg["model"]["det_size"],
            "stride": 2,
            "shrink": cfg["postprocess"]["shrink"],
        }
        step = source.det_step
        val_fn = lambda model: source.eval_det(model, items["val"])
        collate = None
    else:
        sets = [source.RecDataset(items["train"], True)]
        det_rows, det_signatures = task_rows(cfg, "det", limit)
        det_items = v2_items(det_rows["train"], "det")
        predictions = {}
        det_path = Path(source.OUT_DIR) / "det_best.pth"
        if det_path.exists():
            state = json.loads(
                (Path(cfg["run_dir"]) / "source_protocol.json").read_text(encoding="utf-8")
            )
            if state["tasks"]["det"]["signatures"] != det_signatures:
                raise ValueError("DET nguồn dùng dữ liệu khác; train lại DET trước REC")
            detector = build_model(cfg, "det").to(device)
            detector.load_state_dict(
                torch.load(det_path, map_location=device, weights_only=False)["model"]
            )
            predictions = source.predict_train_quads(detector, det_items)
            del detector
        sets.append(source.RecFromDetDataset(det_items, True, predictions))
        print(
            f"REC nguồn: {len(sets[0])} crop có sẵn + {len(sets[1])} crop DET; {len(predictions)} box dự đoán khớp GT",
            flush=True,
        )
        train_ds = ConcatDataset(sets)
        collate = source.rec_collate
        source_cfg = {
            "arch": "FastCRNN",
            "chars": cfg["chars"],
            "h": cfg["model"]["rec_h"],
            "w": cfg["model"]["rec_w"],
            "width": cfg["model"]["rec_width"],
            "dim": cfg["model"]["rec_dim"],
            "neck": cfg["model"]["rec_neck"],
            "neck_layers": cfg["model"].get("rec_neck_layers", 1),
            "resize": cfg["model"]["rec_resize"],
        }
        val_loader = DataLoader(
            source.RecDataset(items["val"], False),
            batch_size=cfg["train"]["rec_batch"],
            collate_fn=collate,
            num_workers=0,
        )
        step = source.rec_step
        val_fn = lambda model: {
            k: v for k, v in source.eval_rec(model, val_loader).items() if not k.startswith("_")
        }
    loader = DataLoader(
        train_ds,
        batch_size=min(cfg["train"][f"{task}_batch"], len(train_ds)),
        shuffle=True,
        drop_last=True,
        collate_fn=collate,
        num_workers=0,
        pin_memory=source.AMP,
    )
    model = build_model(cfg, task).to(device)

    def safe_step(model, batch):
        loss = step(model, batch)
        if not torch.isfinite(loss):
            raise RuntimeError("Loss không hữu hạn; dừng trước khi ghi model")
        return loss

    if (Path(source.OUT_DIR) / f"{task}_last.pth").exists() and not resume:
        raise FileExistsError("Đã có native checkpoint; dùng --resume")
    history = source.train_model(
        task, model, source_cfg, loader, safe_step, val_fn, epochs, cfg["train"][f"{task}_lr"]
    )
    for kind in ("best", "last"):
        native = torch.load(
            Path(source.OUT_DIR) / f"{task}_{kind}.pth", map_location="cpu", weights_only=False
        )
        weights = native["model"] if kind == "best" else native["ema"]
        epoch = native["metrics"]["epoch"] if kind == "best" else native["epoch"] + 1
        canonical = {
            "cfg": cfg,
            "task": task,
            "model": weights,
            "epoch": epoch,
            "signatures": signatures,
            "smoke_only": limit is not None,
            "weights_kind": "EMA",
            "upstream_checkpoint": f"native/{task}_{kind}.pth",
        }
        torch.save(canonical, Path(cfg["run_dir"]) / f"{task}_{kind}.pth")
    save_json(Path(cfg["run_dir"]) / f"{task}_history.json", history)
    return history


def train(cfg, task, *, device="cuda", epochs=None, limit=None, resume=False):
    if cfg["name"] not in ("vietlpr_v1", "vietlpr_v2"):
        raise ValueError("Paddle dùng pretrained")
    if epochs is not None and epochs < 1:
        raise ValueError("Số epoch phải dương")
    total_epochs = epochs or cfg["train"][f"{task}_epochs"]
    torch.set_num_threads(cfg["benchmark"]["threads"])
    cv2.setNumThreads(1)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Torch không nhận GPU")
    device = torch.device(device)
    rows, signatures = task_rows(cfg, task, limit)
    run = Path(cfg["run_dir"])
    run.mkdir(parents=True, exist_ok=True)
    last = run / f"{task}_last.pth"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        if not same_training_config(ck["cfg"], cfg) or ck["signatures"] != signatures:
            raise ValueError("Checkpoint khác cấu hình/dữ liệu")
        if not resume:
            raise FileExistsError("Đã có checkpoint; dùng --resume")
    protocol_file(cfg, task, signatures, limit is not None, total_epochs)
    impl = train_v1 if cfg["name"] == "vietlpr_v1" else train_v2
    return impl(cfg, task, rows, signatures, device, total_epochs, limit, resume)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("vietlpr_v1", "vietlpr_v2"), required=True)
    parser.add_argument("--task", choices=("det", "rec"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run-dir")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.model)
    if args.run_dir:
        cfg["run_dir"] = str(Path(args.run_dir).resolve())
    train(
        cfg, args.task, device=args.device, epochs=args.epochs, limit=args.limit, resume=args.resume
    )


if __name__ == "__main__":
    main()

"""Train/val hai VietLPR theo split cố định; không đụng tập test."""

import argparse
import json
import math
from pathlib import Path
import random

import cv2
import numpy as np
import pyclipper
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .config import load_config, save_json
from .data import image, read_split, split_signature
from .metrics import text_scores
from .runtime import crop_line, decode_boxes, decode_ctc, preprocess_det, resize_rec
from .torch_models import build_model


def mask_for(row, size, cfg):
    height, width = image(row["path"]).shape[:2]
    mh, mw = size
    legacy = cfg["name"] == "vietlpr_v1"
    if legacy:
        mh = mw = cfg["model"]["det_size"] // 4
    target, valid = np.zeros((mh, mw), np.float32), np.ones((mh, mw), np.float32)
    for box in row["boxes"]:
        q = (box + 0.5) * np.float32([mw / width, mh / height]) - 0.5
        if not legacy:
            area, perimeter = abs(cv2.contourArea(q)), cv2.arcLength(q, True)
            offset = pyclipper.PyclipperOffset()
            offset.AddPath(
                (q * 16).round().astype(np.int64).tolist(),
                pyclipper.JT_ROUND,
                pyclipper.ET_CLOSEDPOLYGON,
            )
            result = offset.Execute(
                -16 * area * (1 - cfg["postprocess"]["shrink"] ** 2) / max(perimeter, 1e-6)
            )
            if not result or area < 4:
                cv2.fillPoly(valid, [q.round().astype(np.int32)], 0)
                continue
            q = np.float32(max(result, key=lambda r: abs(cv2.contourArea(np.float32(r))))) / 16
        cv2.fillPoly(target, [(q * 16).round().astype(np.int32)], 1, shift=4)
    if legacy:
        target = cv2.dilate(target, np.ones((3, 3), np.uint8))
        target = cv2.resize(target, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        valid = np.ones(size, np.float32)
    return target, valid


class Samples(Dataset):
    def __init__(self, rows, cfg, task, output_shape=None, augment=False):
        self.rows, self.cfg, self.task, self.augment = rows, cfg, task, augment
        self.masks = [mask_for(row, output_shape, cfg) for row in rows] if task == "det" else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        pixels = image(row["path"])
        if self.augment and random.random() < 0.5:
            pixels = np.clip(
                pixels.astype(np.float32) * random.uniform(0.8, 1.2) + random.uniform(-15, 15),
                0,
                255,
            ).astype(np.uint8)
        if self.task == "det":
            target, valid = self.masks[index]
            return (
                torch.from_numpy(preprocess_det(pixels, self.cfg)[0]),
                torch.from_numpy(target[None]),
                torch.from_numpy(valid[None]),
            )
        if self.cfg["name"] == "vietlpr_v2":
            h, w = pixels.shape[:2]
            quad = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
            data = crop_line(pixels, quad, self.cfg)
        else:
            data = resize_rec(pixels, self.cfg)
        ids = torch.tensor([self.cfg["chars"].index(c) + 1 for c in row["text"]], dtype=torch.long)
        return torch.from_numpy(data[0]), ids, row["text"]


def collate_rec(batch):
    inputs, targets, texts = zip(*batch)
    return (
        torch.stack(inputs),
        torch.cat(targets),
        torch.tensor([len(t) for t in targets]),
        list(texts),
    )


def detection_loss(logits, target, valid, cfg):
    if cfg["name"] == "vietlpr_v1":
        return F.binary_cross_entropy_with_logits(logits, target)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    positive, negative = target * valid, (1 - target) * valid
    count_positive = int(positive.sum().item())
    count_negative = min(int(negative.sum().item()), max(count_positive * 3, 64))
    negative_loss = (
        torch.topk((bce * negative).flatten(), count_negative).values.sum()
        if count_negative
        else bce.sum() * 0
    )
    balanced = ((bce * positive).sum() + negative_loss) / (count_positive + count_negative + 1e-6)
    probability = logits.sigmoid()
    dice = 1 - 2 * (probability * target * valid).sum() / (
        (probability * valid).sum() + positive.sum() + 1e-6
    )
    return balanced + dice


def matched_counts(predictions, truths):
    candidates = []
    for i, pred in enumerate(predictions):
        a = cv2.convexHull(np.float32(pred))
        for j, truth in enumerate(truths):
            b = cv2.convexHull(np.float32(truth))
            intersection = cv2.intersectConvexConvex(a, b)[0]
            union = cv2.contourArea(a) + cv2.contourArea(b) - intersection
            candidates.append((intersection / max(union, 1e-6), i, j))
    used_p, used_g = set(), set()
    for iou, i, j in sorted(candidates, reverse=True):
        if iou >= 0.5 and i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
    return len(used_p), len(predictions) - len(used_p), len(truths) - len(used_g)


@torch.inference_mode()
def validate_model(model, loader, rows, cfg, task, device):
    model.eval()
    if task == "rec":
        predictions, references = [], []
        for inputs, _, _, texts in loader:
            output = model(inputs.to(device)).float().cpu().numpy()
            predictions.extend(
                decode_ctc(output[i : i + 1], cfg["chars"]) for i in range(len(output))
            )
            references.extend(texts)
        result = text_scores(predictions, references)
        result["selection"] = (result["ACC"], -result["CER"])
        return result
    tp = fp = fn = 0
    for row in rows:
        pixels = image(row["path"])
        data = torch.from_numpy(preprocess_det(pixels, cfg)).to(device)
        probability = model(data).sigmoid()[0, 0].cpu().numpy()
        boxes = [quad for quad, _ in decode_boxes(probability, pixels, cfg)]
        a, b, c = matched_counts(boxes, row["boxes"])
        tp, fp, fn = tp + a, fp + b, fn + c
    precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "selection": (f1,)}


def train(cfg, task, *, device="cuda", epochs=None, limit=None, resume=False):
    if epochs is not None and epochs < 1:
        raise ValueError("Số epoch phải dương")
    if cfg["name"] == "paddleocr":
        raise ValueError("PaddleOCR dùng experiments.paddle")
    torch.set_num_threads(cfg["benchmark"]["threads"])
    cv2.setNumThreads(1)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "Môi trường hiện tại không có torch CUDA; chọn --device cpu hoặc cài torch CUDA"
        )
    device = torch.device(device)
    rows = {split: read_split(cfg["dataset"], split, task) for split in ("train", "val")}
    if limit is not None:
        if limit < 2:
            raise ValueError("Chạy thử cần ít nhất hai mẫu")
        rows = {split: values[:limit] for split, values in rows.items()}
    model = build_model(cfg, task).to(device)
    model.eval()
    with torch.inference_mode():
        m = cfg["model"]
        shape = (m["det_size"], m["det_size"]) if task == "det" else (m["rec_h"], m["rec_w"])
        example = model(torch.zeros(1, 3, *shape, device=device))
    out_shape = tuple(example.shape[-2:]) if task == "det" else None
    batch = min(cfg["train"][f"{task}_batch"], len(rows["train"]))
    loaders = {}
    for split in ("train", "val"):
        dataset = Samples(rows[split], cfg, task, out_shape, augment=split == "train")
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch,
            shuffle=split == "train",
            drop_last=split == "train" and len(dataset) % batch == 1,
            num_workers=cfg["train"]["workers"],
            collate_fn=collate_rec if task == "rec" else None,
        )
    run = Path(cfg["run_dir"])
    run.mkdir(parents=True, exist_ok=True)
    signatures = {split: split_signature(values) for split, values in rows.items()}
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"][f"{task}_lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    total_epochs = epochs or cfg["train"][f"{task}_epochs"]
    if total_epochs < 1:
        raise ValueError("Số epoch phải dương")
    amp = cfg["train"]["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ctc = nn.CTCLoss(blank=0, zero_infinity=False)
    start, stale, best, history = 0, 0, (-float("inf"),), []
    last_path = run / f"{task}_last.pth"
    smoke_only = limit is not None
    if resume:
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint["cfg"] != cfg or checkpoint["signatures"] != signatures:
            raise ValueError("Resume khác cấu hình hoặc dữ liệu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        smoke_only = smoke_only or checkpoint.get("smoke_only", False)
        start, stale, best, history = (
            checkpoint["epoch"],
            checkpoint["stale"],
            tuple(checkpoint["best"]),
            checkpoint["history"],
        )
    elif last_path.exists():
        raise FileExistsError("Đã có checkpoint; dùng --resume hoặc chọn run_dir mới")
    for epoch in range(start, total_epochs):
        model.train()
        losses = []
        lr_factor = 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
        for group in optimizer.param_groups:
            group["lr"] = cfg["train"][f"{task}_lr"] * lr_factor
        for pack in loaders["train"]:
            inputs = pack[0].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                output = model(inputs)
                if task == "det":
                    loss = detection_loss(output, pack[1].to(device), pack[2].to(device), cfg)
            if task == "rec":
                required = [
                    len(text) + sum(a == b for a, b in zip(text, text[1:])) for text in pack[3]
                ]
                if max(required) > output.shape[1]:
                    raise ValueError("REC quá hẹp để CTC đọc đủ nhãn; tăng rec_w")
                lengths = torch.full((len(inputs),), output.shape[1], dtype=torch.long)
                loss = ctc(
                    output.float().log_softmax(-1).permute(1, 0, 2),
                    pack[1].to(device),
                    lengths,
                    pack[2],
                )
            if not torch.isfinite(loss):
                raise RuntimeError("Loss không hữu hạn; dừng để tránh checkpoint sai")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        metrics = validate_model(model, loaders["val"], rows["val"], cfg, task, device)
        score = tuple(metrics.pop("selection"))
        improved = score > best
        stale = 0 if improved else stale + 1
        best = score if improved else best
        record = {"epoch": epoch + 1, "train_loss": sum(losses) / max(1, len(losses)), **metrics}
        history.append(record)
        checkpoint = {
            "cfg": cfg,
            "task": task,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch + 1,
            "stale": stale,
            "best": best,
            "signatures": signatures,
            "history": history,
            "smoke_only": smoke_only,
        }
        torch.save(checkpoint, last_path)
        if improved:
            torch.save(checkpoint, run / f"{task}_best.pth")
        save_json(run / f"{task}_history.json", history)
        print(json.dumps(record), "best" if improved else "", flush=True)
        if stale >= cfg["train"]["patience"]:
            break
    return history


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

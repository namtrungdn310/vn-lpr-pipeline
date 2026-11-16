"""Fine-tune LPRNet nguồn bằng CTC; chọn best theo val toàn biển."""

import random
import urllib.request

import numpy as np
import torch
from torch import nn

from ocr.config import ROOT, save_json
from ocr.data import image
from ocr.metrics import text_scores
from ocr.public_vn_runtime import model_input, decode_lprnet
from ocr.public_vn_export import definitions, explicit_pools
from .common import settings, directory, read_json, digest, freeze
from .data import rows


def source_model():
    catalog = read_json(ROOT / "configs/ocr/public_vn.json")
    variant = next(v for v in catalog["variants"] if v["family"] == "lprnet")
    source = next(s for s in catalog["source_files"] if s["filename"] == "lprnet/LPRNet.py")
    folder = directory("LPRNet_VN") / "source"
    folder.mkdir(parents=True, exist_ok=True)
    for spec, filename in [
        (source, "LPRNet.py"),
        (variant["weights"], variant["weights"]["filename"]),
    ]:
        path = folder / filename
        if not path.exists():
            with urllib.request.urlopen(spec["url"], timeout=90) as response:
                path.write_bytes(response.read())
        if digest(path) != spec["sha256"]:
            raise ValueError("LPRNet nguồn hoặc trọng số sai SHA")
    namespace = {"torch": torch, "nn": nn, "__name__": "lprnet_source"}
    definitions(folder / "LPRNet.py", namespace, {"small_basic_block", "LPRNet", "build_lprnet"})
    model = namespace["build_lprnet"](
        lpr_max_len=9, phase="train", class_num=len(variant["chars"]), dropout_rate=0.5
    )
    model.load_state_dict(
        torch.load(folder / variant["weights"]["filename"], map_location="cpu", weights_only=True),
        strict=True,
    )
    return model, variant


def train():
    if not torch.cuda.is_available():
        raise RuntimeError("LPRNet cần CUDA; không train trên CPU laptop")
    cfg = settings()
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    model, variant = source_model()
    model.cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["training"]["epochs"])
    loss_fn = nn.CTCLoss(blank=len(variant["chars"]) - 1, zero_infinity=False)
    base = directory("LPRNet_VN")
    state = {"epoch": 0, "best": [-1, -1], "wait": 0, "config": cfg}
    if (base / "last.pth").exists():
        checkpoint = torch.load(base / "last.pth", map_location="cpu", weights_only=False)
        state = checkpoint["state"]
        if state["config"] != cfg:
            raise ValueError("Cấu hình khác checkpoint LPRNet")
        if state.get("complete"):
            return
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        random.setstate(checkpoint["rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    train_rows, val_rows = rows("train"), rows("val")
    lookup = {c: i for i, c in enumerate(variant["chars"][:-1])}
    if any(set(row["text"]) - lookup.keys() for row in train_rows + val_rows):
        raise ValueError("LPRNet charset không phủ nhãn")
    for epoch in range(state["epoch"], cfg["training"]["epochs"]):
        model.train()
        epoch_rows = train_rows.copy()
        random.shuffle(epoch_rows)
        losses = []
        batch = cfg["training"]["rec_batch"]
        for start in range(0, len(epoch_rows), batch):
            selected = epoch_rows[start : start + batch]
            x = torch.from_numpy(
                np.concatenate([model_input(image(r["path"]), variant) for r in selected])
            ).cuda()
            targets = torch.tensor(
                [lookup[c] for r in selected for c in r["text"]], dtype=torch.long
            )
            lengths = torch.tensor([len(r["text"]) for r in selected], dtype=torch.long)
            output = model(x).log_softmax(1).permute(2, 0, 1)
            if any(
                len(r["text"]) + sum(a == b for a, b in zip(r["text"], r["text"][1:]))
                > output.shape[0]
                for r in selected
            ):
                raise ValueError("CTC không đủ timestep cho nhãn; không bỏ mẫu")
            loss = loss_fn(
                output,
                targets,
                torch.full((len(selected),), output.shape[0], dtype=torch.long),
                lengths,
            )
            if not torch.isfinite(loss):
                raise ValueError("LPRNet loss NaN/Inf")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        model.eval()
        predictions = []
        with torch.inference_mode():
            # Native có global normalization theo batch; val/inference phải batch 1.
            for row in val_rows:
                output = (
                    model(torch.from_numpy(model_input(image(row["path"]), variant)).cuda())
                    .cpu()
                    .numpy()
                )
                predictions.append(decode_lprnet(output, variant["chars"]))
        scores = text_scores(predictions, [r["text"] for r in val_rows])
        criterion = [scores["ACC"], -scores["CER"]]
        state["wait"] += 1
        if criterion > state["best"]:
            state.update(best=criterion, wait=0, best_val=scores, best_epoch=epoch + 1)
            torch.save(
                {k: v.detach().cpu() for k, v in model.state_dict().items()}, base / "best.pth"
            )
        state["epoch"] = epoch + 1
        state["complete"] = (
            state["wait"] >= cfg["training"]["patience"] or epoch + 1 == cfg["training"]["epochs"]
        )
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "state": state,
            "rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        }
        torch.save(checkpoint, base / "last.partial.pth")
        (base / "last.partial.pth").replace(base / "last.pth")
        save_json(base / "progress.json", state)
        print(
            f"LPRNet epoch {epoch+1}: loss={np.mean(losses):.4f}, val ACC={scores['ACC']:.2f} CER={scores['CER']:.2f}",
            flush=True,
        )
        if state["complete"]:
            break


def export():
    import copy
    import onnxruntime as ort

    base = directory("LPRNet_VN")
    state = read_json(base / "progress.json")
    if not state.get("complete"):
        raise ValueError("Chưa train xong LPRNet")
    model, variant = source_model()
    model.load_state_dict(torch.load(base / "best.pth", map_location="cpu", weights_only=True))
    model.eval()
    original = copy.deepcopy(model)
    explicit_pools(model)
    out = base / "onnx"
    out.mkdir(exist_ok=True)
    sample = torch.from_numpy(model_input(image(rows("val")[0]["path"]), variant))
    with torch.inference_mode():
        torch.testing.assert_close(model(sample), original(sample))
    torch.onnx.export(
        model,
        sample,
        str(out / "model.onnx"),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )
    engine = ort.InferenceSession(str(out / "model.onnx"), providers=["CPUExecutionProvider"])
    checks = []
    for row in rows("val")[:3]:
        data = model_input(image(row["path"]), variant)
        with torch.inference_mode():
            reference = model(torch.from_numpy(data)).numpy()
        actual = engine.run(None, {"input": data})[0]
        np.testing.assert_allclose(actual, reference, rtol=1e-3, atol=1e-4)
        checks.append(
            {"image": row["relative"], "max_abs": float(np.max(np.abs(actual - reference)))}
        )
    save_json(out / "runtime.json", {"family": "lprnet", "chars": variant["chars"]})
    return freeze(
        "LPRNet_VN",
        {
            "best_val": state["best_val"],
            "conversion_checks": checks,
            "source_revision": variant["revision"],
            "initialization": variant["weights"],
        },
    )

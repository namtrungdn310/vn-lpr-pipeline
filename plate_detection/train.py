"""Train GPU, chọn best trên val và xuất ONNX YOLO26-OBB FP32."""

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np

from ocr.config import ROOT, save_json
from ocr.paddle_suite_state import model_lock
from .data import audit_dataset, dataset_signature, load_config, read_image, sha256


def compare_outputs(native, actual, confidence):
    """Ghép hộp trước khi so số; top-k có thể đổi thứ tự khi điểm bằng nhau."""
    from scipy.optimize import linear_sum_assignment

    if native.shape != actual.shape or actual.ndim != 3 or actual.shape[0] != 1 or actual.shape[-1] != 7:
        raise ValueError("Graph xuất chưa đúng YOLO26 end-to-end OBB")
    a, b = native[0], actual[0]
    cost = np.abs(a[:, None, :4] - b[None, :, :4]).sum(axis=-1)
    cost += 1000 * (a[:, None, 5] != b[None, :, 5])
    cost += 10 * np.abs(a[:, None, 4] - b[None, :, 4])
    cost += 10 * np.abs(a[:, None, 6] - b[None, :, 6])
    ai, bi = linear_sum_assignment(cost)
    a, b = a[ai], b[bi]
    selected = (a[:, 4] >= confidence) | (b[:, 4] >= confidence)
    np.testing.assert_allclose(a[selected], b[selected], rtol=2e-3, atol=2e-3)
    return int(selected.sum()), float(np.max(np.abs(a[:, 4] - b[:, 4])))


def ultralytics_api(cfg):
    settings_dir = ROOT / "runs/plate_detection/settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(settings_dir))
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import ultralytics
    if ultralytics.__version__ != cfg["ultralytics"]:
        raise RuntimeError("Cài đúng requirements/detection.txt trước khi chạy")
    from ultralytics import YOLO, settings
    settings.update({"sync": False, "wandb": False, "mlflow": False, "clearml": False, "comet": False,
                     "hub": False})
    return YOLO


def train(cfg):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Train yêu cầu GPU CUDA; không tự chuyển sang CPU")
    report = audit_dataset(cfg["dataset"])
    if not report["ready"]:
        raise ValueError("Dataset OBB chưa đạt: " + "; ".join(report["errors"][:5]))
    from .quality import REVIEW
    source = json.loads((Path(cfg["dataset"]) / "source_audit.json").read_text(encoding="utf-8"))
    if source.get("split_review_sha256") != sha256(REVIEW):
        raise ValueError("Cần chuẩn hoá và áp dụng nhóm trùng đã duyệt trước khi train")
    signature = dataset_signature(cfg["dataset"])
    checked = json.loads((Path(cfg["dataset"]) / "normalization_check.json").read_text(encoding="utf-8"))
    if not checked.get("ready") or checked["dataset_signature"] != signature:
        raise ValueError("Bản chuẩn hoá chưa được kiểm hoặc đã đổi; chạy plate_detection.data check")
    YOLO = ultralytics_api(cfg)
    directory = Path(cfg["run_dir"])
    state_file = directory / "training_protocol.json"
    protocol = {"model": cfg["model"], "ultralytics": cfg["ultralytics"], "train": cfg["train"],
                "seed": cfg["seed"], "dataset_signature": signature, "torch": torch.__version__}
    with model_lock(directory):
        if state_file.is_file():
            if json.loads(state_file.read_text(encoding="utf-8")) != protocol:
                raise ValueError("Config/dữ liệu thay đổi; chọn run_dir mới, không ghi đè lượt cũ")
        elif any((directory / n).exists() for n in ("weights", "results.csv", "args.yaml")):
            raise ValueError("Thư mục có lượt train không rõ nguồn; chọn run_dir mới")
        save_json(state_file, protocol)
        if (directory / "training_complete.json").is_file():
            done = json.loads((directory / "training_complete.json").read_text(encoding="utf-8"))
            if sha256(directory / "weights/best.pt") != done["best_sha256"]:
                raise ValueError("best.pt thay đổi sau khi train")
            print("Lượt train đã hoàn tất; dùng lại best.pt.", flush=True)
            return directory / "weights/best.pt"
        last = directory / "weights/last.pt"
        if last.is_file():
            model = YOLO(str(last), task="obb")
            model.train(resume=True, device=cfg["train"]["device"])
        else:
            checkpoint = ROOT / "runs/plate_detection/pretrained" / cfg["model"]
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            model = YOLO(str(checkpoint), task="obb")
            model.train(data=str(Path(cfg["dataset"]) / "data.yaml"), project=str(directory.parent),
                        name=directory.name, exist_ok=True, seed=cfg["seed"], deterministic=True,
                        **cfg["train"])
        save_json(directory / "training_complete.json", {
            "best_sha256": sha256(directory / "weights/best.pt"),
            "selection": "Ultralytics OBB val fitness (mAP50-95); test chưa dùng để chọn model",
            "dataset_signature": signature})
        return directory / "weights/best.pt"


def export(cfg):
    from .runtime import PlateDetector

    YOLO = ultralytics_api(cfg)
    directory = Path(cfg["run_dir"])
    protocol = json.loads((directory / "training_protocol.json").read_text(encoding="utf-8"))
    if protocol["dataset_signature"] != dataset_signature(cfg["dataset"]):
        raise ValueError("Dữ liệu đã đổi sau train")
    model = YOLO(str(directory / "weights/best.pt"), task="obb")
    if model.task != "obb" or list(model.names.values()) != ["license-plate"]:
        raise ValueError("Cần model OBB đã train một lớp biển số")
    path = Path(model.export(format="onnx", imgsz=cfg["train"]["imgsz"], batch=1, dynamic=False,
                             half=False, simplify=False, opset=17, nms=False, device="cpu"))
    target = directory / "onnx/model.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    inf = cfg["inference"]
    runtime = PlateDetector(target, size=cfg["train"]["imgsz"], confidence=inf["confidence"],
                            max_det=inf["max_det"], threads=inf["threads"])
    samples = audit_dataset(cfg["dataset"])["records"]
    samples = [r for r in samples if r["split"] == "val"][:3]
    checks = []
    import torch
    from .runtime import letterbox
    for row in samples:
        image = read_image(row["image"])
        data = letterbox(image, cfg["train"]["imgsz"])[0]
        model.model.float().cpu().eval()
        with torch.no_grad():
            native = model.model(torch.from_numpy(data))
        native = native[0] if isinstance(native, (tuple, list)) else native
        actual = runtime.engine.run(None, {runtime.input: data})[0]
        native = native.detach().cpu().numpy()
        count, score_error = compare_outputs(native, actual, inf["confidence"])
        checks.append({"image": row["relative"], "selected_boxes": count,
                       "max_score_error": score_error})
    if not checks or not any(c["selected_boxes"] for c in checks):
        raise ValueError("Chưa có hộp val để đối chiếu PT/ONNX; cần rà model trước khi dùng")
    record = {"onnx_sha256": sha256(target), "best_sha256": sha256(directory / "weights/best.pt"),
              "dataset_signature": protocol["dataset_signature"], "config": cfg,
              "checks": checks, "precision": "FP32", "output": "cx,cy,w,h,score,class,angle",
              "end2end": True, "selection_split": "val"}
    save_json(directory / "onnx/verified.json", record)
    return target


def validate(cfg, split="val"):
    YOLO = ultralytics_api(cfg)
    model = YOLO(str(Path(cfg["run_dir"]) / "weights/best.pt"), task="obb")
    result = model.val(data=str(Path(cfg["dataset"]) / "data.yaml"), split=split,
                       imgsz=cfg["train"]["imgsz"], device=cfg["train"]["device"], batch=4,
                       workers=0, project=cfg["run_dir"], name=f"metrics_{split}")
    save_json(Path(cfg["run_dir"]) / f"detection_{split}.json", {
        "split": split, "mAP50": float(result.box.map50), "mAP50-95": float(result.box.map),
        "precision": float(result.box.mp), "recall": float(result.box.mr)})
    return result.results_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("train", "export", "val", "test"))
    parser.add_argument("--variant", choices=tuple(load_config()["variants"]), default="n")
    args = parser.parse_args()
    cfg = load_config(variant=args.variant)
    if args.action == "train":
        print(train(cfg))
    elif args.action == "export":
        print(export(cfg))
    else:
        print(validate(cfg, "test" if args.action == "test" else "val"))


if __name__ == "__main__":
    main()

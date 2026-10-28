"""Chuẩn bị cấu hình, train/val và xuất PaddleOCR PP-OCRv4 mobile."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml

from .config import ROOT, load_config, save_json
from .data import image, read_split, split_signature

SOURCE = ROOT / "third_party/PaddleOCR"


def data_signatures(cfg, limit):
    return {
        f"{task}_{split}": split_signature(read_split(cfg["dataset"], split, task)[:limit])
        for task in ("det", "rec")
        for split in ("train", "val")
    }


def prepared_state(cfg):
    """Chặn dùng nhầm cấu hình hoặc dữ liệu sau khi đã chuẩn bị YAML."""
    run = Path(cfg["run_dir"])
    state = json.loads((run / "prepared.json").read_text(encoding="utf-8"))
    if state["config"] != cfg or state["signatures"] != data_signatures(cfg, state["limit"]):
        raise ValueError("Cấu hình hoặc dữ liệu đã đổi; chọn run_dir mới và prepare lại")
    revision = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != state["upstream_revision"]:
        raise ValueError("Source PaddleOCR đã đổi sau prepare")
    for name, expected in state["files"].items():
        if hashlib.sha256((run / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"File đã đổi sau prepare: {name}")
    return state


def prepare(cfg, *, device="gpu", epochs=None, limit=None):
    if epochs is not None and epochs < 1 or limit is not None and limit < 2:
        raise ValueError("Epoch phải dương; chạy thử cần ít nhất hai mẫu")
    if not (SOURCE / "tools/train.py").is_file():
        raise FileNotFoundError("Chạy python -m experiments.bootstrap trước")
    revision = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != cfg["upstream"]["revision"]:
        raise ValueError("PaddleOCR upstream khác phiên bản đã chốt")
    run = Path(cfg["run_dir"])
    # Không cho lượt train cũ bị gắn cấu hình hoặc nhãn smoke khác.
    if run.exists() and (any(run.rglob("*.pdparams")) or (run / "onnx").exists()):
        raise FileExistsError("Đã có weights Paddle; giữ YAML hiện có hoặc dùng run_dir mới")
    run.mkdir(parents=True, exist_ok=True)
    dictionary = run / "chars.txt"
    dictionary.write_text("\n".join(cfg["chars"]) + "\n", encoding="utf-8", newline="\n")
    result = {}
    for task in ("det", "rec"):
        template = SOURCE / cfg["upstream"][f"{task}_config"]
        config = yaml.safe_load(template.read_text(encoding="utf-8"))
        count = None
        for split, section in (("train", "Train"), ("val", "Eval")):
            rows = read_split(cfg["dataset"], split, task)
            if limit is not None:
                rows = rows[:limit]
            lines = []
            for row in rows:
                if task == "det":
                    payload = json.dumps(
                        [
                            {"transcription": text, "points": box.tolist()}
                            for box, text in zip(row["boxes"], row["texts"])
                        ]
                    )
                else:
                    payload = row["text"]
                lines.append(str(row["path"]).replace("\\", "/") + "\t" + payload)
            label = run / f"{task}_{split}.txt"
            label.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
            dataset = config[section]["dataset"]
            dataset.update(data_dir=str(Path(cfg["dataset"])), label_file_list=[str(label)])
            if split == "train":
                count = len(rows)
            loader = config[section]["loader"]
            loader.update(
                num_workers=cfg["train"]["workers"],
                batch_size_per_card=cfg["train"][f"{task}_batch"] if split == "train" else 1,
                shuffle=split == "train",
                drop_last=False,
            )
            if limit is not None and split == "train":
                loader["batch_size_per_card"] = min(
                    loader["batch_size_per_card"], max(1, len(rows) // 2)
                )
        total = epochs or cfg["train"][f"{task}_epochs"]
        global_cfg = config["Global"]
        global_cfg.update(
            use_gpu=device == "gpu",
            distributed=False,
            seed=cfg["seed"],
            epoch_num=total,
            pretrained_model=None,
            checkpoints=None,
            use_visualdl=False,
            save_model_dir=str(run / task),
            save_inference_dir=str(run / f"inference_{task}"),
            save_epoch_step=1,
            eval_batch_step=[0, max(1, math.ceil(count / cfg["train"][f"{task}_batch"]))],
        )
        if limit is not None:
            global_cfg["eval_batch_step"] = [0, 1]
        if task == "det":
            # Giữ DB/PPLCNetV3, thu input về kích thước phù hợp ảnh biển.
            transforms = config["Train"]["dataset"]["transforms"]
            transforms[:] = [t for t in transforms if "CopyPaste" not in t]
            for transform in transforms:
                if "IaaAugment" in transform:
                    transform["IaaAugment"]["augmenter_args"] = [
                        {"type": "Affine", "args": {"rotate": [-5, 5]}}
                    ]
                if "EastRandomCropData" in transform:
                    transform["EastRandomCropData"]["size"] = [cfg["model"]["det_size"]] * 2
                for name in ("MakeShrinkMap", "MakeBorderMap"):
                    if name in transform:
                        transform[name]["total_epoch"] = total
            for transform in config["Eval"]["dataset"]["transforms"]:
                if "DetResizeForTest" in transform:
                    transform["DetResizeForTest"] = {"image_shape": [cfg["model"]["det_size"]] * 2}
            config["PostProcess"].update(
                thresh=cfg["postprocess"]["threshold"],
                box_thresh=cfg["postprocess"]["box_threshold"],
                unclip_ratio=cfg["postprocess"]["unclip"],
            )
        else:
            global_cfg.update(
                character_dict_path=str(dictionary), use_space_char=False, max_text_length=16
            )
            config["Train"].pop("sampler", None)
            config["Train"]["dataset"] = {
                "name": "SimpleDataSet",
                "data_dir": cfg["dataset"],
                "label_file_list": config["Train"]["dataset"]["label_file_list"],
                "transforms": [
                    {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                    {"RecAug": None},
                    {"MultiLabelEncode": {"gtc_encode": "NRTRLabelEncode"}},
                    {
                        "RecResizeImg": {
                            "image_shape": [3, cfg["model"]["rec_h"], cfg["model"]["rec_w"]]
                        }
                    },
                    {
                        "KeepKeys": {
                            "keep_keys": [
                                "image",
                                "label_ctc",
                                "label_gtc",
                                "length",
                                "valid_ratio",
                            ]
                        }
                    },
                ],
            }
            config["Architecture"]["Head"]["head_list"][1]["NRTRHead"]["max_text_length"] = 16
        path = run / f"{task}.yml"
        path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        result[task] = path
    save_json(
        run / "prepared.json",
        {
            "upstream_revision": revision,
            "smoke_only": limit is not None,
            "limit": limit,
            "config": cfg,
            "signatures": data_signatures(cfg, limit),
            "files": {
                name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                for name in (
                    "chars.txt",
                    "det.yml",
                    "rec.yml",
                    "det_train.txt",
                    "det_val.txt",
                    "rec_train.txt",
                    "rec_val.txt",
                )
            },
        },
    )
    return result


def execute(cfg, action, task, *, device="gpu", epochs=None, limit=None, resume=False):
    environment = os.environ.copy()
    environment.setdefault("FLAGS_allocator_strategy", "auto_growth")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[name] = str(cfg["benchmark"]["threads"])
    run = Path(cfg["run_dir"])
    paths = (
        prepare(cfg, device=device, epochs=epochs, limit=limit)
        if action == "prepare"
        else {t: run / f"{t}.yml" for t in ("det", "rec")}
    )
    if action == "prepare":
        return paths
    state = prepared_state(cfg)
    if action in ("train", "val"):
        tool = "train.py" if action == "train" else "eval.py"
        command = [sys.executable, str(SOURCE / "tools" / tool), "-c", str(paths[task])]
        if action == "val":
            command += ["-o", f"Global.checkpoints={run / task / 'best_accuracy'}"]
        elif resume:
            command += ["-o", f"Global.checkpoints={run / task / 'latest'}"]
        elif (run / task / "latest.pdparams").exists():
            raise FileExistsError("Đã có checkpoint Paddle; dùng --resume hoặc run_dir mới")
        subprocess.run(command, cwd=SOURCE, check=True, env=environment)
        return
    export_dir = run / f"inference_{task}"
    subprocess.run(
        [
            sys.executable,
            str(SOURCE / "tools/export_model.py"),
            "-c",
            str(paths[task]),
            "-o",
            f"Global.pretrained_model={run / task / 'best_accuracy'}",
            f"Global.save_inference_dir={export_dir}",
        ],
        cwd=SOURCE,
        check=True,
        env=environment,
    )
    model_file = next(export_dir.rglob("*.pdmodel"), None)
    if model_file is None:
        raise FileNotFoundError("Không tìm thấy inference.pdmodel sau export")
    destination = run / "onnx" / f"{task}.onnx"
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "paddle2onnx.command",
            "--model_dir",
            str(model_file.parent),
            "--model_filename",
            model_file.name,
            "--params_filename",
            model_file.with_suffix(".pdiparams").name,
            "--save_file",
            str(destination),
            "--opset_version",
            "17",
            "--enable_onnx_checker",
            "True",
        ],
        check=True,
        env=environment,
    )
    save_json(destination.parent / "config.json", cfg)
    # So Paddle inference với ONNX trên ảnh val, không dùng test để kiểm chuyển đổi.
    import onnxruntime as ort
    from paddle.inference import Config, create_predictor
    from .runtime import preprocess_det, resize_rec

    native = Config(str(model_file), str(model_file.with_suffix(".pdiparams")))
    native.disable_gpu()
    native.set_cpu_math_library_num_threads(cfg["benchmark"]["threads"])
    native.disable_glog_info()
    predictor = create_predictor(native)
    options = ort.SessionOptions()
    options.intra_op_num_threads = cfg["benchmark"]["threads"]
    session = ort.InferenceSession(str(destination), options, providers=["CPUExecutionProvider"])
    import numpy as np

    for row in read_split(cfg["dataset"], "val", task)[:3]:
        pixels = image(row["path"])
        inputs = preprocess_det(pixels, cfg) if task == "det" else resize_rec(pixels, cfg)
        handle = predictor.get_input_handle(predictor.get_input_names()[0])
        handle.reshape(inputs.shape)
        handle.copy_from_cpu(inputs)
        predictor.run()
        expected = predictor.get_output_handle(predictor.get_output_names()[0]).copy_to_cpu()
        actual = session.run(None, {session.get_inputs()[0].name: inputs})[0]
        np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-3)
    metadata_path = destination.parent / "export_check.json"
    checks = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    checks[task] = {
        "checked_on_val": True,
        "smoke_only": state["smoke_only"],
        "signatures": state["signatures"],
    }
    save_json(metadata_path, checks)
    return destination


def smoke(*, device="cpu", full_size=False):
    """Kiểm DET/REC một epoch trong thư mục tạm, chỉ đọc train và val."""
    from .runtime import onnx_pipeline

    with tempfile.TemporaryDirectory(prefix="paddleocr-smoke-") as directory:
        cfg = load_config("paddleocr")
        cfg["run_dir"] = directory
        if not full_size:
            cfg["model"]["det_size"] = 64
        cfg["train"]["det_batch"] = cfg["train"]["rec_batch"] = 2
        prepare(cfg, device=device, epochs=1, limit=4)
        for task in ("det", "rec"):
            execute(cfg, "train", task)
            execute(cfg, "export", task)
        sample = read_split(cfg["dataset"], "val", "det")[0]
        result = onnx_pipeline(cfg)(image(sample["path"]))
        assert isinstance(result["text"], str)
        print("PaddleOCR", device, "train/val/export/ONNX pipeline OK", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train", "val", "export", "smoke"))
    parser.add_argument("--task", choices=("det", "rec"), default="det")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--full-size", action="store_true", help="Giữ input DET khi chạy thử")
    args = parser.parse_args()
    if args.action == "smoke":
        smoke(device=args.device, full_size=args.full_size)
        return
    cfg = load_config("paddleocr")
    if args.run_dir:
        cfg["run_dir"] = str(Path(args.run_dir).resolve())
    print(
        execute(
            cfg,
            args.action,
            args.task,
            device=args.device,
            epochs=args.epochs,
            limit=args.limit,
            resume=args.resume,
        )
    )


if __name__ == "__main__":
    main()

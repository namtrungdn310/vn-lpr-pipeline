"""Fine-tune DET/REC bằng vòng train PaddleOCR gốc."""

import math
import os
import shutil
import subprocess
import sys

import yaml

from ocr.config import ROOT, save_json
from .common import settings, entry, directory, read_json, digest, freeze
from .sources import checkout, asset
from .data import rows


def config_for(name, task, source):
    cfg, model = settings(), entry(name)
    path = source / f"configs/{task}/{model['version']}/{name}_{task}.yml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    run = directory(name) / task
    training = cfg["training"]
    batch = training["det_batch" if task == "det" else "rec_batch"]
    global_ = config["Global"]
    global_.update(
        use_gpu=True,
        epoch_num=training["epochs"],
        distributed=False,
        use_amp=False,
        seed=cfg["seed"],
        save_model_dir=str(run),
        save_epoch_step=1,
        pretrained_model=str(asset(f"{name}_{task}_pretrained.pdparams")),
        checkpoints=None,
        eval_batch_step=[0, max(1, math.ceil(len(rows("train", task)) / batch))],
        use_visualdl=False,
    )
    config["Optimizer"]["lr"]["learning_rate"] = training["lr"]
    for split, key in (("train", "Train"), ("val", "Eval")):
        data = config[key]["dataset"]
        data["data_dir"] = str(ROOT)
        data["label_file_list"] = [str(directory() / "data" / f"{split}_{task}.txt")]
        loader = config[key]["loader"]
        loader.update(
            num_workers=0, batch_size_per_card=batch if split == "train" else 1, drop_last=False
        )
        if "sampler" in config[key]:
            config[key]["sampler"]["first_bs"] = batch
        for op in data["transforms"]:
            # Không lật ngang chữ; giữ các augment còn lại của cấu hình nguồn.
            if op.get("IaaAugment"):
                op["IaaAugment"]["augmenter_args"] = [
                    a for a in op["IaaAugment"]["augmenter_args"] if a["type"] != "Fliplr"
                ]
            for value in op.values():
                if isinstance(value, dict) and "total_epoch" in value:
                    value["total_epoch"] = training["epochs"]
    # Dùng nguyên dictionary nguồn để nạp được head pretrained, không đổi số lớp.
    if global_.get("character_dict_path"):
        global_["character_dict_path"] = str(source / global_["character_dict_path"])
    target = directory(name) / f"{task}.yml"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and yaml.safe_load(target.read_text(encoding="utf-8")) != config:
        raise ValueError("Config đã đổi; không resume lên lượt khác cấu hình")
    target.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return target, config


def execute(command, source):
    subprocess.run(
        [str(x) for x in command],
        cwd=source,
        check=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "FLAGS_enable_pir_api": "1"},
    )


def train(name):
    import paddle

    if not paddle.is_compiled_with_cuda():
        raise RuntimeError("Paddle không có CUDA")
    paddle.set_device("gpu:0")
    paddle.matmul(paddle.ones([32, 32]), paddle.ones([32, 32])).numpy()
    source = checkout("paddle")
    for task in ("det", "rec"):
        path, config = config_for(name, task, source)
        run = directory(name) / task
        complete = run / "completed.json"
        if complete.is_file():
            if read_json(complete)["config_sha256"] != digest(path):
                raise ValueError("Config khác lượt đã train")
            continue
        cmd = [sys.executable, "-u", source / "tools/train.py", "-c", path]
        if (run / "latest.pdparams").is_file():
            cmd += ["-o", f"Global.checkpoints={run / 'latest'}"]
        execute(cmd, source)
        if not (run / "best_accuracy.pdparams").is_file():
            raise FileNotFoundError("Train chưa tạo best trên val")
        save_json(
            complete,
            {
                "config_sha256": digest(path),
                "selection_split": "val",
                "checkpoint": "best_accuracy",
            },
        )


def export(name):
    import numpy as np
    import onnxruntime as ort
    import paddle.inference as inference
    from ocr.data import image
    from ocr.paddle_suite_runtime import Preprocessor, PaddlePipeline

    source = checkout("paddle")
    base = directory(name)
    (base / "onnx").mkdir(exist_ok=True)
    for task in ("det", "rec"):
        checkpoint = base / task / "best_accuracy"
        if not checkpoint.with_suffix(".pdparams").is_file():
            raise FileNotFoundError(checkpoint)
        target = base / "source" / task
        execute(
            [
                sys.executable,
                source / "tools/export_model.py",
                "-c",
                base / f"{task}.yml",
                "-o",
                f"Global.pretrained_model={checkpoint}",
                f"Global.save_inference_dir={target}",
                "Global.use_gpu=False",
            ],
            source,
        )
        graphs = list(target.glob("*.json")) + list(target.glob("*.pdmodel"))
        if len(graphs) != 1:
            raise ValueError("Không xác định được graph Paddle đã xuất")
        params = target / "inference.pdiparams"
        execute(
            [
                sys.executable,
                "-m",
                "paddle2onnx.command",
                "--model_dir",
                target,
                "--model_filename",
                graphs[0].name,
                "--params_filename",
                params.name,
                "--save_file",
                base / "onnx" / f"{task}.onnx",
                "--opset_version",
                "17",
            ],
            source,
        )
    resize = {"limit_side_len": 960, "limit_type": "max", "max_side_limit": 4000}
    pre = Preprocessor(base, resize)
    checks = []
    for task in ("det", "rec"):
        target = base / "source" / task
        graph = next(iter([*target.glob("*.json"), *target.glob("*.pdmodel")]))
        config = inference.Config(str(graph), str(target / "inference.pdiparams"))
        config.disable_gpu()
        config.disable_mkldnn()
        config.set_cpu_math_library_num_threads(4)
        predictor = inference.create_predictor(config)
        engine = ort.InferenceSession(
            str(base / "onnx" / f"{task}.onnx"), providers=["CPUExecutionProvider"]
        )
        for row in rows("val", task)[:3]:
            pixels = image(row["path"])
            data = pre.det_input(pixels)[0] if task == "det" else pre.rec_input(pixels)
            handle = predictor.get_input_handle(predictor.get_input_names()[0])
            handle.reshape(data.shape)
            handle.copy_from_cpu(data)
            predictor.run()
            reference = predictor.get_output_handle(predictor.get_output_names()[0]).copy_to_cpu()
            actual = engine.run(None, {engine.get_inputs()[0].name: data})[0]
            np.testing.assert_allclose(actual, reference, rtol=1e-3, atol=1e-4)
            checks.append(
                {
                    "task": task,
                    "image": row["relative"],
                    "max_abs": float(np.max(np.abs(actual - reference))),
                }
            )
        # YAML suy luận cũng nằm trong bộ checksum portable.
        shutil.copy2(target / "inference.yml", base / "onnx" / f"{task}.yml")
    save_json(base / "onnx/runtime.json", {"family": "paddle", "resize": resize})
    return freeze(
        name,
        {
            "conversion_checks": checks,
            "source_commit": settings()["sources"]["paddle"]["revision"],
            "initialization": "official pretrained DET/REC; original dictionaries",
            "best": "native val hmean/acc",
        },
    )

"""Xuất checkpoint VietLPR tốt nhất sang ONNX FP32 và đối chiếu số học."""

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from .config import load_config, save_json
from .data import image, read_split
from .runtime import crop_line, preprocess_det, resize_rec
from .torch_models import DeployModel, build_model


def export(cfg):
    import onnx
    import onnxruntime as ort

    torch.set_num_threads(cfg["benchmark"]["threads"])
    run = Path(cfg["run_dir"])
    out = run / "onnx"
    out.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for task in ("det", "rec"):
        path = run / f"{task}_best.pth"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (
            checkpoint["cfg"]["name"] != cfg["name"]
            or checkpoint["cfg"]["chars"] != cfg["chars"]
            or checkpoint["cfg"]["model"] != cfg["model"]
        ):
            raise ValueError("Checkpoint khác kiến trúc hoặc bộ ký tự")
        model = build_model(cfg, task)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        m = cfg["model"]
        size = (m["det_size"], m["det_size"]) if task == "det" else (m["rec_h"], m["rec_w"])
        example = torch.rand(1, 3, *size)
        with torch.inference_mode():
            original = model(example)
        if cfg["name"] == "vietlpr_v2":
            from third_party.VietLPR_v2.models import reparameterize_model

            model = reparameterize_model(copy.deepcopy(model)).eval()
            with torch.inference_mode():
                np.testing.assert_allclose(
                    model(example).numpy(), original.numpy(), atol=2e-5, rtol=2e-4
                )
        deploy = DeployModel(model, cfg, task).eval()
        with torch.inference_mode():
            expected = (original.sigmoid() if task == "det" else original.softmax(-1)).numpy()
            np.testing.assert_allclose(deploy(example).numpy(), expected, atol=2e-5, rtol=2e-4)
        destination = out / f"{task}.onnx"
        torch.onnx.export(
            deploy,
            example,
            str(destination),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamo=False,
            do_constant_folding=True,
        )
        onnx.checker.check_model(str(destination))
        options = ort.SessionOptions()
        options.intra_op_num_threads = cfg["benchmark"]["threads"]
        session = ort.InferenceSession(
            str(destination), options, providers=["CPUExecutionProvider"]
        )
        actual = session.run(None, {session.get_inputs()[0].name: example.numpy()})[0]
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
        for row in read_split(cfg["dataset"], "val", task)[:3]:
            pixels = image(row["path"])
            if task == "det":
                data = preprocess_det(pixels, cfg)
            elif cfg["name"] == "vietlpr_v2":
                height, width = pixels.shape[:2]
                quad = np.float32(
                    [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
                )
                data = crop_line(pixels, quad, cfg)
            else:
                data = resize_rec(pixels, cfg)
            with torch.inference_mode():
                native = deploy(torch.from_numpy(data)).numpy()
            converted = session.run(None, {session.get_inputs()[0].name: data})[0]
            np.testing.assert_allclose(converted, native, atol=2e-5, rtol=2e-4)
        provenance[task] = {
            "epoch": checkpoint["epoch"],
            "signatures": checkpoint["signatures"],
            "smoke_only": checkpoint.get("smoke_only", False),
            "checked_on_val": True,
            "max_absolute_error": float(np.max(np.abs(actual - expected))),
        }
    save_json(out / "config.json", cfg)
    save_json(out / "export_check.json", provenance)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("vietlpr_v1", "vietlpr_v2"), required=True)
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    cfg = load_config(args.model)
    if args.run_dir:
        cfg["run_dir"] = str(Path(args.run_dir).resolve())
    print(export(cfg))


if __name__ == "__main__":
    main()

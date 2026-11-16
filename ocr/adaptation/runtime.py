"""Chạy bảy OCR FP32, không dùng GT khi suy luận."""

import shutil

import numpy as np

from ocr.metrics import normalize
from ocr.public_vn_runtime import session, model_input, decode_lprnet
from .common import directory, validate_frozen, read_json


def pipeline(name, threads=4):
    validate_frozen(name)
    base = directory(name)
    runtime = read_json(base / "onnx/runtime.json")
    family = runtime["family"]
    if family == "vietlpr":
        from third_party.VietLPR_v2.inference import LPR

        native = LPR(
            str(base / "onnx/det.onnx"),
            str(base / "onnx/rec.onnx"),
            str(base / "onnx/lpr_config.json"),
            threads=threads,
        )

        def infer(pixels):
            result = native(pixels)
            return {
                "text": normalize(result["plate"]),
                "lines": [t for t, c in result["lines"]],
                "boxes": [q.tolist() for q in result["boxes"]],
                "confidence": result["conf"],
            }

        return infer
    if family == "paddle":
        from ocr.paddle_suite_runtime import PaddlePipeline

        for task in ("det", "rec"):
            dest = base / "source" / task / "inference.yml"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(base / "onnx" / f"{task}.yml", dest)
        return PaddlePipeline(base, runtime["resize"], threads)
    engine = session(base / "onnx/model.onnx", threads)
    spec = engine.get_inputs()[0]

    def infer(pixels):
        if family == "cct":
            from .preprocessing import cct_input

            data = cct_input(pixels)
        else:
            data = model_input(pixels, runtime)
        if data is None:
            return {"text": ""}
        data = data.astype(np.float32 if spec.type == "tensor(float)" else np.uint8, copy=False)
        output = engine.run(None, {spec.name: data})[0]
        if not np.isfinite(output).all():
            raise ValueError("Output NaN/Inf")
        if family == "cct":
            text = "".join(
                runtime["chars"][int(i)]
                for i in output.reshape(9, len(runtime["chars"])).argmax(-1)
            ).rstrip("_")
        else:
            text = decode_lprnet(output, runtime["chars"])
        return {"text": normalize(text)}

    return infer

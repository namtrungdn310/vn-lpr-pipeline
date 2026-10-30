"""Chuyển model v4 bằng môi trường Paddle cũ và đối chiếu số học trên val."""

import argparse
import subprocess
import sys

import numpy as np

from .config import save_json
from .data import image, read_split
from .paddle_suite_runtime import Preprocessor, session


def convert(directory, task, catalog):
    import paddle.inference as native

    model = directory / "source" / task / "inference.pdmodel"
    params = model.with_suffix(".pdiparams")
    output = directory / "onnx" / f"{task}.onnx"
    output.parent.mkdir(exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "paddle2onnx.command", "--model_dir", str(model.parent),
         "--model_filename", model.name, "--params_filename", params.name,
         "--save_file", str(output), "--opset_version", "17", "--enable_onnx_checker", "True"],
        check=True,
    )
    pre = Preprocessor(directory, catalog["det_resize"])
    config = native.Config(str(model), str(params))
    config.disable_gpu()
    config.set_cpu_math_library_num_threads(catalog["benchmark"]["threads"])
    config.disable_glog_info()
    predictor = native.create_predictor(config)
    onnx = session(output, catalog["benchmark"]["threads"])
    errors = []
    rows = read_split(catalog["dataset"], "val", task)[:3]
    for row in rows:
        pixels = image(row["path"])
        inputs = pre.det_input(pixels)[0] if task == "det" else pre.rec_input(pixels)
        handle = predictor.get_input_handle(predictor.get_input_names()[0])
        handle.reshape(inputs.shape)
        handle.copy_from_cpu(inputs)
        predictor.run()
        expected = predictor.get_output_handle(predictor.get_output_names()[0]).copy_to_cpu()
        actual = onnx.run(None, {onnx.get_inputs()[0].name: inputs})[0]
        np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-3)
        if task == "rec":
            if actual.shape[-1] != len(pre.decoder.character):
                raise ValueError("Dictionary v4 khác trọng số")
            if pre.decoder([actual])[0] != pre.decoder([expected])[0]:
                raise ValueError("Chuỗi REC khác sau chuyển đổi")
        errors.append(float(np.max(np.abs(actual - expected))))
    from .paddle_suite import digest

    result = {"verification": "local_paddle_vs_onnx", "checked_on_val": True,
              "samples": len(rows), "max_absolute_errors": errors, "sha256": digest(output)}
    save_json(directory / "onnx" / f"{task}_conversion.json", result)


def main():
    from .paddle_suite import load_catalog, run_dir

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", choices=("det", "rec"), required=True)
    args = parser.parse_args()
    catalog = load_catalog()
    convert(run_dir(catalog, args.model), args.task, catalog)


if __name__ == "__main__":
    main()

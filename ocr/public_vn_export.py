"""Xuất checkpoint OCR Việt Nam bằng lớp nguồn đã kiểm; không train."""

import ast
import copy
import __future__
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn


def definitions(path, namespace, names):
    """Chỉ nạp định nghĩa đã chọn; bỏ import và lệnh chạy ngoài hàm."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError(f"Nguồn thiếu định nghĩa: {path}")
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)


def graph_class(path, source_class, method, namespace):
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == source_class)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return type("GraphModel", (nn.Module,), {method: namespace[method], "forward": namespace[method]})


def native_model(variant, directory, source):
    family = variant["family"]
    namespace = {"torch": torch, "nn": nn, "warnings": warnings, "TORCH_1_11": True,
                 "__name__": "public_vn_native"}
    if family == "lprnet":
        definitions(source / "lprnet/LPRNet.py", namespace, {"small_basic_block", "LPRNet", "build_lprnet"})
        model = namespace["build_lprnet"](lpr_max_len=9, phase=False, class_num=len(variant["chars"]), dropout_rate=0)
        model.load_state_dict(torch.load(directory / "source" / variant["weights"]["filename"],
                                        map_location="cpu", weights_only=True), strict=True)
        return model.float().eval()
    aliases = []
    if family == "yolov5":
        definitions(source / "yolo5/common.py", namespace,
                    {"autopad", "Conv", "Bottleneck", "C3", "SPPF", "Concat"})
        # Torch hiện tại luôn mới hơn 1.10; chỉ dùng cho nhánh meshgrid nguồn.
        namespace["check_version"] = lambda version, minimum: True
        definitions(source / "yolo5/yolo.py", namespace, {"Detect"})
        root = graph_class(source / "yolo5/yolo.py", "Model", "_forward_once", namespace)
        aliases = [(namespace[n], "models.common." + n) for n in ("Conv", "Bottleneck", "C3", "SPPF", "Concat")]
        aliases += [(namespace["Detect"], "models.yolo.Detect"), (root, "models.yolo.Model")]
    elif family == "yolov8":
        definitions(source / "yolo8/conv.py", namespace, {"autopad", "Conv", "Concat"})
        definitions(source / "yolo8/block.py", namespace, {"Bottleneck", "C2f", "DFL", "SPPF"})
        definitions(source / "yolo8/tal.py", namespace, {"make_anchors", "dist2bbox"})
        definitions(source / "yolo8/head.py", namespace, {"Detect"})
        root = graph_class(source / "yolo8/tasks.py", "BaseModel", "_predict_once", namespace)
        aliases = [(namespace[n], "ultralytics.nn.modules.conv." + n) for n in ("Conv", "Concat")]
        aliases += [(namespace[n], "ultralytics.nn.modules.block." + n) for n in ("Bottleneck", "C2f", "DFL", "SPPF")]
        aliases += [(namespace["Detect"], "ultralytics.nn.modules.head.Detect"),
                    (root, "ultralytics.nn.tasks.DetectionModel")]
    else:
        raise ValueError("Model này không cần chuyển từ Torch")
    allowed = aliases + [nn.SiLU, nn.Conv2d, nn.Upsample, nn.BatchNorm2d, nn.MaxPool2d, nn.Sequential, nn.ModuleList]
    checkpoint = directory / "source" / variant["weights"]["filename"]
    allowed_names = {entry[1] if isinstance(entry, tuple) else entry.__module__ + "." + entry.__qualname__ for entry in allowed}
    unexpected = set(torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint)) - allowed_names
    if unexpected:
        raise ValueError(f"Checkpoint có kiểu ngoài danh sách đã kiểm: {unexpected}")
    with torch.serialization.safe_globals(allowed):
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = data["ema"] if data.get("ema") is not None else data["model"]
    model = model.float().eval()
    names = model.names
    actual = "".join(names[i] for i in range(len(names))) if isinstance(names, dict) else "".join(names)
    if actual != variant["chars"]:
        raise ValueError("Bộ ký tự không khớp checkpoint")
    head = model.model[-1]
    head.stride = head.stride.float()
    if family == "yolov5":
        head.onnx_dynamic = True
    else:
        head.dynamic = True
        head.export = True
        head.format = "onnx"
    return model


class Output(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        result = self.model(x)
        return result[0] if isinstance(result, tuple) else result


class ExplicitPool(nn.Module):
    """Ghi rõ trục batch ngầm của MaxPool3d khi nguồn dùng tensor 4D."""
    def __init__(self, pool):
        super().__init__()
        self.pool = pool

    def forward(self, x):
        return self.pool(x.unsqueeze(0)).squeeze(0)


def explicit_pools(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MaxPool3d):
            setattr(module, name, ExplicitPool(child))
        else:
            explicit_pools(child)


def export(variant, directory, source, samples):
    """Đối chiếu Torch/ONNX trên val, gồm ảnh có kích thước khác nhau."""
    import onnx
    import onnxruntime as ort

    torch.set_num_threads(4)
    model = Output(native_model(variant, directory, source)).eval()
    if variant["family"] == "lprnet":
        reference = copy.deepcopy(model)
        explicit_pools(model)
        with torch.inference_mode():
            for pixels in samples:
                torch.testing.assert_close(model(torch.from_numpy(pixels)), reference(torch.from_numpy(pixels)),
                                           rtol=0, atol=0)
    destination = directory / "onnx/model.onnx"
    destination.parent.mkdir(parents=True, exist_ok=True)
    axes = None if variant["family"] == "lprnet" else {
        "input": {2: "height", 3: "width"},
        "output": {1 if variant["family"] == "yolov5" else 2: "candidates"},
    }
    with torch.inference_mode():
        torch.onnx.export(model, torch.from_numpy(samples[0]), destination,
                          input_names=["input"], output_names=["output"], opset_version=17,
                          dynamic_axes=axes, do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(str(destination))
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
    engine = ort.InferenceSession(str(destination), options, providers=["CPUExecutionProvider"])
    errors = []
    with torch.inference_mode():
        for pixels in samples:
            expected = model(torch.from_numpy(pixels)).numpy()
            actual = engine.run(None, {"input": pixels})[0]
            np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)
            errors.append(float(np.max(np.abs(actual - expected))))
    return {"checked_on": "val", "max_absolute_errors": errors, "weights_only": True,
            "native_definitions": "upstream AST; graph forward method and checkpoint classes preserved"}

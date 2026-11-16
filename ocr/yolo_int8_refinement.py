"""Chỉ mở rộng thử nghiệm YOLO khi vòng đầu chưa đạt chất lượng val."""

import hashlib
from pathlib import Path
import re
import shutil

import onnx

from . import int8_optimization as parent
from .config import save_json
from .paddle_suite_state import model_lock
from plate_detection import quantization as yq
from plate_detection.data import load_config, read_image

VARIANTS = (
    {"name": "head_fp32_more_train", "method": "minmax", "exclude": "head", "calibration_images": 512},
    {"name": "stem_head_fp32", "method": "minmax", "exclude": "stem_head"},
    {"name": "backbone_only", "method": "minmax", "exclude": "neck_head"},
)
SILU_VARIANTS = (
    {"name": "conv_silu_head_fp32", "method": "minmax", "exclude": "head", "ops": ["Conv", "Sigmoid", "Mul"]},
    {"name": "conv_silu_backbone_only", "method": "minmax", "exclude": "neck_head", "ops": ["Conv", "Sigmoid", "Mul"]},
)


def signature():
    return hashlib.sha256(Path(__file__).read_bytes() + parent.signature().encode()).hexdigest()


def quantize(source, target, rows, spec):
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
    model = onnx.load(source)
    key = model.graph.input[0].name
    excluded = []
    ops = spec.get("ops", ["Conv"])
    for node in model.graph.node:
        if node.op_type not in ops:
            continue
        index = re.search(r"/model\.(\d+)/", node.name)
        if not index:
            excluded.append(node.name)
            continue
        index = int(index.group(1))
        if index == 23 or (spec["exclude"] == "stem_head" and index == 0) or (spec["exclude"] == "neck_head" and index > 10):
            excluded.append(node.name)
    class Reader(CalibrationDataReader):
        def __init__(self):
            self.items = iter(rows)
        def get_next(self):
            row = next(self.items, None)
            return None if row is None else {key: yq.letterbox(read_image(row["image"]), 640)[0]}
    quantize_static(str(source), str(target), Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    per_channel=True, op_types_to_quantize=ops, nodes_to_exclude=excluded,
                    calibrate_method=CalibrationMethod.Percentile if spec["method"] == "percentile" else CalibrationMethod.MinMax,
                    extra_options={"ActivationSymmetric": False, "WeightSymmetric": True,
                                   "CalibTensorRangeSymmetric": False, "CalibPercentile": 99.99,
                                   "DedicatedQDQPair": False})
    onnx.checker.check_model(str(target))


def prepare():
    cfg = parent.settings()
    base = parent.validate_selected("yolo")
    run = Path(cfg["run_dir"]) / "yolo/refinement"
    with model_lock(run):
        if (run / "selected.json").is_file():
            return validate_selected()
        baseline = base["baseline_val"]
        results = parent.read_json(run.parent / "trials.json")
        rows = parent.read_json(run.parent / "calibration.json")["rows"]
        candidates = (*SILU_VARIANTS, *(VARIANTS if not base["quality_gate_passed"] else ()))
        if candidates:
            for spec in candidates:
                folder = run / "candidates" / spec["name"]
                folder.mkdir(parents=True, exist_ok=True)
                print("YOLO REFINE", spec, flush=True)
                if not (folder / "assessment.json").is_file():
                    selected_rows = (yq.calibration_rows({**cfg, "calibration_train_images": spec["calibration_images"]}, load_config())
                                     if spec.get("calibration_images") else rows)
                    save_json(folder / "calibration.json", {"split": "train", "rows": selected_rows})
                    quantize(run.parent / "model_preprocessed.onnx", folder / "model.onnx", selected_rows, spec)
                    parent.execute("assess", "--task", "yolo", "--folder", folder)
                result = parent.read_json(folder / "assessment.json")
                if result["signature"] != parent.signature():
                    raise ValueError("Assessment khác code tối ưu")
                result["spec"] = {**spec, "activation": "u8", "weight": "s8", "mixed": True}
                result["eligible"] = parent.eligible("yolo", result, baseline, cfg)
                results.append(result)
        selected, passed = parent.choose("yolo", results, baseline, cfg)
        target = run / "selected"
        target.mkdir(exist_ok=True)
        for filename in selected["artifacts"]:
            shutil.copy2(Path(selected["folder"]) / filename, target / filename)
        state = {"protocol": {"signature": signature(), "source_protocol": base["protocol"],
                              "calibration_split": "train", "selection_split": "val"},
                 "artifacts": selected["artifacts"], "folder": str(target), "scope": selected["spec"],
                 "selected_val": selected, "baseline_val": baseline, "quality_gate_passed": passed,
                 "training_performed": False}
        save_json(run / "trials.json", results)
        save_json(run / "selected.json", state)
        print("YOLO REFINED SELECTED", selected["spec"], "quality_gate", passed, flush=True)
        return state


def validate_selected():
    parent.validate_selected("yolo")
    state = parent.read_json(Path(parent.settings()["run_dir"]) / "yolo/refinement/selected.json")
    if state["protocol"]["signature"] != signature():
        raise ValueError("Code refinement YOLO đã đổi")
    for filename, expected in state["artifacts"].items():
        if parent.sha256(Path(state["folder"]) / filename) != expected:
            raise ValueError("Model refinement YOLO đã đổi")
    return state


if __name__ == "__main__":
    prepare()

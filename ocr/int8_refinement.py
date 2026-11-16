"""Thu hẹp INT8 ở REC; giữ DET FP32 để tránh chi phí không có lợi."""

import hashlib
import logging
from pathlib import Path
import shutil

import onnx

from . import int8_optimization as parent
from . import quantization as old
from .config import ROOT, save_json
from .paddle_suite_state import model_lock

CONFIG = ROOT / "configs/ocr/int8_refinement.json"


def signature():
    return hashlib.sha256(Path(__file__).read_bytes() + CONFIG.read_bytes() + parent.signature().encode()).hexdigest()


def quantize(source, output, tensors, spec):
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_dynamic, quantize_static

    if spec["method"] == "dynamic":
        quantize_dynamic(str(source), str(output), op_types_to_quantize=["MatMul"],
                         per_channel=True, weight_type=QuantType.QInt8,
                         extra_options={"MatMulConstBOnly": True, "WeightSymmetric": True})
        return
    model = onnx.load(source)
    weights = {w.name: list(w.dims) for w in model.graph.initializer}
    # ORT 1.20.1 QDQMatMul không dùng MatMulConstBOnly; phải loại attention động rõ ràng.
    excluded = [n.name for n in model.graph.node if n.op_type == "MatMul" and
                (n.input[1] not in weights or (spec.get("exclude_classifier") and weights[n.input[1]][-1] == 33))]
    key = model.graph.input[0].name
    class Reader(CalibrationDataReader):
        def __init__(self):
            self.items = (x[i:i + 1] for x in tensors for i in range(x.shape[0]))
        def get_next(self):
            x = next(self.items, None)
            return {key: x} if x is not None else None
    quantize_static(str(source), str(output), Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    op_types_to_quantize=spec["ops"], per_channel=True, nodes_to_exclude=excluded,
                    calibrate_method=CalibrationMethod.MinMax,
                    extra_options={"ActivationSymmetric": False, "WeightSymmetric": True,
                                   "CalibTensorRangeSymmetric": False, "MatMulConstBOnly": True,
                                   "DedicatedQDQPair": False})


def prepare():
    logging.getLogger().setLevel(logging.WARNING)
    cfg = parent.settings()
    base = parent.validate_selected("vietlpr")
    run = Path(cfg["run_dir"]) / "vietlpr/refinement"
    with model_lock(run):
        if (run / "selected.json").is_file():
            return validate_selected()
        source, original, _ = old.source_model(old.settings())
        samples, recipe = old.calibration(old.settings(), source)
        save_json(run / "calibration.json", recipe)
        baseline = base["baseline_val"]
        results = parent.read_json(run.parent / "trials.json")
        for spec in parent.read_json(CONFIG)["variants"]:
            folder = run / "candidates" / spec["name"]
            folder.mkdir(parents=True, exist_ok=True)
            print("REFINE", spec["name"], flush=True)
            if not (folder / "assessment.json").is_file():
                shutil.copy2(original / "det.onnx", folder / "det.onnx")
                quantize(run.parent / "rec_preprocessed.onnx", folder / "rec.onnx", samples["rec"], spec)
                onnx.checker.check_model(str(folder / "rec.onnx"))
                parent.execute("assess", "--task", "vietlpr", "--folder", folder)
            result = parent.read_json(folder / "assessment.json")
            if result["signature"] != parent.signature():
                raise ValueError("Val khác code tối ưu")
            result["spec"] = {**spec, "quantized_part": "rec", "det_precision": "FP32"}
            result["eligible"] = parent.eligible("vietlpr", result, baseline, cfg)
            results.append(result)
        chosen, passed = parent.choose("vietlpr", results, baseline, cfg)
        target = run / "selected"
        target.mkdir(exist_ok=True)
        for filename in chosen["artifacts"]:
            shutil.copy2(Path(chosen["folder"]) / filename, target / filename)
        state = {"protocol": {"signature": signature(), "parent_signature": parent.signature(),
                              "source_protocol": base["protocol"], "calibration_split": "train", "selection_split": "val"},
                 "artifacts": chosen["artifacts"], "folder": str(target), "scope": chosen["spec"],
                 "selected_val": chosen, "baseline_val": baseline, "quality_gate_passed": passed,
                 "training_performed": False}
        save_json(run / "trials.json", results)
        save_json(run / "selected.json", state)
        print("REFINED SELECTED", chosen["spec"], "quality_gate", passed, flush=True)
        return state


def validate_selected():
    parent.validate_selected("vietlpr")
    run = Path(parent.settings()["run_dir"]) / "vietlpr/refinement"
    state = parent.read_json(run / "selected.json")
    if state["protocol"]["signature"] != signature():
        raise ValueError("Code/config refinement đã thay đổi")
    for filename, expected in state["artifacts"].items():
        if parent.sha256(Path(state["folder"]) / filename) != expected:
            raise ValueError("Model refinement đã đổi")
    return state


if __name__ == "__main__":
    prepare()

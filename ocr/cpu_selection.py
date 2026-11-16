"""Chọn lại tốc độ val trên cùng lõi CPU; không lượng tử hoá hay train lại."""

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

from . import cpu_protocol as cpu
from . import int8_optimization as parent
from . import int8_refinement as v2
from . import yolo_int8_refinement as yolo
from .config import ROOT, save_json
from .paddle_suite_state import model_lock


def signature():
    return hashlib.sha256(Path(__file__).read_bytes() + cpu.signature().encode()
                          + v2.signature().encode() + yolo.signature().encode()).hexdigest()


def prepare(task):
    context = cpu.apply()
    cfg = parent.settings()
    source = v2.validate_selected() if task == "vietlpr" else yolo.validate_selected()
    run = Path(cfg["run_dir"]) / "fixed_cpu" / task
    with model_lock(run):
        if (run / "selected.json").is_file():
            return validate_selected(task)
        trials = parent.read_json(Path(source["folder"]).parent / "trials.json")
        baseline = source["baseline_val"]
        candidates = [baseline, *(r for r in trials if parent.eligible(task, r, baseline, cfg))]
        unique, results = set(), []
        for index, candidate in enumerate(candidates):
            key = tuple(sorted(candidate["artifacts"].items()))
            if key in unique:
                continue
            unique.add(key)
            folder = run / ("fp32" if index == 0 else f"candidate_{index:02d}")
            folder.mkdir(parents=True, exist_ok=True)
            for filename, expected in candidate["artifacts"].items():
                src = Path(candidate["folder"]) / filename
                if parent.sha256(src) != expected:
                    raise ValueError("Ứng viên đã đổi trọng số")
                shutil.copy2(src, folder / filename)
            if not (folder / "assessment.json").is_file():
                subprocess.run([sys.executable, "-u", "-B", "-m", "ocr.cpu_selection", "assess",
                                "--task", task, "--folder", str(folder)], cwd=ROOT, check=True)
            result = parent.read_json(folder / "assessment.json")
            if result["cpu_protocol"] != context or result["selection_signature"] != signature():
                raise ValueError("Lượt val khác CPU/code")
            result["spec"] = {"name": "fp32"} if index == 0 else candidate["spec"]
            if index == 0:
                baseline = result
            else:
                result["eligible"] = parent.eligible(task, result, baseline, cfg)
                results.append(result)
        chosen, passed = parent.choose(task, results, baseline, cfg)
        target = run / "selected"
        target.mkdir(exist_ok=True)
        for filename in chosen["artifacts"]:
            shutil.copy2(Path(chosen["folder"]) / filename, target / filename)
        state = {"protocol": {"signature": signature(), "cpu_protocol": context, "selection_split": "val"},
                 "artifacts": chosen["artifacts"], "folder": str(target), "scope": chosen["spec"],
                 "selected_val": chosen, "baseline_val": baseline, "quality_gate_passed": passed,
                 "training_performed": False}
        save_json(run / "trials.json", results)
        save_json(run / "selected.json", state)
        print("FIXED CPU SELECTED", task, chosen["spec"], chosen["scores"], chosen["timing"], flush=True)
        return state


def validate_selected(task):
    state = parent.read_json(Path(parent.settings()["run_dir"]) / "fixed_cpu" / task / "selected.json")
    if state["protocol"]["signature"] != signature():
        raise ValueError("Code chọn model đã đổi")
    for name, expected in state["artifacts"].items():
        if parent.sha256(Path(state["folder"]) / name) != expected:
            raise ValueError("Model đã đổi sau khi chọn")
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "assess"))
    parser.add_argument("--task", choices=("vietlpr", "yolo"), required=True)
    parser.add_argument("--folder", type=Path)
    args = parser.parse_args()
    if args.action == "assess":
        context = cpu.apply()
        parent.assess(args.task, args.folder)
        path = args.folder / "assessment.json"
        result = parent.read_json(path)
        result.update(cpu_protocol=context, selection_signature=signature())
        save_json(path, result)
    else:
        prepare(args.task)

"""Điều phối sáu lượt train; VietLPR đã chốt không được train lại."""

import argparse
import os
import subprocess
from datetime import datetime

from ocr.config import ROOT, save_json
from ocr.paddle_suite_state import model_lock
from .common import settings, entry, directory, validate_frozen


def python_for(family):
    folder = ".venv-train-paddle" if family == "paddle" else ".venv-train-torch"
    path = ROOT / folder / "Scripts/python.exe"
    if not path.is_file():
        raise FileNotFoundError(f"Chạy scripts/setup_pc.ps1 trước: {path}")
    return path


def checked_run(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(subprocess.list2cmdline([str(x) for x in command]) + "\n")
        with subprocess.Popen(
            [str(x) for x in command],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "KERAS_BACKEND": "torch"},
        ) as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    stream.write(line)
                    stream.flush()
                code = process.wait()
            except KeyboardInterrupt:
                if process.poll() is None:
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        check=False,
                        capture_output=True,
                    )
                raise
            if code:
                raise RuntimeError(f"Train/export lỗi {code}; xem {log}")


def run(models=None):
    from .data import prepare

    cfg = settings()
    names = models or [m["name"] for m in cfg["models"] if m["train"]]
    if any(not entry(n)["train"] for n in names):
        raise ValueError("VietLPR_v1 đã chốt, không train lại")
    with model_lock(directory() / "training"):
        validate_frozen("VietLPR_v1")
        prepare()
        for name in names:
            model = entry(name)
            if (directory(name) / "frozen.json").is_file():
                frozen = validate_frozen(name)
                if frozen["training_config"] != cfg:
                    raise ValueError("Cấu hình khác model đã train")
                print(name, "đã train/export xong; bỏ qua.", flush=True)
                continue
            for action in ("train", "export"):
                save_json(
                    directory() / "training_state.json",
                    {"status": "running", "model": name, "action": action},
                )
                log = directory() / "logs" / f"{name}_{action}_{datetime.now():%Y%m%d_%H%M%S}.log"
                try:
                    checked_run(
                        [
                            python_for(model["family"]),
                            "-u",
                            "-B",
                            "-m",
                            "ocr.adaptation.suite",
                            action,
                            "--model",
                            name,
                        ],
                        log,
                    )
                except BaseException as error:
                    save_json(
                        directory() / "training_state.json",
                        {
                            "status": "interrupted",
                            "model": name,
                            "action": action,
                            "error": str(error),
                            "log": str(log),
                        },
                    )
                    raise
            validate_frozen(name)
        save_json(directory() / "training_state.json", {"status": "complete", "models": names})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "train", "export", "prepare"))
    parser.add_argument("--model", nargs="+")
    args = parser.parse_args()
    if args.action == "prepare":
        from .data import prepare

        print(prepare())
        return
    if args.action == "run":
        run(args.model)
        return
    if not args.model or len(args.model) != 1:
        raise ValueError("Chọn đúng một model")
    name = args.model[0]
    family = entry(name)["family"]
    if family == "vietlpr":
        raise ValueError("Không train/export lại VietLPR đã chốt")
    if family == "paddle":
        from . import paddle

        getattr(paddle, args.action)(name)
    else:
        from importlib import import_module

        getattr(import_module(f"ocr.adaptation.{family}"), args.action)()


if __name__ == "__main__":
    main()

"""Chạy CLI từ notebook, hiện log và tiếp tục checkpoint khi chạy lại cell."""

from collections import deque
from datetime import datetime
import os
from pathlib import Path
import subprocess

from .config import ROOT, load_config
from .run_state import prepare_vietlpr_run


def run(python, *arguments):
    """Chuyển stdout/stderr của tiến trình con vào output Jupyter và file log."""
    python = Path(python)
    if not python.is_file():
        raise FileNotFoundError(f"Thiếu môi trường Python: {python}")
    directory = ROOT / "runs/notebook_logs"
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / f"{datetime.now():%Y%m%d_%H%M%S_%f}.log"
    command = [str(python), "-u", *map(str, arguments)]
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    print("Chạy:", subprocess.list2cmdline(command), flush=True)
    print("Log:", log_path, flush=True)
    tail = deque(maxlen=20)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(subprocess.list2cmdline(command) + "\n")
        with subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        ) as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                    tail.append(line)
                code = process.wait()
            except KeyboardInterrupt:
                # Dừng cả CLI và các tiến trình con.
                if process.poll() is None:
                    if os.name == "nt":
                        subprocess.run(
                            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            check=False,
                        )
                    else:
                        process.terminate()
                    process.wait()
                print("\nĐã dừng. Chạy lại cell train để tiếp tục checkpoint đã lưu.", flush=True)
                raise
    if code:
        raise RuntimeError(f"Lệnh lỗi (mã {code}). Log: {log_path}\n" + "".join(tail))
    return log_path


def require_files(paths, instruction):
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(instruction + "\nThiếu:\n" + "\n".join(missing))


def train_vietlpr(python, name, device="cuda"):
    cfg = load_config(name)
    prepare_vietlpr_run(cfg)
    for task in ("det", "rec"):
        arguments = [
            "-m",
            "ocr.train_torch",
            "--model",
            name,
            "--task",
            task,
            "--device",
            device,
        ]
        if any(
            (Path(cfg["run_dir"]) / sub / f"{task}_last.pth").is_file() for sub in (".", "native")
        ):
            arguments.append("--resume")
        run(python, *arguments)

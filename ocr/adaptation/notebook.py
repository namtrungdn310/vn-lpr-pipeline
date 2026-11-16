"""Hiện log tiến trình con trong notebook, không import framework train."""

from datetime import datetime
import subprocess

from ocr.config import ROOT
from .common import directory


def run(python, *arguments):
    path = directory() / "logs" / f"notebook_{datetime.now():%Y%m%d_%H%M%S}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(python), "-u", "-B", *map(str, arguments)]
    print(subprocess.list2cmdline(command))
    print("Log:", path, flush=True)
    with path.open("w", encoding="utf-8") as log, subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    ) as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except KeyboardInterrupt:
            if process.poll() is None:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    check=False,
                )
            raise
        if code:
            raise RuntimeError(f"Lệnh lỗi {code}; xem {path}")

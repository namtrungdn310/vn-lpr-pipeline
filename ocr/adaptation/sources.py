"""Tải đúng mã nguồn và trọng số train; không lưu chúng lên Git."""

import shutil
import subprocess
import urllib.request

from ocr.config import ROOT
from .common import settings, directory, digest


def checkout(kind):
    source = settings()["sources"][kind]
    target = ROOT / source["directory"]
    if not target.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--no-checkout",
                "--filter=blob:none",
                source["repository"],
                str(target),
            ],
            check=True,
        )
    if not (target / ".git").exists():
        raise ValueError("Thư mục nguồn không phải checkout Git")
    actual = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD"], text=True, capture_output=True
    )
    expected_file = target / ("configs" if kind == "paddle" else "fast_plate_ocr")
    if (
        actual.returncode
        or actual.stdout.strip() != source["revision"]
        or not expected_file.exists()
    ):
        subprocess.run(
            ["git", "-C", str(target), "fetch", "--depth", "1", "origin", source["revision"]],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(target), "checkout", "--detach", source["revision"]], check=True
        )
    if subprocess.run(["git", "-C", str(target), "diff", "--quiet", "HEAD", "--", "."]).returncode:
        raise ValueError("Mã nguồn upstream đã sửa; không train/export khi khác commit đã chốt")
    return target


def asset(name):
    entry = settings()["assets"][name]
    target = directory() / "pretrained" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        cached = ROOT / "runs/pc_setup/assets" / name
        if cached.is_file() and digest(cached) == entry["sha256"]:
            shutil.copy2(cached, target)
        else:
            temporary = target.with_suffix(target.suffix + ".partial")
            with urllib.request.urlopen(entry["url"], timeout=90) as response, temporary.open(
                "wb"
            ) as stream:
                shutil.copyfileobj(response, stream)
            if digest(temporary) != entry["sha256"]:
                raise ValueError(f"Sai SHA trọng số tải về: {name}")
            temporary.replace(target)
    if digest(target) != entry["sha256"]:
        raise ValueError(f"Trọng số tải về bị thay đổi: {name}")
    return target

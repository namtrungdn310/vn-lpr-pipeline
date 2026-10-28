"""Đọc nhãn DET/REC và các thông số đã duyệt."""

import json
from dataclasses import dataclass
from pathlib import Path

from .io import checked_path

SPLITS = ("train", "val", "test")
CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def read_label_rows(label: Path):
    for number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
        if "\t" not in line:
            raise ValueError(f"Nhãn thiếu tab: {label}:{number}")
        relative, value = line.split("\t", 1)
        path = checked_path(label.parent, relative)
        if not path.is_relative_to(label.parent.resolve() / "images") or not path.is_file():
            raise ValueError(f"Đường dẫn ảnh sai: {label}:{number}")
        yield relative, value


def validate_text(text: str) -> None:
    if not isinstance(text, str) or not text or set(text) - set(CHARACTERS):
        raise ValueError(f"Nhãn chữ không hợp lệ: {text!r}")


@dataclass
class Dataset:
    root: Path
    det: dict[str, list]
    rec: dict[str, str]

    @classmethod
    def read(cls, root: Path, *, strict_text=True):
        root = Path(root).resolve(strict=True)
        det, rec = {}, {}
        for split in SPLITS:
            for branch, collection in (("det", det), ("rec", rec)):
                label = checked_path(root, f"{split}/{branch}/{branch}_labels.txt")
                for relative, value in read_label_rows(label):
                    name = f"{split}/{branch}/{relative}"
                    if name in collection:
                        raise ValueError(f"Nhãn ảnh bị lặp: {name}")
                    if branch == "rec":
                        if strict_text:
                            validate_text(value)
                        collection[name] = value
                    else:
                        objects = json.loads(value)
                        if not isinstance(objects, list) or not objects:
                            raise ValueError(f"Danh sách DET rỗng hoặc sai kiểu: {name}")
                        for obj in objects:
                            if not isinstance(obj, dict) or set(obj) != {"transcription", "points"}:
                                raise ValueError(f"Schema DET sai: {name}")
                            if strict_text:
                                validate_text(obj["transcription"])
                            elif not isinstance(obj["transcription"], str):
                                raise ValueError(f"Nhãn DET phải là chuỗi: {name}")
                        collection[name] = objects
        return cls(root, det, rec)


def read_samples(workspace: Path) -> list[dict]:
    path = Path(workspace) / "metadata/samples.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

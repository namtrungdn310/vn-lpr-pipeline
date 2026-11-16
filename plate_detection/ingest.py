"""Giải nén ZIP Roboflow vào vùng dữ liệu riêng, không ghi đè dataset OCR."""

import argparse
from pathlib import Path
import stat
import zipfile

from ocr.config import ROOT, save_json
from .data import load_config, sha256


def extract_zip(archive, destination=None):
    archive = Path(archive).resolve()
    target = Path(destination or load_config()["raw_dataset"]).resolve()
    raw = (ROOT / "data/raw").resolve()
    if target == raw or not target.is_relative_to(raw) or target.exists():
        raise ValueError("Chọn thư mục mới bên trong data/raw; không ghi đè nguồn")
    with zipfile.ZipFile(archive) as z:
        if sum(i.file_size for i in z.infolist()) > 30 * 1024 ** 3:
            raise ValueError("ZIP vượt 30 GiB, cần kiểm tra trước")
        targets = set()
        for item in z.infolist():
            name = item.filename.replace("\\", "/")
            path = (target / name).resolve()
            if not path.is_relative_to(target) or ":" in name or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("ZIP chứa đường dẫn hoặc liên kết không an toàn")
            if path in targets:
                raise ValueError("ZIP chứa đường dẫn trùng")
            targets.add(path)
        yaml_files = [i.filename for i in z.infolist() if Path(i.filename).name == "data.yaml"]
        if len(yaml_files) != 1:
            raise ValueError("ZIP cần đúng một data.yaml của Roboflow")
        z.extractall(target)
    dataset = target / Path(yaml_files[0]).parent
    save_json(target / "download_provenance.json", {
        "source": load_config()["source"], "source_version": 1, "archive_sha256": sha256(archive),
        "archive": archive.name, "license": "CC BY 4.0", "dataset": str(dataset)})
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(extract_zip(args.archive))

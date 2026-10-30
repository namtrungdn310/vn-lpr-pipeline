"""Đọc gói pretrained V4 cho bộ bảy bản Paddle."""

from pathlib import Path
import tarfile


def extract_inference(archive, directory):
    """Chỉ lấy hai file inference; không giải nén đường dẫn tuỳ ý từ tar."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as bundle:
        for name in ("inference.pdmodel", "inference.pdiparams"):
            members = [m for m in bundle.getmembers() if m.isfile() and Path(m.name).name == name]
            if len(members) != 1:
                raise ValueError(f"Gói pretrained cần đúng một file {name}")
            with bundle.extractfile(members[0]) as source:
                (directory / name).write_bytes(source.read())
    return directory / "inference.pdmodel"

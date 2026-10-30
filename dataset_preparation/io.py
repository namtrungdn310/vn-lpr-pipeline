"""Đường dẫn, checksum và ghi file an toàn."""

import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(files: dict) -> str:
    content = json.dumps(files, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def checked_path(root: Path, relative: str) -> Path:
    """Chỉ nhận đường dẫn tương đối, không đi qua symlink/junction."""
    if not isinstance(relative, str) or not relative:
        raise ValueError("Đường dẫn rỗng hoặc sai kiểu")
    parts = PurePosixPath(relative).parts
    if (
        relative.startswith("/")
        or PureWindowsPath(relative).drive
        or "\\" in relative
        or ":" in relative
        or ".." in parts
        or any(ord(char) < 32 for char in relative)
    ):
        raise ValueError(f"Đường dẫn không hợp lệ: {relative!r}")
    root = Path(root).absolute()
    if root.is_symlink() or root.is_junction():
        raise ValueError(f"Thư mục gốc là liên kết: {root}")
    path = root
    for part in parts:
        path /= part
        if path.is_symlink() or path.is_junction():
            raise ValueError(f"Không nhận liên kết: {path}")
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Đường dẫn vượt ra ngoài thư mục gốc")
    return path


def file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(
            checked_path(root, path.relative_to(root).as_posix())
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path: Path, value: dict) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    write_bytes(path, content.encode("utf-8"))

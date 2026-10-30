"""Đọc cấu hình, tách đường dẫn và khoá thông số chung."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SOURCE_CHARS = "0123456789ABCDEFGHJKLMNPRSTUVXYZ"
NAMES = ("vietlpr_v1", "vietlpr_v2")


def load_config(name_or_path):
    path = Path(name_or_path)
    if not path.is_file():
        path = ROOT / "configs/ocr" / f"{name_or_path}.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg["name"] not in NAMES or cfg["chars"] != SOURCE_CHARS:
        raise ValueError("Tên hệ thống hoặc bộ ký tự không hợp lệ")
    for key in ("dataset", "run_dir"):
        p = Path(cfg[key])
        cfg[key] = str((ROOT / p).resolve() if not p.is_absolute() else p.resolve())
    cfg["config_path"] = str(path.resolve())
    return cfg


def same_training_config(first, second):
    """Đổi vị trí file cấu hình không làm thay đổi một lượt train."""
    if not isinstance(first, dict) or not isinstance(second, dict):
        return False
    return ({k: v for k, v in first.items() if k != "config_path"}
            == {k: v for k, v in second.items() if k != "config_path"})


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

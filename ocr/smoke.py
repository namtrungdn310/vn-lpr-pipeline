"""Chạy một epoch ít mẫu để kiểm đường đi; không tạo kết quả benchmark chính."""

import argparse
from pathlib import Path
import tempfile
import torch

from .config import load_config
from .data import image, read_split
from .export import export
from .runtime import onnx_pipeline
from .train_torch import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("vietlpr_v1", "vietlpr_v2", "all"), default="all")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--full-size", action="store_true", help="Giữ input DET theo cấu hình train"
    )
    args = parser.parse_args()
    names = ("vietlpr_v1", "vietlpr_v2") if args.model == "all" else (args.model,)
    for name in names:
        with tempfile.TemporaryDirectory(prefix=f"{name}-smoke-") as directory:
            cfg = load_config(name)
            cfg["run_dir"] = str(Path(directory))
            # Giảm kích thước DET riêng lượt kiểm code; cấu hình train chính giữ nguyên.
            if not args.full_size:
                cfg["model"]["det_size"] = 64
            cfg["train"]["det_batch"] = cfg["train"]["rec_batch"] = 2
            cfg["benchmark"]["threads"] = 2
            for task in ("det", "rec"):
                train(cfg, task, device=args.device, epochs=1, limit=4)
                train(cfg, task, device=args.device, epochs=1, limit=4, resume=True)
            if name == "vietlpr_v2":
                for task in ("det", "rec"):
                    native = torch.load(
                        Path(directory) / "native" / f"{task}_best.pth",
                        map_location="cpu",
                        weights_only=False,
                    )
                    exported = torch.load(
                        Path(directory) / f"{task}_best.pth", map_location="cpu", weights_only=False
                    )
                    assert exported["weights_kind"] == "EMA"
                    assert all(
                        torch.equal(v, exported["model"][k]) for k, v in native["model"].items()
                    )
            directory = export(cfg)
            pipeline = onnx_pipeline(cfg, directory)
            sample = read_split(cfg["dataset"], "val", "det")[0]
            result = pipeline(image(sample["path"]))
            assert isinstance(result["text"], str)
            print(name, args.device, "train/val/export/ONNX pipeline OK", flush=True)


if __name__ == "__main__":
    main()

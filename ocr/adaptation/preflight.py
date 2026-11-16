"""Kiểm GPU bằng phép tính thật; không coi nhận tên GPU là đủ."""

import argparse
import json
import subprocess
import sys

from .common import directory
from ocr.config import save_json


def check_torch():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Torch không nhận CUDA")
    device = torch.cuda.get_device_properties(0)
    if "5070" not in device.name:
        raise RuntimeError("Chỉ chạy train dài trên PC RTX 5070")
    x = torch.randn(2, 3, 64, 128, device="cuda", requires_grad=True)
    net = torch.nn.Conv2d(3, 16, 3).cuda()
    net(x).square().mean().backward()
    torch.cuda.synchronize()
    return {
        "framework": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": device.name,
        "capability": torch.cuda.get_device_capability(0),
        "backward": True,
    }


def check_paddle():
    import paddle
    import paddle2onnx

    if not paddle.is_compiled_with_cuda():
        raise RuntimeError("Paddle không có CUDA")
    paddle.set_device("gpu:0")
    x = paddle.randn([2, 3, 64, 128])
    x.stop_gradient = False
    model = paddle.nn.Conv2D(3, 16, 3)
    model(x).square().mean().backward()
    paddle.device.cuda.synchronize()
    return {
        "framework": paddle.__version__,
        "cuda": paddle.version.cuda(),
        "paddle2onnx": paddle2onnx.__version__,
        "backward": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("torch", "paddle", "sources"))
    args = parser.parse_args()
    if args.action == "sources":
        from .sources import checkout

        source = checkout("cct")
        checkout("paddle")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(source)], check=True
        )
        return
    result = check_torch() if args.action == "torch" else check_paddle()
    save_json(directory() / f"preflight_{args.action}.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

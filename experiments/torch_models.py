"""Nạp đúng hai kiến trúc VietLPR; dùng chung bộ ký tự của v2."""

import importlib
import sys

import torch
from torch import nn

from .config import ROOT


def build_model(cfg, task):
    p = cfg["model"]
    if cfg["name"] == "vietlpr_v1":
        source = ROOT / "third_party/VietLPR_v1"
        if not (source / "vietlpr/models.py").is_file():
            raise FileNotFoundError("Chạy python -m experiments.bootstrap trước")
        sys.path.insert(0, str(source))
        models = importlib.import_module("vietlpr.models")
        if task == "det":
            return models.TinyDBNet(alpha=p["det_alpha"])
        return models.TinyCRNN(
            len(cfg["chars"]) + 1, alpha=p["rec_alpha"], hidden_size=p["rec_hidden"]
        )
    from third_party.VietLPR_v2.models import FastDBNet, FastCRNN

    if task == "det":
        return FastDBNet(width=p["det_width"], fpn_ch=p["det_fpn"])
    return FastCRNN(
        len(cfg["chars"]) + 1,
        img_h=p["rec_h"],
        img_w=p["rec_w"],
        width=p["rec_width"],
        neck=p["rec_neck"],
        dim=p["rec_dim"],
    )


class DeployModel(nn.Module):
    """Gắn sigmoid/softmax vào ONNX; pooling v1 đổi sang phép mean tương đương."""

    def __init__(self, model, cfg, task):
        super().__init__()
        self.model, self.name, self.task = model, cfg["name"], task

    def forward(self, x):
        if self.name == "vietlpr_v1" and self.task == "rec":
            x = self.model.features(x).mean(2).permute(0, 2, 1)
            x = self.model.pre_rnn(x)
            x, _ = self.model.rnn(x)
            output = self.model.fc(x)
        else:
            output = self.model(x)
        return torch.sigmoid(output) if self.task == "det" else output.softmax(-1)

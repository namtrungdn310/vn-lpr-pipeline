"""Kiến trúc notebook đã chốt; công bố với tên VietLPR_v1."""

import torch
from torch import nn


def build_model(cfg, task):
    p = cfg["model"]
    if cfg["name"] != "vietlpr_v2":
        raise ValueError("Kiến trúc VietLPR cũ không còn được train/export")
    from third_party.VietLPR_v2.models import FastDBNet, FastCRNN

    if task == "det":
        return FastDBNet(width=p["det_width"], fpn_ch=p["det_fpn"])
    return FastCRNN(
        len(cfg["chars"]) + 1,
        img_h=p["rec_h"],
        img_w=p["rec_w"],
        width=p["rec_width"],
        neck=p["rec_neck"],
        neck_layers=p.get("rec_neck_layers", 1),
        dim=p["rec_dim"],
    )


class DeployModel(nn.Module):
    """Gắn sigmoid/softmax vào graph FP32 theo nguồn notebook."""

    def __init__(self, model, cfg, task):
        super().__init__()
        self.model, self.name, self.task = model, cfg["name"], task

    def forward(self, x):
        output = self.model(x)
        return torch.sigmoid(output) if self.task == "det" else output.softmax(-1)

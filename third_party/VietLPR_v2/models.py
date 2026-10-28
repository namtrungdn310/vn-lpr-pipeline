"""Kiến trúc tách từ VietLPR_v2.ipynb; không chạy cell train hoặc video."""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RepConv(nn.Module):
    """Nhiều nhánh khi train; gộp thành một convolution tương đương khi xuất model."""

    def __init__(self, cin, cout, k=3, stride=1, groups=1, act=True, use_1x1=True):
        super().__init__()
        self.cin, self.cout, self.k, self.stride, self.groups = (cin, cout, k, stride, groups)
        self.pad = k // 2
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
        self.deployed = False
        self.conv_k = nn.Sequential(
            nn.Conv2d(cin, cout, k, stride, self.pad, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
        )
        self.conv_1 = (
            nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, 0, groups=groups, bias=False), nn.BatchNorm2d(cout)
            )
            if use_1x1 and k > 1
            else None
        )
        s = stride if isinstance(stride, int) else max(stride)
        self.bn_id = nn.BatchNorm2d(cout) if cin == cout and s == 1 else None

    def forward(self, x):
        if self.deployed:
            return self.act(self.fused(x))
        y = self.conv_k(x)
        if self.conv_1 is not None:
            y = y + self.conv_1(x)
        if self.bn_id is not None:
            y = y + self.bn_id(x)
        return self.act(y)

    @staticmethod
    def _fuse_bn(kernel, bn):
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return (kernel * t, bn.bias - bn.running_mean * bn.weight / std)

    def _equivalent_kernel_bias(self):
        k, b = self._fuse_bn(self.conv_k[0].weight, self.conv_k[1])
        if self.conv_1 is not None:
            k1, b1 = self._fuse_bn(self.conv_1[0].weight, self.conv_1[1])
            k = k + F.pad(k1, [self.pad] * 4)
            b = b + b1
        if self.bn_id is not None:
            in_per_g = self.cin // self.groups
            idk = torch.zeros(self.cout, in_per_g, self.k, self.k, device=k.device, dtype=k.dtype)
            for i in range(self.cout):
                idk[i, i % in_per_g, self.pad, self.pad] = 1.0
            ki, bi = self._fuse_bn(idk, self.bn_id)
            k = k + ki
            b = b + bi
        return (k, b)

    @torch.no_grad()
    def reparameterize(self):
        if self.deployed:
            return
        k, b = self._equivalent_kernel_bias()
        self.fused = nn.Conv2d(
            self.cin, self.cout, self.k, self.stride, self.pad, groups=self.groups, bias=True
        ).to(k.device)
        self.fused.weight.copy_(k)
        self.fused.bias.copy_(b)
        for n in ["conv_k", "conv_1", "bn_id"]:
            if hasattr(self, n):
                delattr(self, n)
        self.deployed = True


class MOBlock(nn.Module):
    """Khối MobileOne gồm convolution theo kênh rồi convolution 1×1."""

    def __init__(self, cin, cout, stride=1, k=3):
        super().__init__()
        self.dw = RepConv(cin, cin, k, stride, groups=cin)
        self.pw = RepConv(cin, cout, 1, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class ConvBNAct(nn.Sequential):

    def __init__(self, cin, cout, k=3, s=1, groups=1, act=True):
        super().__init__(
            nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True) if act else nn.Identity(),
        )


def reparameterize_model(model):
    """Gộp nhánh RepConv → 1 conv. Gọi trước khi benchmark / export ONNX. Trả về bản copy."""
    m = copy.deepcopy(model).eval()
    for mod in m.modules():
        if isinstance(mod, RepConv):
            mod.reparameterize()
    return m


class FastDBNet(nn.Module):
    """Nhận ảnh BGR (B,3,S,S); trả logits vùng lõi chữ (B,1,S/2,S/2)."""

    def __init__(self, width=1.0, fpn_ch=48):
        super().__init__()
        c = lambda v: max(8, int(round(v * width / 8)) * 8)
        c1, c2, c3, c4 = (c(16), c(32), c(64), c(96))
        self.stem = RepConv(3, c1, 3, 2)
        self.s4 = nn.Sequential(MOBlock(c1, c2, 2), MOBlock(c2, c2))
        self.s8 = nn.Sequential(MOBlock(c2, c3, 2), MOBlock(c3, c3))
        self.s16 = nn.Sequential(MOBlock(c3, c4, 2), MOBlock(c4, c4, k=5), MOBlock(c4, c4, k=5))
        C = fpn_ch
        self.lat2, self.lat3, self.lat4 = (
            ConvBNAct(c2, C, 1, act=False),
            ConvBNAct(c3, C, 1, act=False),
            ConvBNAct(c4, C, 1, act=False),
        )
        self.smooth = nn.Sequential(ConvBNAct(C, C, 3, groups=C), ConvBNAct(C, C, 1))
        self.head = nn.Sequential(
            ConvBNAct(C, C, 3, groups=C),
            ConvBNAct(C, C // 2, 1),
            nn.ConvTranspose2d(C // 2, C // 4, 2, 2, bias=False),
            nn.BatchNorm2d(C // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 4, 1, 1),
        )

    def forward(self, x):
        x = (x - 0.5) * 2.0
        f2 = self.s4(self.stem(x))
        f3 = self.s8(f2)
        f4 = self.s16(f3)
        p4 = self.lat4(f4)
        p3 = self.lat3(f3) + F.interpolate(p4, scale_factor=2.0, mode="nearest")
        p2 = self.lat2(f2) + F.interpolate(p3, scale_factor=2.0, mode="nearest")
        return self.head(self.smooth(p2))


class MHSA(nn.Module):
    """Lớp attention tự viết, hỗ trợ xuất ONNX."""

    def __init__(self, dim, heads=4, mlp_ratio=2.0, drop=0.1):
        super().__init__()
        self.h = heads
        self.dh = dim // heads
        self.n1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        a = torch.softmax(q @ k.transpose(-2, -1) * self.dh ** (-0.5), dim=-1)
        y = (a @ v).transpose(1, 2).reshape(B, T, C)
        x = x + self.drop(self.proj(y))
        return x + self.drop(self.mlp(self.n2(x)))


class ConvSeqBlock(nn.Module):
    """Khối convolution cho chuỗi, có nhánh cộng tắt."""

    def __init__(self, dim, k=5):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, k, padding=k // 2, groups=dim, bias=False)
        self.bn1 = nn.BatchNorm1d(dim)
        self.pw = nn.Conv1d(dim, dim, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        y = x.transpose(1, 2)
        y = F.relu(self.bn1(self.dw(y)))
        y = self.bn2(self.pw(y))
        return F.relu(x + y.transpose(1, 2))


class FastCRNN(nn.Module):
    """Nhận ảnh BGR (B,3,H,W); trả logits CTC (B,W/4,số lớp)."""

    def __init__(
        self, num_classes, img_h=32, img_w=128, width=1.0, neck="attn", neck_layers=1, dim=128
    ):
        super().__init__()
        c = lambda v: max(8, int(round(v * width / 8)) * 8)
        c1, c2, c3, c4 = (c(32), c(64), c(96), c(dim))
        self.backbone = nn.Sequential(
            RepConv(3, c1, 3, 2),
            MOBlock(c1, c2, 2),
            MOBlock(c2, c2),
            MOBlock(c2, c3, (2, 1)),
            MOBlock(c3, c3),
            MOBlock(c3, c3, k=5),
            MOBlock(c3, c4, (2, 1)),
            MOBlock(c4, c4, k=5),
        )
        hf = img_h // 16
        self.collapse = (
            ConvBNAct(c4, c4, k=1)
            if hf == 1
            else nn.Sequential(
                nn.Conv2d(c4, c4, (hf, 1), bias=False), nn.BatchNorm2d(c4), nn.ReLU(inplace=True)
            )
        )
        self.T = img_w // 4
        self.neck_type = neck
        if neck == "attn":
            self.pos = nn.Parameter(torch.zeros(1, self.T, c4))
            nn.init.trunc_normal_(self.pos, std=0.02)
            self.neck = nn.Sequential(*[MHSA(c4) for _ in range(neck_layers)])
            self.norm = nn.LayerNorm(c4)
        elif neck == "conv":
            self.neck = nn.Sequential(*[ConvSeqBlock(c4) for _ in range(max(1, neck_layers * 2))])
            self.norm = nn.Identity()
        elif neck == "lstm":
            self.neck = nn.LSTM(
                c4, c4 // 2, num_layers=neck_layers, bidirectional=True, batch_first=True
            )
            self.norm = nn.Identity()
        else:
            raise ValueError(neck)
        self.fc = nn.Linear(c4, num_classes)

    def forward(self, x):
        x = (x - 0.5) * 2.0
        x = self.collapse(self.backbone(x))
        x = x.squeeze(2).transpose(1, 2)
        if self.neck_type == "attn":
            x = self.norm(self.neck(x + self.pos))
        elif self.neck_type == "lstm":
            x, _ = self.neck(x)
        else:
            x = self.neck(x)
        return self.fc(x)

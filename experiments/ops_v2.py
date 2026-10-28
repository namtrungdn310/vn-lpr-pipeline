"""Fit bốn cạnh v2 tách từ notebook, không chạy cell train."""

import math
import cv2
import numpy as np


def _fit_line(x, y, default):
    """Bình phương tối thiểu y = k·x + b (cần ≥3 điểm trải đủ rộng)."""
    if len(x) >= 3 and np.ptp(x) > 1.0:
        xm, ym = (x.mean(), y.mean())
        k = float(((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum())
        return (k, float(ym - k * xm))
    return default


def fit_quad(cnt, unclip=0.0, pad_x=0.0):
    """Fit cạnh trên/dưới riêng; nới bốn cạnh và hai đầu để bao vùng chữ."""
    p = cnt.reshape(-1, 2).astype(np.float32)
    (cx, cy), (w, h), ang = cv2.minAreaRect(p)
    a = math.radians(ang if w >= h else ang + 90)
    U = np.array([math.cos(a), math.sin(a)], np.float32)
    if U[0] < 0:
        U = -U
    V = np.array([-U[1], U[0]], np.float32)
    c = np.array([cx, cy], np.float32)
    pu, pv = ((p - c) @ U, (p - c) @ V)
    mu = (pu > pu.min() + 0.15 * np.ptp(pu)) & (pu < pu.max() - 0.15 * np.ptp(pu))
    kt, bt = _fit_line(pu[mu & (pv < 0)], pv[mu & (pv < 0)], (0.0, float(pv.min())))
    kb, bb = _fit_line(pu[mu & (pv > 0)], pv[mu & (pv > 0)], (0.0, float(pv.max())))
    nl, nr = (float(pu.min()), float(pu.max()))
    corners = lambda: np.array(
        [
            c + u * U + (k * u + b) * V
            for u, k, b in ((nl, kt, bt), (nr, kt, bt), (nr, kb, bb), (nl, kb, bb))
        ],
        np.float32,
    )
    if unclip:
        q = corners()
        D = abs(cv2.contourArea(q)) * unclip / max(cv2.arcLength(q, True), 1e-06)
        bt -= D * math.hypot(1, kt)
        bb += D * math.hypot(1, kb)
        nl -= D
        nr += D
    if pad_x:
        nl -= pad_x * (bb - bt)
        nr += pad_x * (bb - bt)
    return corners()

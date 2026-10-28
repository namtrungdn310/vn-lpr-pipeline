"""Đề xuất bốn cạnh theo nét chữ; cần xem và duyệt trước khi áp dụng."""

import cv2
import numpy as np


def _slope(x, y, separation):
    slopes = [
        (yb - ya) / (xb - xa)
        for xa, ya in zip(x, y)
        for xb, yb in zip(x, y)
        if xb - xa > separation
    ]
    return float(np.clip(np.median(slopes), -0.35, 0.35)) if slopes else 0.0


def glyph_quad(rgb, probability):
    """Tìm nét chữ tối, loại viền và fit bốn cạnh có lề nhỏ.

    Xác suất chỉ dùng chọn nét; ảnh RGB giữ nguyên và khung phải chứa đủ nét đã chọn.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    scale = 3
    enlarged = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    kh = max(3, round(h * scale * 0.25) // 2 * 2 + 1)
    kw = max(3, round(h * scale * 0.25) // 2 * 2 + 1)
    background = cv2.morphologyEx(enlarged, cv2.MORPH_CLOSE, np.ones((kh, kw), np.uint8))
    contrast = cv2.subtract(background, enlarged)
    threshold, _ = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (contrast > max(3, threshold * 0.90)).astype(np.uint8)
    prob = cv2.resize(
        np.asarray(probability, dtype=np.float32),
        (w * scale, h * scale),
        interpolation=cv2.INTER_LINEAR,
    )
    count, labels, stats, centres = cv2.connectedComponentsWithStats(mask, 8)
    components = []
    candidates = []
    for number, (x, y, cw, ch, area) in enumerate(stats[1:], 1):
        if ch < h * scale * 0.30 or area < max(8, h * h * scale * scale * 0.007):
            continue
        if cw > w * scale * 0.95 or cw < max(1, h * scale * 0.018):
            continue
        yy, xx = np.where(labels == number)
        confidence = float(prob[yy, xx].mean())
        xx = np.clip((xx + 0.5) / scale - 0.5, 0, w - 1)
        yy = np.clip((yy + 0.5) / scale - 0.5, 0, h - 1)
        candidates.append(
            {
                "x": float(xx.mean()),
                "y": float(yy.mean()),
                "top": float(np.quantile(yy, 0.008)),
                "bottom": float(np.quantile(yy, 0.992)),
                "width": cw / scale,
                "height": ch / scale,
                "probability": confidence,
                "xs": xx,
                "ys": yy,
            }
        )
    supported = [c for c in candidates if c["probability"] >= 0.022]
    if supported:
        typical_height = float(np.median([c["height"] for c in supported]))
        typical_y = float(np.median([c["y"] for c in supported]))
        supported_left = min(float(c["xs"].min()) for c in supported)
        supported_right = max(float(c["xs"].max()) for c in supported)
        for c in candidates:
            if c["probability"] >= 0.022:
                components.append(c)
            elif (
                0.6 * typical_height <= c["height"] <= 1.06 * typical_height
                and abs(c["y"] - typical_y) < h * 0.16
                and c["xs"].max() >= supported_left - h * 0.15
                and c["xs"].min() <= supported_right + h * 0.15
            ):
                # Giữ ký tự mờ ở đầu/cuối dù DB bỏ sót mép chữ.
                components.append(c)
    if not components:
        return {"status": "needs_visual_bounds", "reason": "no_supported_glyph_components"}
    xs = np.concatenate([c["xs"] for c in components])
    ys = np.concatenate([c["ys"] for c in components])
    if np.ptp(xs) < w * 0.30 or np.ptp(ys) < h * 0.25:
        return {"status": "needs_visual_bounds", "reason": "incomplete_line_support"}
    fit = [c for c in components if c["width"] < h * 1.25]
    cx = [c["x"] for c in fit]
    top_slope = _slope(cx, [c["top"] for c in fit], w * 0.25)
    bottom_slope = _slope(cx, [c["bottom"] for c in fit], w * 0.25)
    # Chừa lề nhỏ để giữ nét mờ và phần ảnh hưởng của nội suy.
    pad = max(0.65, min(1.1, h * 0.025))
    top = float(np.min(ys - top_slope * xs)) - pad
    bottom = float(np.max(ys - bottom_slope * xs)) + pad
    left = max(0.0, float(xs.min()) - pad)
    right = min(w - 1.0, float(xs.max()) + pad)
    # Ước lượng độ nghiêng của mép chữ, không bám theo viền biển.
    lateral = []
    edge_slopes = []
    for side in ("left", "right"):
        edge = []
        for a, b in zip(
            np.linspace(ys.min(), ys.max(), 13)[:-1], np.linspace(ys.min(), ys.max(), 13)[1:]
        ):
            values = xs[(ys >= a) & (ys <= b)]
            if len(values):
                edge.append(((a + b) / 2, float(values.min() if side == "left" else values.max())))
        if len(edge) >= 4:
            yv, xv = zip(*edge)
            slope = _slope(yv, xv, h * 0.25)
        else:
            slope = 0.0
        edge_slopes.append(slope)
    # Dùng chung trục đứng của chữ; bám theo nét xiên số 7 có thể làm méo crop.
    side_slope = float(np.clip(np.mean(edge_slopes), -0.12, 0.12))
    lateral = [
        (side_slope, float(np.min(xs - side_slope * ys)) - pad),
        (side_slope, float(np.max(xs - side_slope * ys)) + pad),
    ]

    def corner(vertical, horizontal):
        a, b = vertical
        c, d = horizontal  # Hai đường: x=a*y+b và y=c*x+d.
        x = (a * d + b) / (1 - a * c)
        return [x, c * x + d]

    quad = np.array(
        [
            corner(lateral[0], (top_slope, top)),
            corner(lateral[1], (top_slope, top)),
            corner(lateral[1], (bottom_slope, bottom)),
            corner(lateral[0], (bottom_slope, bottom)),
        ],
        dtype=np.float32,
    )
    if np.any(quad < [-w * 0.15, -h * 0.15]) or np.any(quad > [w * 1.15, h * 1.15]):
        return {"status": "needs_visual_bounds", "reason": "unstable_edge_intersections"}
    # Thu lề theo biên ảnh nhưng không được cắt vào nét chữ đã chọn.
    omitted = sum(
        cv2.pointPolygonTest(quad, (float(x), float(y)), True) < -0.4 for x, y in zip(xs, ys)
    )
    if omitted > max(2, len(xs) * 0.003):
        return {"status": "needs_visual_bounds", "reason": "foreground_outside_quad"}
    ratio = float(cv2.contourArea(quad) / (max(1, (w - 1) * (h - 1))))
    return {
        "status": "proposed",
        "inner": quad.tolist(),
        "ratio": ratio,
        "components": [{k: v for k, v in c.items() if k not in ("xs", "ys")} for c in components],
        "top_slope": top_slope,
        "bottom_slope": bottom_slope,
        "mask_bounds": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
    }

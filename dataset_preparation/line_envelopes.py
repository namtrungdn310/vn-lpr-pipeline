"""Đề xuất box theo nét chữ trên ảnh gốc; tách riêng từng tầng và chờ duyệt."""

import cv2
import numpy as np


def find_row_overlaps(rows, tolerance=0.1):
    """Tìm các khung dòng chữ giao nhau trên cùng ảnh."""
    from collections import defaultdict

    groups = defaultdict(list)
    for r in rows:
        groups[r["det_image"]].append(r)
    findings = []
    for path, group in groups.items():
        for index, a in enumerate(group):
            for b in group[index + 1 :]:
                area = float(
                    cv2.intersectConvexConvex(np.float32(a["points"]), np.float32(b["points"]))[0]
                )
                if area > tolerance:
                    findings.append(
                        {
                            "det_image": path,
                            "object_indices": [a["object_index"], b["object_index"]],
                            "area": area,
                        }
                    )
    return findings


def robust_slope(x, y, minimum):
    pairs = [
        (yb - ya) / (xb - xa) for xa, ya in zip(x, y) for xb, yb in zip(x, y) if xb - xa > minimum
    ]
    return float(np.median(pairs)) if pairs else 0.0


def propose_line(rgb, points, neighbours, text):
    p = np.asarray(points, np.float32)
    direction = (p[1] - p[0]) + (p[2] - p[3])
    angle = np.degrees(np.arctan2(direction[1], direction[0]))
    image_h, image_w = rgb.shape[:2]
    transform = cv2.getRotationMatrix2D((image_w / 2, image_h / 2), float(angle), 1.0)
    corners = cv2.transform(
        np.float32([[[0, 0], [image_w - 1, 0], [image_w - 1, image_h - 1], [0, image_h - 1]]]),
        transform,
    )[0]
    minimum = corners.min(axis=0)
    maximum = corners.max(axis=0)
    transform[:, 2] -= minimum
    rotated = cv2.warpAffine(
        rgb,
        transform,
        tuple(np.ceil(maximum - minimum + 1).astype(int)),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    q = cv2.transform(p[None], transform)[0]
    centre = q.mean(axis=0)
    expected_h = float((np.linalg.norm(p[3] - p[0]) + np.linalg.norm(p[2] - p[1])) / 2)
    margin = max(2, expected_h * 0.13)
    left = max(0, int(np.floor(q[:, 0].min() - margin)))
    right = min(rotated.shape[1], int(np.ceil(q[:, 0].max() + margin + 1)))
    top = max(0, int(np.floor(q[:, 1].min() - margin)))
    bottom = min(rotated.shape[0], int(np.ceil(q[:, 1].max() + margin + 1)))
    for other in neighbours:
        oc = cv2.transform(np.asarray(other, np.float32)[None], transform)[0].mean(axis=0)
        boundary = (centre[1] + oc[1]) / 2
        if oc[1] < centre[1]:
            top = max(top, int(np.floor(boundary)))
        else:
            bottom = min(bottom, int(np.ceil(boundary)))
    if right - left < 5 or bottom - top < 5:
        return {"status": "uncertain", "reason": "small_roi"}
    roi = rotated[top:bottom, left:right]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    scale = 3
    g = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    kh = max(3, round(expected_h * scale * 0.45) // 2 * 2 + 1)
    background = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((kh, kh), np.uint8))
    contrast = cv2.subtract(background, g)
    threshold, _ = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants = []
    for factor in (0.65, 0.85, 1.05):
        mask = (contrast > max(4, threshold * factor)).astype(np.uint8)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        components = []
        for number, (x, y, w, h, area) in enumerate(stats[1:], 1):
            width, height = w / scale, h / scale
            if not (0.38 * expected_h <= height <= 1.48 * expected_h):
                continue
            if width > expected_h * 1.7 or area < max(8, expected_h**2 * scale**2 * 0.005):
                continue
            yy, xx = np.where(labels == number)
            xs = (xx + 0.5) / scale - 0.5 + left
            ys = (yy + 0.5) / scale - 0.5 + top
            if abs(float(ys.mean()) - centre[1]) > expected_h * 0.47:
                continue
            components.append(
                {
                    "xs": xs,
                    "ys": ys,
                    "height": height,
                    "width": width,
                    "x": float(xs.mean()),
                    "y": float(ys.mean()),
                    "top": float(np.quantile(ys, 0.005)),
                    "bottom": float(np.quantile(ys, 0.995)),
                }
            )
        if not components:
            continue
        median_h = float(np.median([c["height"] for c in components]))
        components = [c for c in components if c["height"] >= median_h * 0.67]
        variants.append((abs(len(components) - len(text)), abs(factor - 0.85), factor, components))
    if not variants:
        return {"status": "uncertain", "reason": "no_character_components"}
    _, _, factor, components = min(variants, key=lambda v: (v[0], v[1]))
    components.sort(key=lambda c: c["x"])
    xs = np.concatenate([c["xs"] for c in components])
    ys = np.concatenate([c["ys"] for c in components])
    width = float(np.ptp(xs))
    top_slope = robust_slope(
        [c["x"] for c in components], [c["top"] for c in components], width * 0.22
    )
    bottom_slope = robust_slope(
        [c["x"] for c in components], [c["bottom"] for c in components], width * 0.22
    )
    pad = max(0.8, min(1.25, expected_h * 0.025))
    cap = float(np.min(ys - top_slope * xs)) - pad
    base = float(np.max(ys - bottom_slope * xs)) + pad
    # Ước lượng trục đứng từ nhiều ký tự để tránh lệch theo một nét cong hoặc xiên.
    slopes = []
    for c in components:
        cx = c["xs"]
        cy = c["ys"]
        bands = np.linspace(np.quantile(cy, 0.15), np.quantile(cy, 0.85), 8)
        edge = []
        for a, b in zip(bands[:-1], bands[1:]):
            ink = cx[(cy >= a) & (cy <= b)]
            if len(ink):
                edge.append(
                    ((a + b) / 2, float((np.quantile(ink, 0.05) + np.quantile(ink, 0.95)) / 2))
                )
        if len(edge) >= 4:
            yv, xv = zip(*edge)
            slope = robust_slope(yv, xv, c["height"] * 0.3)
            if abs(slope) < 0.6:
                slopes.append(slope)
    side = float(np.median(slopes)) if slopes else 0.0
    lo = float(np.min(xs - side * ys)) - pad
    hi = float(np.max(xs - side * ys)) + pad

    def corner(offset, slope, intercept):
        x = (side * intercept + offset) / (1 - side * slope)
        return [x, slope * x + intercept]

    proposal = np.float32(
        [
            corner(lo, top_slope, cap),
            corner(hi, top_slope, cap),
            corner(hi, bottom_slope, base),
            corner(lo, bottom_slope, base),
        ]
    )
    inverse = cv2.invertAffineTransform(transform)
    native = cv2.transform(proposal[None], inverse)[0]
    native = np.clip(native, [0, 0], [image_w - 1, image_h - 1])
    # Đối chiếu bốn góc cũ với đỉnh và chân chữ trên ảnh đã xoay.
    expected_top = top_slope * q[:, 0] + cap
    expected_bottom = bottom_slope * q[:, 0] + base
    corner_excess = [
        float(expected_top[0] - q[0, 1]),
        float(expected_top[1] - q[1, 1]),
        float(q[2, 1] - expected_bottom[2]),
        float(q[3, 1] - expected_bottom[3]),
    ]
    relative_area = abs(cv2.contourArea(native.astype(np.float32))) / max(
        0.01, abs(cv2.contourArea(p))
    )
    mismatch = len(components) != len(text)
    unstable = (
        abs(top_slope) > 0.6
        or abs(bottom_slope) > 0.6
        or abs(top_slope - bottom_slope) * width > expected_h * 0.4
    )
    status = "uncertain" if mismatch or unstable else "supported"
    return {
        "status": status,
        "reason": (
            "component_count_or_edge_fit"
            if status == "uncertain"
            else "independent_native_cap_baseline_fit"
        ),
        "points": native.tolist(),
        "component_count": len(components),
        "expected_characters": len(text),
        "threshold_factor": factor,
        "relative_area": relative_area,
        "corner_excess": corner_excess,
        "max_excess_height": max(corner_excess) / expected_h,
        "max_missing_height": -min(corner_excess) / expected_h,
        "top_slope": top_slope,
        "bottom_slope": bottom_slope,
        "side_slope": side,
        "component_bounds": [
            [float(c["xs"].min()), c["top"], float(c["xs"].max()), c["bottom"]] for c in components
        ],
    }

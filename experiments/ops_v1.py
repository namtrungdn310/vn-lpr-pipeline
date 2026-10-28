"""Hình học/hậu xử lý v1 tách từ upstream, không cần torch khi inference."""

from typing import Tuple
import cv2
import numpy as np


def letterbox_resize(
    image: np.ndarray, target_size: Tuple[int, int] = (128, 32), pad_value: int = 0
) -> np.ndarray:
    target_w, target_h = target_size
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_crop(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    rect = order_points(box)
    tl, tr, br, bl = rect
    max_w = max(1, int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    max_h = max(1, int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype=np.float32
    )
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, m, (max_w, max_h))


def get_box_height(box: np.ndarray) -> float:
    box = order_points(box.astype(np.float32))
    tl, tr, br, bl = box
    return float((np.linalg.norm(tr - br) + np.linalg.norm(tl - bl)) / 2.0)


def build_box_info_from_mask(component_mask: np.ndarray, min_area: int = 20):
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    pts = []
    total_area = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 3:
            continue
        total_area += area
        pts.append(cnt.reshape(-1, 2))
    if not pts or total_area < min_area:
        return None
    pts = np.vstack(pts).astype(np.float32)
    x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
    if w < 4 or h < 4:
        return None
    rect = cv2.minAreaRect(pts)
    box = order_points(cv2.boxPoints(rect))
    return {
        "box": box.astype(np.float32),
        "area": float(total_area),
        "cy": float(np.mean(box[:, 1])),
        "axis_height": float(h),
        "rotated_height": get_box_height(box),
        "bbox": (int(x), int(y), int(w), int(h)),
    }


def split_tall_component_by_projection(
    component_mask: np.ndarray, prob_up: np.ndarray, min_area: int = 20
):
    ys, xs = np.where(component_mask > 0)
    if len(xs) == 0:
        return []
    y_min = int(ys.min())
    y_max = int(ys.max())
    h = y_max - y_min + 1
    if h < 40:
        return []
    local_mask = component_mask[y_min : y_max + 1, :]
    local_prob = prob_up[y_min : y_max + 1, :]
    row_count = (local_mask > 0).sum(axis=1).astype(np.float32)
    row_prob = (local_prob * (local_mask > 0)).sum(axis=1).astype(np.float32)
    if row_count.max() <= 0 or row_prob.max() <= 0:
        return []
    row_count_n = row_count / max(1e-06, row_count.max())
    row_prob_n = row_prob / max(1e-06, row_prob.max())
    score = 0.55 * row_prob_n + 0.45 * row_count_n
    score = cv2.GaussianBlur(score.reshape(-1, 1), (1, 9), 0).flatten()
    search_top = int(h * 0.32)
    search_bot = int(h * 0.68)
    if search_bot <= search_top:
        return []
    split_local = search_top + int(np.argmin(score[search_top : search_bot + 1]))
    zero_rows = np.where(row_count[search_top : search_bot + 1] == 0)[0]
    if len(zero_rows) > 0:
        split_local = search_top + int(zero_rows[len(zero_rows) // 2])
    split_y = y_min + split_local
    upper_area = row_count[:split_local].sum()
    lower_area = row_count[split_local:].sum()
    upper_h = np.count_nonzero(row_count[:split_local] > 0)
    lower_h = np.count_nonzero(row_count[split_local:] > 0)
    if upper_area < min_area or lower_area < min_area or upper_h < 8 or (lower_h < 8):
        return []
    upper = np.zeros_like(component_mask)
    lower = np.zeros_like(component_mask)
    upper[:split_y, :] = component_mask[:split_y, :]
    lower[split_y:, :] = component_mask[split_y:, :]
    infos = []
    upper_info = build_box_info_from_mask(upper, min_area=min_area)
    lower_info = build_box_info_from_mask(lower, min_area=min_area)
    if upper_info is not None:
        infos.append(upper_info)
    if lower_info is not None:
        infos.append(lower_info)
    return sorted(infos, key=lambda x: x["cy"])


def get_boxes_from_overlay_mask(
    mask: np.ndarray,
    prob_up: np.ndarray,
    min_area: int = 20,
    max_lines: int = 2,
    max_single_line_height: int = 100,
):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    infos = []
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        component_mask = np.zeros_like(mask)
        component_mask[labels == label_id] = 255
        info = build_box_info_from_mask(component_mask, min_area=min_area)
        if info is None:
            continue
        if info["rotated_height"] > max_single_line_height:
            split_infos = split_tall_component_by_projection(
                component_mask, prob_up, min_area=min_area
            )
            infos.extend(split_infos if len(split_infos) >= 2 else [info])
        else:
            infos.append(info)
    if not infos:
        return ([], [])
    infos = sorted(infos, key=lambda x: x["area"], reverse=True)[:max_lines]
    infos = sorted(infos, key=lambda x: x["cy"])
    return ([i["box"] for i in infos], infos)


def pad_rotated_box(box: np.ndarray, pad: int, img_w: int, img_h: int) -> np.ndarray:
    box = order_points(box.astype(np.float32))
    center = np.mean(box, axis=0)
    padded = []
    for p in box:
        v = p - center
        norm = np.linalg.norm(v)
        p_new = center + v / norm * (norm + pad) if norm > 1e-06 else p
        p_new[0] = np.clip(p_new[0], 0, img_w - 1)
        p_new[1] = np.clip(p_new[1], 0, img_h - 1)
        padded.append(p_new)
    return np.array(padded, dtype=np.float32)

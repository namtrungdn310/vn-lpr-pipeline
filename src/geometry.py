"""Kiểm tra quad và nắn phối cảnh vùng chữ."""

import io
import math

import numpy as np
from PIL import Image


def signed_area(points: list) -> float:
    return sum(
        points[i][0] * points[(i + 1) % len(points)][1]
        - points[(i + 1) % len(points)][0] * points[i][1]
        for i in range(len(points))
    ) / 2.0


def cross(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def validate_quad(points: list, width: int, height: int) -> list:
    """Quad lồi, đủ bốn góc, nằm trong [0,W] × [0,H]."""
    if not isinstance(points, (list, tuple)) or len(points) != 4:
        raise ValueError("Quad phải có bốn góc")
    if not all(
        isinstance(point, (list, tuple))
        and len(point) == 2
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in point
        )
        for point in points
    ):
        raise ValueError("Tọa độ phải là số hữu hạn")
    quad = [list(map(float, point)) for point in points]
    if not all(0 <= x <= width and 0 <= y <= height for x, y in quad):
        raise ValueError("Quad vượt biên ảnh")
    turns = [cross(quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]) for i in range(4)]
    if not (all(turn > 1e-8 for turn in turns) or all(turn < -1e-8 for turn in turns)):
        raise ValueError("Quad suy biến, không lồi hoặc tự cắt")
    return quad


def orient_text_quad(points: list) -> list:
    """Quy tắc cũ dùng cho những recipe đã được duyệt."""
    quad = [list(map(float, point)) for point in points]
    if signed_area(quad) < 0:
        quad = [quad[0], quad[3], quad[2], quad[1]]
    lengths = [math.dist(quad[i], quad[(i + 1) % 4]) for i in range(4)]
    if lengths[0] + lengths[2] < lengths[1] + lengths[3]:
        quad = quad[1:] + quad[:1]
    if quad[1][0] < quad[0][0]:
        quad = quad[2:] + quad[:2]
    return quad


def rectify_crop(image: Image.Image, points: list, *, preserve_text_order=False) -> Image.Image:
    """Nắn vùng chữ; bật preserve_text_order khi đã biết thứ tự TL,TR,BR,BL."""
    quad = validate_quad(points, image.width, image.height)
    if not preserve_text_order:
        quad = orient_text_quad(quad)
    width = max(2, round(max(math.dist(quad[0], quad[1]), math.dist(quad[2], quad[3]))))
    height = max(2, round(max(math.dist(quad[0], quad[3]), math.dist(quad[1], quad[2]))))
    if width * height > image.width * image.height * 16:
        raise ValueError("Kích thước crop bất thường")
    output_points = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
    source_points = [
        [min(max(x, 0), image.width - 1), min(max(y, 0), image.height - 1)]
        for x, y in quad
    ]
    matrix, target = [], []
    for (x, y), (u, v) in zip(output_points, source_points):
        matrix.extend((
            [x, y, 1, 0, 0, 0, -u * x, -u * y],
            [0, 0, 0, x, y, 1, -v * x, -v * y],
        ))
        target.extend((u, v))
    try:
        coefficients = np.linalg.solve(
            np.asarray(matrix, dtype=np.float64), np.asarray(target, dtype=np.float64)
        )
    except np.linalg.LinAlgError as error:
        raise ValueError("Không tìm được phép nắn phối cảnh") from error
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Hệ số phối cảnh không hữu hạn")
    return image.convert("RGB").transform(
        (width, height), Image.Transform.PERSPECTIVE,
        tuple(map(float, coefficients)), Image.Resampling.BICUBIC,
    )


def encoded_jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(
        buffer, format="JPEG", quality=95, subsampling=0, optimize=False
    )
    return buffer.getvalue()


def intersection_area(subject: list, clip: list) -> float:
    """Diện tích giao thật của hai polygon lồi."""
    output = [tuple(point) for point in subject]
    orientation = 1 if signed_area(clip) >= 0 else -1
    for index, start in enumerate(clip):
        end = clip[(index + 1) % len(clip)]
        source, output = output, []
        if not source:
            break
        previous = source[-1]
        for current in source:
            current_side = orientation * cross(start, end, current)
            previous_side = orientation * cross(start, end, previous)
            current_inside, previous_inside = current_side >= -1e-8, previous_side >= -1e-8
            if current_inside != previous_inside:
                denominator = previous_side - current_side
                if abs(denominator) > 1e-8:
                    ratio = previous_side / denominator
                    output.append((
                        previous[0] + ratio * (current[0] - previous[0]),
                        previous[1] + ratio * (current[1] - previous[1]),
                    ))
            if current_inside:
                output.append(current)
            previous = current
    return abs(signed_area(output)) if len(output) >= 3 else 0.0

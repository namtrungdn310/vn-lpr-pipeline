"""Xem ảnh thật và nhãn hiện tại; không ghi vào dataset."""

import math
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .dataset import Dataset, SPLITS
from .geometry import rectify_crop

COLORS = ("#00852d", "#c45200")


def font(size):
    for name in ("C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def choose(items, count, seed):
    if count < 1:
        raise ValueError("Số ảnh phải lớn hơn 0")
    return random.Random(seed).sample(items, min(count, len(items)))


def det_grid(dataset: Dataset, split="val", lines=2, count=12, seed=42) -> Image.Image:
    if split not in SPLITS or lines not in (1, 2):
        raise ValueError("Chọn split hợp lệ và biển 1 hoặc 2 tầng")
    samples = choose(
        [
            (name, objects)
            for name, objects in dataset.det.items()
            if name.startswith(split + "/") and len(objects) == lines
        ],
        count,
        seed,
    )
    if not samples:
        raise ValueError("Không có ảnh trong nhóm đã chọn")
    width, height, columns = 460, 310, 2
    sheet = Image.new(
        "RGB", (columns * width, math.ceil(len(samples) / columns) * height), "#f2f2f2"
    )
    draw = ImageDraw.Draw(sheet)
    for index, (name, objects) in enumerate(samples):
        x, y = index % columns * width, index // columns * height
        draw.text((x + 12, y + 6), name, font=font(18), fill="black")
        with Image.open(dataset.root / name) as source:
            original_size = source.size
            panel = ImageOps.contain(
                source.convert("RGB"), (width - 28, 220), Image.Resampling.LANCZOS
            )
        sx, sy = panel.width / original_size[0], panel.height / original_size[1]
        panel_draw = ImageDraw.Draw(panel)
        for number, obj in enumerate(objects):
            points = [(px * sx, py * sy) for px, py in obj["points"]]
            color = COLORS[number % len(COLORS)]
            panel_draw.line(points + points[:1], fill=color, width=2)
            draw.text(
                (x + 12, y + 258 + number * 24),
                f"Dòng {number + 1}: {obj['transcription']}",
                font=font(17),
                fill=color,
            )
        sheet.paste(panel, (x + (width - panel.width) // 2, y + 32 + (220 - panel.height) // 2))
    return sheet


def rec_grid(dataset: Dataset, split="val", count=18, seed=42) -> Image.Image:
    if split not in SPLITS:
        raise ValueError("Split không hợp lệ")
    samples = choose(
        [(name, text) for name, text in dataset.rec.items() if name.startswith(split + "/")],
        count,
        seed,
    )
    if not samples:
        raise ValueError("Không có crop REC")
    columns, width, height = 6, 230, 138
    sheet = Image.new(
        "RGB", (columns * width, math.ceil(len(samples) / columns) * height), "#f2f2f2"
    )
    draw = ImageDraw.Draw(sheet)
    for index, (name, text) in enumerate(samples):
        x, y = index % columns * width, index // columns * height
        draw.text((x + 8, y + 4), Path(name).name, font=font(17), fill="#444444")
        draw.text((x + 8, y + 26), f"Nhãn: {text}", font=font(18), fill="black")
        with Image.open(dataset.root / name) as source:
            panel = ImageOps.contain(
                source.convert("RGB"), (width - 16, 78), Image.Resampling.LANCZOS
            )
        sheet.paste(panel, (x + (width - panel.width) // 2, y + 53 + (78 - panel.height) // 2))
    return sheet


def proposal_grid(dataset: Dataset, proposals: list, count=6, offset=0) -> Image.Image:
    """Cam: khung đầu vào; xanh: đề xuất. Crop bên phải là trước/sau."""
    if count < 1 or offset < 0:
        raise ValueError("Số mẫu hoặc vị trí xem không hợp lệ")
    rows = proposals[offset : offset + count]
    if not rows:
        raise ValueError("Không có đề xuất để xem")
    width, height, columns = 600, 340, 2
    sheet = Image.new("RGB", (width * columns, math.ceil(len(rows) / columns) * height), "#f2f2f2")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        x, y = index % columns * width, index // columns * height
        draw.text((x + 8, y + 5), row["det_image"], font=font(16), fill="black")
        draw.text(
            (x + 8, y + 27),
            f"{row['text']} | {row['diagnostics']['status']}",
            font=font(16),
            fill="black",
        )
        with Image.open(dataset.root / row["det_image"]) as opened:
            source = opened.convert("RGB")
        panel = ImageOps.contain(source, (240, 260), Image.Resampling.LANCZOS)
        points_after = row["proposal_points"]
        painter = ImageDraw.Draw(panel)
        for points, color in ((row["input_points"], "#de7100"), (points_after, "#00852d")):
            if points is not None:
                scaled = [
                    (a * panel.width / source.width, b * panel.height / source.height)
                    for a, b in points
                ]
                painter.line(scaled + scaled[:1], fill=color, width=2)
        sheet.paste(panel, (x + 8 + (240 - panel.width) // 2, y + 60))
        for points, dy, title in (
            (row["input_points"], 70, "Before"),
            (points_after, 205, "Proposal"),
        ):
            draw.text((x + 265, y + dy - 21), title, font=font(16), fill="black")
            if points is not None:
                try:
                    crop = rectify_crop(source, points, preserve_text_order=True)
                    crop = ImageOps.contain(crop, (320, 100), Image.Resampling.LANCZOS)
                    sheet.paste(crop, (x + 265, y + dy))
                except ValueError:
                    draw.text((x + 265, y + dy), "Invalid quad", font=font(16), fill="red")
    return sheet

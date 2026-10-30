"""Chuẩn bị đầu vào REC, giữ tỉ lệ và đủ ký tự."""

from pathlib import Path

from PIL import Image

from .dataset import SPLITS, read_label_rows, validate_text
from .geometry import rectify_crop


def prepare_rec_crop(image: Image.Image, text_quad: list) -> Image.Image:
    """Dùng khi inference: quad một dòng, theo TL,TR,BR,BL."""
    return rectify_crop(image, text_quad, preserve_text_order=True)


def resize_pad(image: Image.Image, height=48, width=320) -> Image.Image:
    """Giữ tỉ lệ; thêm nền bên phải/dưới, không cắt chữ."""
    if height < 1 or width < 1:
        raise ValueError("Kích thước REC phải dương")
    scale = min(height / image.height, width / image.width)
    resized = image.convert("RGB").resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.BICUBIC,
    )
    background = resized.getpixel((resized.width - 1, resized.height // 2))
    output = Image.new("RGB", (width, height), background)
    output.paste(resized, (0, 0))
    return output


def iter_rec_samples(dataset_root, split="train"):
    if split not in SPLITS:
        raise ValueError("Split phải là train, val hoặc test")
    directory = (Path(dataset_root) / split / "rec").resolve(strict=True)
    seen = set()
    for relative, text in read_label_rows(directory / "rec_labels.txt"):
        validate_text(text)
        if relative in seen:
            raise ValueError(f"Ảnh REC bị lặp: {relative}")
        seen.add(relative)
        yield directory / relative, text

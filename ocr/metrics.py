"""ACC/CER chung cho ba hệ thống, không tự sửa O thành 0."""

import math


def normalize(text):
    return "".join(c for c in str(text).upper() if c.isascii() and c.isalnum())


def edit_distance(first, second):
    previous = list(range(len(second) + 1))
    for i, a in enumerate(first, 1):
        current = [i]
        for j, b in enumerate(second, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def text_scores(predictions, references):
    if len(predictions) != len(references) or not references:
        raise ValueError("Thiếu, thừa hoặc không có kết quả để đánh giá")
    predictions = [normalize(text) for text in predictions]
    references = [normalize(text) for text in references]
    if any(not text for text in references):
        raise ValueError("Nhãn chuẩn không được rỗng")
    correct = sum(p == g for p, g in zip(predictions, references))
    errors = sum(edit_distance(g, p) for p, g in zip(predictions, references))
    characters = sum(map(len, references))
    return {
        "ACC": 100 * correct / len(references),
        "CER": 100 * errors / characters,
        "correct": correct,
        "samples": len(references),
        "character_errors": errors,
        "reference_characters": characters,
    }


def speed_scores(milliseconds):
    if not milliseconds or any(not math.isfinite(t) or t <= 0 for t in milliseconds):
        raise ValueError("Thời gian đo phải dương và hữu hạn")
    mean = sum(milliseconds) / len(milliseconds)
    return {"Speed": mean, "FPS": 1000 / mean}

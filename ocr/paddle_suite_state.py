"""Lưu từng số đo để tiếp tục benchmark dài khi tiến trình bị ngắt."""

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import time


@contextmanager
def model_lock(directory):
    """Khoá tự nhả khi tiến trình kết thúc, kể cả khi bị dừng đột ngột."""
    path = Path(directory) / "benchmark.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            lock = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            unlock = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            lock = lambda: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            unlock = lambda: fcntl.flock(handle, fcntl.LOCK_UN)
        try:
            lock()
        except OSError as error:
            raise RuntimeError("Model này đang được đo ở tiến trình khác") from error
        try:
            yield
        finally:
            handle.seek(0)
            unlock()


class BenchmarkState:
    def __init__(self, path, protocol, samples, repeats):
        self.path = Path(path)
        self.protocol, self.samples, self.repeats = protocol, samples, repeats
        if self.path.is_file():
            state = json.loads(self.path.read_text(encoding="utf-8"))
            if state["protocol"] != protocol:
                raise ValueError("Lượt đo dở khác model, dữ liệu, code hoặc điều kiện đo")
            self.predictions, self.times = state["predictions"], state["times_ms"]
        else:
            self.predictions = [None] * samples
            self.times = [[] for _ in range(samples)]
        if len(self.predictions) != samples or len(self.times) != samples:
            raise ValueError("Số ảnh trong tiến độ không khớp tập test")
        for prediction, times in zip(self.predictions, self.times):
            if len(times) > repeats or any(not math.isfinite(t) or t <= 0 for t in times):
                raise ValueError("Tiến độ chứa số đo không hợp lệ")
            if bool(times) != (prediction is not None) or (prediction is not None and not isinstance(prediction, str)):
                raise ValueError("Tiến độ thiếu chuỗi dự đoán tương ứng số đo")
        counts = [len(t) for t in self.times]
        if counts and max(counts) - min(counts) > 1:
            raise ValueError("Thứ tự lượt đo trong tiến độ không hợp lệ")

    @property
    def completed(self):
        return sum(map(len, self.times))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pending = self.path.with_suffix(".tmp")
        with pending.open("w", encoding="utf-8") as stream:
            json.dump({"protocol": self.protocol, "predictions": self.predictions, "times_ms": self.times}, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        # Windows có thể đang quét file vừa ghi; thử lại thao tác đổi tên.
        for attempt in range(40):
            try:
                pending.replace(self.path)
                break
            except PermissionError:
                if attempt == 39:
                    raise
                time.sleep(0.05)

    def record(self, index, repeat, prediction, milliseconds):
        if len(self.times[index]) != repeat:
            raise ValueError("Số đo bị lặp hoặc sai lượt")
        if not math.isfinite(milliseconds) or milliseconds <= 0:
            raise ValueError("Thời gian cần dương và hữu hạn")
        if self.predictions[index] is not None and prediction != self.predictions[index]:
            raise ValueError("Dự đoán không ổn định giữa các lượt")
        self.predictions[index] = prediction
        self.times[index].append(milliseconds)
        self.save()

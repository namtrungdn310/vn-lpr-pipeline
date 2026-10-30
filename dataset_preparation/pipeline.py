"""Tái tạo REC, kiểm tra v2 và đóng gói dataset."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

from PIL import Image

from .dataset import Dataset, SPLITS, read_samples
from .geometry import encoded_jpeg, intersection_area, rectify_crop, validate_quad
from .io import checked_path, file_hashes, fingerprint, sha256_file, write_bytes

CODE_DIRECTORIES = ("dataset_preparation", "ocr", "configs", "notebooks", "requirements", "tests", "third_party")


def load_baseline(workspace: Path) -> dict:
    workspace = Path(workspace).resolve(strict=True)
    baseline = json.loads((workspace / "metadata/baseline.json").read_text(encoding="utf-8"))
    if baseline["schema_version"] != 1 or baseline["dataset_name"] != "vn-plate-dataset_v2":
        raise ValueError("Metadata không đúng phiên bản v2")
    if fingerprint(baseline["files"]) != baseline["content_fingerprint"]:
        raise ValueError("Danh sách checksum không khớp fingerprint")
    if sha256_file(workspace / "metadata/samples.jsonl") != baseline["samples_sha256"]:
        raise ValueError("Recipe hoặc thông tin duyệt đã thay đổi")
    return baseline


def render_rec(workspace: Path, sample: dict, *, dataset_root=None) -> bytes:
    """Tái tạo một crop từ đúng nguồn và quad đã duyệt."""
    roots = {"dataset": "vn-plate-dataset_v2", "sources": "sources"}
    if sample["source_area"] not in roots:
        raise ValueError("Loại nguồn REC không hợp lệ")
    source_root = Path(workspace) / roots[sample["source_area"]]
    if sample["source_area"] == "dataset" and dataset_root is not None:
        source_root = Path(dataset_root)
    source = checked_path(source_root, sample["source_image"])
    if sha256_file(source) != sample["source_sha256"]:
        raise ValueError(f"Nguồn REC đã thay đổi: {source}")
    with Image.open(source) as image:
        crop = rectify_crop(
            image,
            sample["crop_points"],
            preserve_text_order=sample["preserve_text_order"],
        )
    return encoded_jpeg(crop)


def check_metadata(baseline: dict, dataset: Dataset, samples: list) -> None:
    """Mỗi dòng DET có đúng một REC, cùng split và cùng chuỗi nhãn."""
    objects = {
        (path, index): obj for path, items in dataset.det.items() for index, obj in enumerate(items)
    }
    det_seen, rec_seen, used_sources = set(), set(), set()
    for sample in samples:
        index = sample["object_index"]
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("Chỉ số object DET không hợp lệ")
        key, rec = (sample["det_image"], index), sample["rec_image"]
        if key in det_seen or rec in rec_seen or key not in objects or rec not in dataset.rec:
            raise ValueError(f"Liên kết DET–REC lặp hoặc thiếu: {rec}")
        det_seen.add(key)
        rec_seen.add(rec)
        obj = objects[key]
        if obj["transcription"] != sample["text"] or dataset.rec[rec] != sample["text"]:
            raise ValueError(f"Nhãn DET–REC không khớp: {rec}")
        if obj["points"] != sample["det_points"]:
            raise ValueError(f"Quad DET khác bản đã duyệt: {sample['det_image']}")
        split = rec.split("/")[0]
        if sample["det_image"].split("/")[0] != split:
            raise ValueError("Liên kết DET–REC vượt split")
        if not isinstance(sample["preserve_text_order"], bool):
            raise ValueError("Cờ trục chữ sai kiểu")
        if sample["source_area"] == "dataset":
            if sample["source_image"] != sample["det_image"]:
                raise ValueError("REC phải dùng đúng ảnh DET đã liên kết")
            expected = baseline["files"].get(sample["source_image"])
        elif sample["source_area"] == "sources":
            if not sample["source_image"].startswith(f"rec/{split}/"):
                raise ValueError("Ảnh REC nguồn vượt split")
            used_sources.add(sample["source_image"])
            expected = baseline["sources"].get(sample["source_image"])
        else:
            raise ValueError("Loại nguồn REC không hợp lệ")
        if expected != sample["source_sha256"]:
            raise ValueError("Checksum nguồn REC khác baseline")
    if det_seen != set(objects) or rec_seen != set(dataset.rec):
        raise ValueError("Recipe không phủ hết dữ liệu")
    if used_sources != set(baseline["sources"]):
        raise ValueError("Thiếu hoặc thừa ảnh REC nguồn")


def inspect_dataset(workspace: Path, baseline: dict):
    """Kiểm tra file, giải mã ảnh, hình học và trùng giữa các split."""
    root = checked_path(workspace, baseline["dataset_name"])
    hashes = file_hashes(root)
    if hashes != baseline["files"]:
        differences = sorted(
            name
            for name in set(hashes) | set(baseline["files"])
            if hashes.get(name) != baseline["files"].get(name)
        )
        raise ValueError(f"Dữ liệu khác bản đã duyệt: {differences[:5]}")
    directories = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir()
    )
    if directories != baseline["directories"]:
        raise ValueError("Cấu trúc thư mục v2 thay đổi")
    if file_hashes(workspace / "sources") != baseline["sources"]:
        raise ValueError("Ảnh REC nguồn bị thiếu, thừa hoặc thay đổi")
    dataset = Dataset.read(root)
    expected_images = set(dataset.det) | set(dataset.rec)
    actual_images = {name for name in hashes if name.lower().endswith(".jpg")}
    if actual_images != expected_images:
        raise ValueError("Ảnh và nhãn tham chiếu không khớp")
    sizes, pixel_groups = {}, defaultdict(list)
    for name in sorted(expected_images):
        with Image.open(checked_path(root, name)) as image:
            image.load()
            if image.format != "JPEG":
                raise ValueError(f"Ảnh không phải JPEG: {name}")
            rgb = image.convert("RGB")
            sizes[name] = rgb.size
            digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        pixel_groups[digest].append(name)
    for names in pixel_groups.values():
        if len({name.split("/")[0] for name in names}) > 1:
            raise ValueError(f"Ảnh trùng pixel giữa các split: {names}")
    plates = defaultdict(set)
    for name, objects in dataset.det.items():
        for obj in objects:
            validate_quad(obj["points"], *sizes[name])
        for index, first in enumerate(objects):
            for second in objects[index + 1 :]:
                if intersection_area(first["points"], second["points"]) > 0.1:
                    raise ValueError(f"Hai dòng chữ giao nhau: {name}")
        plates["".join(obj["transcription"] for obj in objects)].add(name.split("/")[0])
    if any(len(splits) > 1 for splits in plates.values()):
        raise ValueError("Chuỗi biển đầy đủ trùng giữa các split")
    samples = read_samples(workspace)
    check_metadata(baseline, dataset, samples)
    return dataset, samples


def reproduce_rec(
    workspace: Path, samples: list, baseline: dict, destination: Path | None = None
) -> int:
    """Đối chiếu JPEG từng byte; chỉ ghi khi có thư mục đích riêng."""
    workspace = Path(workspace).resolve(strict=True)
    if destination is not None:
        destination = Path(destination).resolve()
        protected = [
            workspace / name for name in ("vn-plate-dataset_v2", "sources", "metadata", *CODE_DIRECTORIES)
        ]
        if any(
            destination.is_relative_to(path) or path.is_relative_to(destination)
            for path in protected
        ):
            raise ValueError("Thư mục tái tạo không được chồng lên dữ liệu hoặc code")
        if destination.exists() and any(destination.iterdir()):
            raise ValueError("Thư mục tái tạo phải rỗng")
    for sample in samples:
        data = render_rec(workspace, sample)
        name = sample["rec_image"]
        if hashlib.sha256(data).hexdigest() != baseline["files"][name]:
            raise ValueError(f"Crop tái tạo không khớp JPEG đã duyệt: {name}")
        if destination is not None:
            write_bytes(checked_path(destination, name), data)
    return len(samples)


def validate(workspace: Path) -> dict:
    workspace = Path(workspace).resolve(strict=True)
    baseline = load_baseline(workspace)
    dataset, samples = inspect_dataset(workspace, baseline)
    reproduced = reproduce_rec(workspace, samples, baseline)
    counts = Counter(
        (path.split("/")[0], path.split("/")[1]) for path in [*dataset.det, *dataset.rec]
    )
    return {
        "technical_ready": True,
        "blocking_findings": [],
        "dataset": baseline["dataset_name"],
        "content_fingerprint": baseline["content_fingerprint"],
        "files": len(baseline["files"]),
        "images_decoded": len(dataset.det) + len(dataset.rec),
        "det_images": len(dataset.det),
        "det_objects": sum(len(objects) for objects in dataset.det.values()),
        "rec_images": len(dataset.rec),
        "rec_reproduced_exactly": reproduced,
        "native_rec_sources": len(baseline["sources"]),
        "split_counts": {
            split: {branch: counts[split, branch] for branch in ("det", "rec")} for split in SPLITS
        },
        "cross_row_polygon_overlaps": 0,
        "exact_cross_split_pixel_groups": 0,
        "full_plate_cross_split_groups": 0,
        "inherited_links_without_pixel_proof": sum(
            not sample["parent_pixel_supported"] for sample in samples
        ),
        "benchmark_leakage_free_certified": False,
        "model_trained": False,
    }


def build_v2(workspace: Path, destination: Path, *, input_root=None, reviewed_boxes=None) -> dict:
    """Dựng cả v2 từ ảnh nguồn và quyết định duyệt, không đọc crop REC thành phẩm."""
    workspace = Path(workspace).resolve(strict=True)
    baseline = load_baseline(workspace)
    input_root = Path(input_root or workspace / baseline["dataset_name"]).resolve(strict=True)
    destination = Path(destination).resolve()
    protected = [input_root, *(workspace / name for name in ("sources", "metadata", *CODE_DIRECTORIES))]
    if any(
        destination.is_relative_to(path) or path.is_relative_to(destination) for path in protected
    ):
        raise ValueError("Đích xuất chồng lên nguồn hoặc code")
    if destination.exists():
        raise ValueError("Đích xuất phải chưa tồn tại")
    samples = read_samples(workspace)
    if reviewed_boxes is not None:
        approved = {(row["det_image"], row["object_index"]): row for row in reviewed_boxes}
        if len(approved) != len(samples) or len(reviewed_boxes) != len(samples):
            raise ValueError("Thiếu hoặc lặp box đã duyệt")
        for sample in samples:
            row = approved.get((sample["det_image"], sample["object_index"]))
            if (
                row is None
                or row["points"] != sample["det_points"]
                or row["text"] != sample["text"]
            ):
                raise ValueError(
                    "Quyết định mới khác bản v2 đã chốt; cần duyệt và cập nhật metadata riêng"
                )
    groups = defaultdict(list)
    rec_seen = set()
    for sample in samples:
        key = (sample["det_image"], sample["object_index"])
        if key in groups or sample["rec_image"] in rec_seen:
            raise ValueError("Quyết định duyệt bị lặp")
        groups[key] = sample
        rec_seen.add(sample["rec_image"])
        if sample["det_image"].split("/")[0] != sample["rec_image"].split("/")[0]:
            raise ValueError("Quyết định duyệt vượt split")
    expected_rec = {name for name in baseline["files"] if "/rec/images/" in name}
    if rec_seen != expected_rec:
        raise ValueError("Thiếu quyết định duyệt REC")
    labels = {}
    det_images = defaultdict(list)
    for sample in samples:
        det_images[sample["det_image"]].append(sample)
    for name, rows in det_images.items():
        rows.sort(key=lambda row: row["object_index"])
        if [row["object_index"] for row in rows] != list(range(len(rows))):
            raise ValueError("Chỉ số dòng DET không liên tục")
        source = checked_path(input_root, name)
        if sha256_file(source) != baseline["files"][name]:
            raise ValueError(f"Ảnh DET nguồn đã thay đổi: {name}")
        with Image.open(source) as image:
            for row in rows:
                validate_quad(row["det_points"], image.width, image.height)
    for split in SPLITS:
        det_lines = []
        for name, rows in sorted(det_images.items()):
            if name.startswith(split + "/"):
                objects = [
                    {"transcription": row["text"], "points": row["det_points"]} for row in rows
                ]
                det_lines.append(
                    name.split("/det/", 1)[1] + "\t" + json.dumps(objects, ensure_ascii=False)
                )
        rec_lines = [
            row["rec_image"].split("/rec/", 1)[1] + "\t" + row["text"]
            for row in sorted(samples, key=lambda row: row["rec_image"])
            if row["rec_image"].startswith(split + "/")
        ]
        labels[f"{split}/det/det_labels.txt"] = ("\n".join(det_lines) + "\n").encode("utf-8")
        labels[f"{split}/rec/rec_labels.txt"] = ("\n".join(rec_lines) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v2-build-", dir=destination.parent) as temporary:
        staged = Path(temporary) / baseline["dataset_name"]
        staged.mkdir()
        for directory in baseline["directories"]:
            checked_path(staged, directory).mkdir(parents=True, exist_ok=True)
        for name, data in labels.items():
            write_bytes(checked_path(staged, name), data)
        for name, digest in baseline["files"].items():
            if name in labels or name in rec_seen:
                continue
            source = checked_path(input_root, name)
            if sha256_file(source) != digest:
                raise ValueError(f"File nguồn đã thay đổi: {name}")
            shutil.copyfile(source, checked_path(staged, name))
        for sample in samples:
            data = render_rec(workspace, sample, dataset_root=input_root)
            write_bytes(checked_path(staged, sample["rec_image"]), data)
        actual = file_hashes(staged)
        if actual != baseline["files"]:
            changed = [
                name
                for name in set(actual) | set(baseline["files"])
                if actual.get(name) != baseline["files"].get(name)
            ]
            raise ValueError(f"V2 dựng lại khác bản đã duyệt: {sorted(changed)[:5]}")
        check_metadata(baseline, Dataset.read(staged), samples)
        os.replace(staged, destination)
    return {
        "files": len(baseline["files"]),
        "rec_rebuilt": len(samples),
        "content_fingerprint": baseline["content_fingerprint"],
        "matches_approved_v2": True,
    }


def verify_archive(workspace: Path, archive: Path) -> dict:
    baseline = load_baseline(workspace)
    expected = {
        f"{baseline['dataset_name']}/{name}": digest for name, digest in baseline["files"].items()
    }
    allowed_directories = {f"{baseline['dataset_name']}/"} | {
        f"{baseline['dataset_name']}/{name}/" for name in baseline["directories"]
    }
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        files = {entry.filename for entry in bundle.infolist() if not entry.is_dir()}
        directories = set(names) - files
        if (
            len(names) != len(set(names))
            or files != set(expected)
            or not directories <= allowed_directories
        ):
            raise ValueError("ZIP thiếu, thừa hoặc lặp file")
        for name, digest in expected.items():
            if hashlib.sha256(bundle.read(name)).hexdigest() != digest:
                raise ValueError(f"ZIP khác dataset đã duyệt: {name}")
    return {"files": len(expected), "sha256": sha256_file(archive)}


def package_dataset(workspace: Path) -> dict:
    """Chỉ thay ZIP sau khi kiểm tra toàn bộ dữ liệu và nội dung ZIP mới."""
    workspace = Path(workspace).resolve(strict=True)
    validate(workspace)
    baseline = load_baseline(workspace)
    dataset = workspace / baseline["dataset_name"]
    archive = workspace / f"{baseline['dataset_name']}.zip"
    descriptor, temporary = tempfile.mkstemp(suffix=".zip", dir=workspace)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for relative in baseline["files"]:
                bundle.write(dataset / relative, f"{baseline['dataset_name']}/{relative}")
        result = verify_archive(workspace, temporary)
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "package"), nargs="?", default="validate")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    args = parser.parse_args()
    action = validate if args.command == "validate" else package_dataset
    print(json.dumps(action(args.workspace), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

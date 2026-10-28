import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from PIL import Image, ImageDraw

from src.dataset import Dataset, SPLITS
from src.geometry import encoded_jpeg, rectify_crop
from src.io import checked_path, file_hashes, fingerprint, sha256_file, write_json
from src.pipeline import build_v2, package_dataset, reproduce_rec, validate, verify_archive
from src.rec_preprocessing import iter_rec_samples


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "vn-plate-dataset_v2"
        (self.root / "metadata").mkdir()
        (self.root / "sources").mkdir()
        samples = []
        for index, split in enumerate(SPLITS):
            det_dir = self.dataset / split / "det"
            rec_dir = self.dataset / split / "rec"
            (det_dir / "images").mkdir(parents=True)
            (rec_dir / "images").mkdir(parents=True)
            source = Image.new("RGB", (80, 40), (230 - index * 30, 220, 210))
            ImageDraw.Draw(source).rectangle((8 + index, 8, 68, 30), fill=(20, 30, 40))
            source.save(det_dir / "images/a.jpg", quality=95, subsampling=0)
            points = [[6, 6], [72, 6], [72, 34], [6, 34]]
            text = f"0001{index}"
            with Image.open(det_dir / "images/a.jpg") as image:
                data = encoded_jpeg(rectify_crop(image, points))
            (rec_dir / "images/a.jpg").write_bytes(data)
            (det_dir / "det_labels.txt").write_text(
                "images/a.jpg\t" + json.dumps([{"transcription": text, "points": points}]) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            (rec_dir / "rec_labels.txt").write_text(
                f"images/a.jpg\t{text}\n", encoding="utf-8", newline="\n"
            )
            samples.append(
                {
                    "det_image": f"{split}/det/images/a.jpg",
                    "object_index": 0,
                    "text": text,
                    "det_points": points,
                    "rec_image": f"{split}/rec/images/a.jpg",
                    "source_area": "dataset",
                    "source_image": f"{split}/det/images/a.jpg",
                    "source_sha256": sha256_file(det_dir / "images/a.jpg"),
                    "crop_points": [point[:] for point in points],
                    "preserve_text_order": False,
                    "parent_pixel_supported": True,
                }
            )
        self.samples = samples
        self.seal()

    def seal(self):
        path = self.root / "metadata/samples.jsonl"
        path.write_text(
            "".join(json.dumps(sample) + "\n" for sample in self.samples), encoding="utf-8"
        )
        files = file_hashes(self.dataset)
        self.baseline = {
            "schema_version": 1,
            "dataset_name": "vn-plate-dataset_v2",
            "files": files,
            "content_fingerprint": fingerprint(files),
            "directories": sorted(
                path.relative_to(self.dataset).as_posix()
                for path in self.dataset.rglob("*")
                if path.is_dir()
            ),
            "sources": file_hashes(self.root / "sources"),
            "samples_sha256": sha256_file(path),
        }
        write_json(self.root / "metadata/baseline.json", self.baseline)

    def test_validate_reproduce_and_package_without_v1(self):
        result = validate(self.root)
        self.assertTrue(result["technical_ready"])
        self.assertEqual(result["rec_reproduced_exactly"], 3)
        self.assertEqual(next(iter_rec_samples(self.dataset))[1], "00010")
        destination = self.root / "rebuilt"
        self.assertEqual(reproduce_rec(self.root, self.samples, self.baseline, destination), 3)
        for sample in self.samples:
            self.assertEqual(
                sha256_file(destination / sample["rec_image"]),
                self.baseline["files"][sample["rec_image"]],
            )
        archive = package_dataset(self.root)
        self.assertEqual(archive["files"], 12)
        self.assertEqual(verify_archive(self.root, self.root / "vn-plate-dataset_v2.zip"), archive)
        self.assertFalse((self.root / "reports").exists())

    def test_native_rec_source_is_reproducible_and_required(self):
        sample = self.samples[0]
        source = self.root / "sources/rec/train/a.jpg"
        source.parent.mkdir(parents=True)
        source.write_bytes((self.dataset / sample["det_image"]).read_bytes())
        sample.update(source_area="sources", source_image="rec/train/a.jpg")
        self.seal()
        self.assertTrue(validate(self.root)["technical_ready"])
        source.unlink()
        with self.assertRaisesRegex(ValueError, "nguồn"):
            validate(self.root)
        self.assertFalse((self.root / "reports").exists())

    def test_tampered_recipe_and_changed_dataset_are_blocked(self):
        with (self.root / "metadata/samples.jsonl").open("a") as stream:
            stream.write("{}\n")
        with self.assertRaisesRegex(ValueError, "Recipe"):
            validate(self.root)
        self.seal()
        (self.dataset / "train/rec/images/a.jpg").write_bytes(b"invalid")
        with self.assertRaisesRegex(ValueError, "Dữ liệu"):
            validate(self.root)

    def test_duplicate_mapping_and_cross_split_source_are_rejected(self):
        self.samples[1]["rec_image"] = self.samples[0]["rec_image"]
        self.seal()
        with self.assertRaisesRegex(ValueError, "lặp"):
            validate(self.root)
        self.samples[1]["rec_image"] = "val/rec/images/a.jpg"
        self.samples[1]["source_image"] = "train/det/images/a.jpg"
        self.seal()
        with self.assertRaisesRegex(ValueError, "đúng ảnh DET"):
            validate(self.root)

    def test_wrong_crop_and_unlabelled_image_are_rejected(self):
        self.samples[0]["crop_points"][0][0] = 5
        self.seal()
        with self.assertRaisesRegex(ValueError, "Crop tái tạo"):
            validate(self.root)
        self.samples[0]["crop_points"][0][0] = 6
        extra = self.dataset / "train/rec/images/extra.jpg"
        extra.write_bytes((self.dataset / "train/rec/images/a.jpg").read_bytes())
        self.seal()
        with self.assertRaisesRegex(ValueError, "tham chiếu"):
            validate(self.root)

    def test_path_escape_and_overwriting_current_dataset_are_rejected(self):
        for relative in ("../a.jpg", "C:/outside.jpg", "/outside.jpg", "images/../a.jpg"):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                checked_path(self.dataset, relative)
        label = self.dataset / "train/rec/rec_labels.txt"
        label.write_text("../det/images/a.jpg\t00010\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            Dataset.read(self.dataset)
        with self.assertRaises(ValueError):
            reproduce_rec(self.root, self.samples, self.baseline, self.dataset)

    def test_cross_split_pixel_duplicates_are_rejected(self):
        (self.dataset / "val/rec/images/a.jpg").write_bytes(
            (self.dataset / "train/rec/images/a.jpg").read_bytes()
        )
        self.seal()
        with self.assertRaisesRegex(ValueError, "trùng pixel"):
            validate(self.root)

    def test_archive_with_extra_member_is_rejected(self):
        archive = self.root / "duplicate.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            for path in self.baseline["files"]:
                name = f"vn-plate-dataset_v2/{path}"
                bundle.write(self.dataset / path, name)
        with zipfile.ZipFile(archive, "a") as bundle:
            bundle.writestr("extra.txt", b"extra")
        with self.assertRaisesRegex(ValueError, "ZIP"):
            verify_archive(self.root, archive)

    def test_archive_may_include_valid_directory_entries(self):
        archive = self.root / "directories.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("vn-plate-dataset_v2/", b"")
            for path in self.baseline["directories"]:
                bundle.writestr(f"vn-plate-dataset_v2/{path}/", b"")
            for path in self.baseline["files"]:
                bundle.write(self.dataset / path, f"vn-plate-dataset_v2/{path}")
        self.assertEqual(verify_archive(self.root, archive)["files"], 12)

    def test_build_whole_dataset_without_reading_finished_rec_or_labels(self):
        source = self.root / "raw"
        for sample in self.samples:
            path = source / sample["det_image"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((self.dataset / sample["det_image"]).read_bytes())
        destination = self.root / "rebuilt-v2"
        summary = build_v2(self.root, destination, input_root=source)
        self.assertTrue(summary["matches_approved_v2"])
        self.assertEqual(file_hashes(destination), self.baseline["files"])
        self.assertFalse((self.root / "reports").exists())
        with self.assertRaises(ValueError):
            build_v2(self.root, self.dataset, input_root=source)

    def test_build_rejects_unreviewed_changed_geometry_and_leaves_no_output(self):
        reviewed = [
            {
                "det_image": row["det_image"],
                "object_index": row["object_index"],
                "points": row["det_points"],
                "text": row["text"],
            }
            for row in self.samples
        ]
        reviewed[0] = {**reviewed[0], "points": [[0, 0], [75, 0], [75, 35], [0, 35]]}
        destination = self.root / "new-v2"
        with self.assertRaisesRegex(ValueError, "khác bản v2"):
            build_v2(self.root, destination, reviewed_boxes=reviewed)
        self.assertFalse(destination.exists())

    def test_build_checks_source_hash_and_does_not_leave_partial_dataset(self):
        (self.dataset / self.samples[0]["det_image"]).write_bytes(b"invalid")
        destination = self.root / "new-v2"
        with self.assertRaisesRegex(ValueError, "DET nguồn"):
            build_v2(self.root, destination)
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()

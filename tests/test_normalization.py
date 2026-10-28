from pathlib import Path
import json
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image, ImageDraw

from src.dataset import Dataset
from src.line_envelopes import propose_line
from src.normalization import audit_source, normalize_text, propose_boxes, read_source, review_boxes
from src.text_bounds import glyph_quad


class NormalizationTests(unittest.TestCase):
    def scene(self, missing_last=False):
        image = Image.new("RGB", (90, 115), (220, 220, 220))
        draw = ImageDraw.Draw(image)
        for x in (15, 30, 45, 60):
            draw.rectangle((x, 12, x + 9, 49), fill=(35, 35, 35))
        ink = []
        for index, x in enumerate((15, 30, 45, 60)[: 3 if missing_last else 4]):
            top = 60 + index
            draw.rectangle((x, top, x + 10, 98), fill=(35, 35, 35))
            ink.extend((xx, yy) for yy in range(top, 99) for xx in range(x, x + 11))
        return image, ink

    def test_native_algorithm_fixes_protruding_corner_and_keeps_all_ink(self):
        image, ink = self.scene()
        result = propose_line(
            np.asarray(image),
            [[9, 48], [73, 64], [75, 104], [15, 99]],
            [[[12, 10], [74, 12], [74, 52], [12, 52]]],
            "7976",
        )
        self.assertEqual(result["status"], "supported")
        quad = np.asarray(result["points"], np.float32)
        self.assertGreater(quad[0, 1], 56)
        self.assertTrue(
            all(cv2.pointPolygonTest(quad, (float(x), float(y)), True) >= -0.5 for x, y in ink)
        )

    def test_missing_character_does_not_count_as_complete_support(self):
        image, _ = self.scene(missing_last=True)
        result = propose_line(
            np.asarray(image),
            [[9, 48], [73, 64], [75, 104], [15, 99]],
            [[[12, 10], [74, 12], [74, 52], [12, 52]]],
            "7976",
        )
        self.assertEqual(result["status"], "uncertain")

    def test_glyph_edges_require_foreground_and_cover_the_line(self):
        empty = np.full((50, 90, 3), 220, dtype=np.uint8)
        self.assertEqual(glyph_quad(empty, np.ones((50, 90)))["status"], "needs_visual_bounds")
        image, _ = self.scene()
        rgb = np.asarray(image)[55:105]
        result = glyph_quad(rgb, np.ones(rgb.shape[:2]))
        self.assertEqual(result["status"], "proposed")
        self.assertLess(result["mask_bounds"][0], 16)
        self.assertGreater(result["mask_bounds"][2], 69)

    def test_normalize_text_keeps_leading_zeros_without_guessing_digits(self):
        self.assertEqual(normalize_text(" 51f-882.70 "), "51F88270")
        self.assertEqual(normalize_text("08.26"), "0826")
        self.assertEqual(normalize_text("5OS2"), "5OS2")

    def test_proposals_need_explicit_review_and_leave_input_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "train/det/images/a.jpg"
            (root / name).parent.mkdir(parents=True)
            image, _ = self.scene()
            image.save(root / name)
            points = [[12, 10], [74, 12], [74, 52], [12, 52]]
            dataset = Dataset(root, {name: [{"transcription": "51L7", "points": points}]}, {})
            proposals = propose_boxes(dataset)
            self.assertTrue(proposals[0]["review_required"])
            with self.assertRaises(ValueError):
                review_boxes(dataset, proposals, {})
            decisions = {(name, 0): {"action": "keep", "reason": "visible strokes checked"}}
            approved = review_boxes(dataset, proposals, decisions)
            self.assertEqual(approved[0]["points"], points)
            self.assertEqual(dataset.det[name][0]["points"], points)
            decisions[(name, 0)] = {
                "action": "manual",
                "reason": "native check",
                "points": [[0, 0]] * 4,
            }
            with self.assertRaises(ValueError):
                review_boxes(dataset, proposals, decisions)

    def test_review_rejects_overlap_between_two_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image, _ = self.scene()
            image.save(root / "plate.jpg")
            quad = [[12, 10], [74, 12], [74, 52], [12, 52]]
            rows = [
                {
                    "det_image": "plate.jpg",
                    "object_index": index,
                    "text": "1234",
                    "input_points": quad,
                    "proposal_points": quad,
                }
                for index in range(2)
            ]
            decisions = {
                ("plate.jpg", index): {"action": "keep", "reason": "checked"} for index in range(2)
            }
            with self.assertRaisesRegex(ValueError, "giao nhau"):
                review_boxes(Dataset(root, {}, {}), rows, decisions)

    def test_quality_thresholds_only_use_train_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = [f"{split}/rec/images/a.jpg" for split in ("train", "val", "test")]
            for name in names:
                (root / name).parent.mkdir(parents=True)
                image = Image.new("RGB", (80, 40), "white")
                ImageDraw.Draw(image).rectangle((10, 10, 60, 30), fill="black")
                image.save(root / name)
            dataset = Dataset(root, {}, {name: "00012" for name in names})
            before = audit_source(dataset)
            Image.new("RGB", (80, 40), "black").save(root / names[1])
            after = audit_source(dataset)
            self.assertEqual(before["train_thresholds"], after["train_thresholds"])
            self.assertTrue(before["exact_cross_split_pixels"])

    def test_read_source_keeps_raw_labels_and_only_removes_reviewed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            (root / "metadata").mkdir()
            removed = ["val/det/images/a.jpg", "val/rec/images/a.jpg"]
            (root / "metadata/provenance.json").write_text(
                json.dumps({"quarantined_files": removed})
            )
            for split in ("train", "val", "test"):
                for branch in ("det", "rec"):
                    folder = raw / split / branch
                    (folder / "images").mkdir(parents=True)
                    Image.new("RGB", (80, 40), "white").save(folder / "images/a.jpg")
                    value = " 51f-882.70 "
                    if branch == "det":
                        value = json.dumps(
                            [
                                {
                                    "transcription": value,
                                    "points": [[1, 1], [79, 1], [79, 39], [1, 39]],
                                }
                            ]
                        )
                    (folder / f"{branch}_labels.txt").write_text("images/a.jpg\t" + value + "\n")
            source = read_source(root, raw)
            self.assertEqual(len(source.det), 2)
            self.assertEqual(len(source.rec), 2)
            self.assertEqual(source.rec["train/rec/images/a.jpg"], " 51f-882.70 ")
            self.assertEqual(normalize_text(source.rec["train/rec/images/a.jpg"]), "51F88270")


if __name__ == "__main__":
    unittest.main()

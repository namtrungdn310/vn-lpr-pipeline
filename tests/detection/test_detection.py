"""Kiểm những lỗi làm sai nhãn, crop và kết quả toàn luồng."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import cv2
import numpy as np

from plate_detection.data import audit_dataset, check_prepared, load_config, ordered_quad, parse_label, prepare_dataset
from plate_detection.ingest import extract_zip
from plate_detection.runtime import crop_plate, decode, letterbox
from lpr.evaluate import load_manifest, output_dir, score_predictions
from lpr.pipeline import Pipeline, ocr_names
from plate_detection.evaluate import operating_scores
from plate_detection.quality import REVIEW
from plate_detection.train import compare_outputs


class DetectionTests(unittest.TestCase):
    def test_only_nano_is_enabled_and_training_parameters_are_preserved(self):
        cfg = load_config()
        self.assertEqual(cfg["variants"], {"n": 4})
        self.assertEqual(cfg["model"], "yolo26n-obb.pt")
        self.assertEqual(cfg["train"]["batch"], 4)
        self.assertEqual(cfg["train"]["device"], 0)
        self.assertEqual(cfg["train"]["epochs"], 150)
        self.assertEqual(cfg["train"]["fliplr"], 0)
        self.assertEqual(cfg["train"]["flipud"], 0)
        for variant in "smlx":
            with self.assertRaises(ValueError):
                load_config(variant=variant)

    def test_aabb_cannot_silently_become_oriented_ground_truth(self):
        with self.assertRaisesRegex(ValueError, "AABB"):
            parse_label("0 .5 .5 .3 .2", 640, 640, {0: "plate"})

    def test_closed_roboflow_polygon_becomes_enclosing_rotated_rectangle(self):
        q = np.float32([[100, 120], [310, 90], [300, 170], [110, 185]])
        closed = np.vstack((q, q[0])) / 640
        label = "0 " + " ".join(map(str, closed.flatten()))
        actual = parse_label(label, 640, 640, {0: "plate"}) * 640
        self.assertEqual(actual.shape, (4, 2))
        for point in q:
            self.assertGreaterEqual(cv2.pointPolygonTest(actual.astype(np.float32), tuple(map(float, point)), True), -0.001)
        with self.assertRaises(ValueError):
            parse_label("0 .1 .1 .8 .8 .1 .8 .8 .1", 640, 640, {0: "plate"})

    def test_polygon_near_edge_preserves_outside_rect_for_later_padding(self):
        label = "0 0 .1 .2 0 .3 .1 .1 .2 0 .1"
        q = parse_label(label, 100, 100, {0: "plate"})
        self.assertEqual(q.shape, (4, 2))

    def test_letterbox_and_end_to_end_decode_restore_original_geometry(self):
        image = np.zeros((240, 480, 3), dtype=np.uint8)
        data, ratio, pad = letterbox(image, 640)
        self.assertEqual(data.shape, (1, 3, 640, 640))
        original = np.array([[160, 120, 100, 40, .9, 0, .25]], dtype=np.float32)
        output = original.copy()
        output[:, :4] *= ratio
        output[:, :2] += pad
        boxes = decode(output[None], ratio, pad)
        center = np.array(boxes[0]["quad"]).mean(axis=0)
        np.testing.assert_allclose(center, [160, 120], atol=1e-4)
        self.assertEqual(decode(output[None], ratio, pad, confidence=.95), [])
        with self.assertRaisesRegex(ValueError, "end-to-end"):
            decode(np.zeros((1, 7, 300)), ratio, pad)

    def test_two_line_portrait_crop_is_not_turned_sideways_or_mirrored(self):
        image = np.zeros((210, 160, 3), dtype=np.uint8)
        image[20:90, 30:130] = (0, 0, 255)
        image[110:180, 30:130] = (255, 0, 0)
        q = [[30, 20], [130, 20], [130, 180], [30, 180]]
        crop, inverse = crop_plate(image, q, padding=0)
        self.assertGreater(crop.shape[0], crop.shape[1])
        self.assertGreater(int(crop[20, 50, 2]), 240)
        self.assertGreater(int(crop[-20, 50, 0]), 240)
        recovered = cv2.perspectiveTransform(np.float32([[[0, 0]]]), inverse)
        np.testing.assert_allclose(recovered[0, 0], [30, 20], atol=1e-4)

    def test_pipeline_uses_predicted_plate_not_ground_truth_and_maps_lines(self):
        q = [[20, 10], [100, 10], [100, 70], [20, 70]]
        def ocr(crop):
            h, w = crop.shape[:2]
            return {"text": "59-A1", "boxes": [[[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]]}
        pipeline = Pipeline(lambda image: [{"quad": q, "confidence": .9}], ocr, padding=0)
        prediction = pipeline(np.zeros((100, 140, 3), np.uint8))
        self.assertEqual(prediction[0]["text"], "59A1")
        np.testing.assert_allclose(prediction[0]["line_boxes"][0], q, atol=1e-4)
        self.assertEqual(Pipeline(lambda image: [], ocr)(np.zeros((20, 20, 3), np.uint8)), [])

    def test_misses_and_false_positives_are_counted_and_text_cannot_choose_matching(self):
        q = [[0, 0], [80, 0], [80, 30], [0, 30]]
        other = [[100, 0], [180, 0], [180, 30], [100, 30]]
        rows = [{"plates": [{"quad": q, "text": "1234"}]}]
        missed = score_predictions([[]], rows)
        self.assertEqual((missed["ACC"], missed["CER"], missed["FN"]), (0, 100, 1))
        wrong_place = score_predictions([[{"quad": other, "text": "1234"}]], rows)
        self.assertEqual((wrong_place["ACC"], wrong_place["CER"], wrong_place["FP"]), (0, 200, 1))
        extra = score_predictions([[{"quad": q, "text": "1234"}, {"quad": other, "text": "AA"}]], rows)
        self.assertEqual((extra["ACC"], extra["CER"], extra["FP"]), (100, 50, 1))

    def test_one_to_one_matching_does_not_reuse_a_detection(self):
        q = [[0, 0], [80, 0], [80, 30], [0, 30]]
        rows = [{"plates": [{"quad": q, "text": "1234"}, {"quad": q, "text": "1234"}]}]
        scores = score_predictions([[{"quad": q, "text": "1234"}]], rows)
        self.assertEqual((scores["ACC"], scores["CER"], scores["FN"]), (50, 50, 1))

    def test_zip_traversal_is_rejected_without_extracting(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("../../outside.txt", "bad")
                z.writestr("data.yaml", "names: [plate]")
            with patch("plate_detection.ingest.ROOT", root):
                with self.assertRaisesRegex(ValueError, "không an toàn"):
                    extract_zip(archive, root / "data/raw/source")
            self.assertFalse((root / "outside.txt").exists())

    def test_full_frame_protocol_uses_seven_adapted_ocr_models(self):
        self.assertEqual(len(ocr_names()), 7)
        self.assertEqual(len(ocr_names(include_public=False)), 7)
        self.assertNotIn("VietLPR_v2", ocr_names())

    def test_detector_metrics_count_misses_and_duplicate_boxes(self):
        q = [[10, 10], [90, 10], [90, 40], [10, 40]]
        scores = operating_scores([[{"quad": q}, {"quad": q}]], [[q]])
        self.assertEqual((scores["Precision"], scores["Recall"], scores["TP"], scores["FP"]), (50, 100, 1, 1))
        self.assertEqual(operating_scores([[]], [[q]])["FN"], 1)

    def test_export_parity_matches_geometry_when_topk_ties_change_order(self):
        native = np.float32([[[100, 100, 40, 20, .9, 0, .2], [300, 300, 50, 25, .9, 0, .3]]])
        reordered = native[:, ::-1].copy()
        self.assertEqual(compare_outputs(native, reordered, .25)[0], 2)
        duplicate_geometry = native.copy()
        duplicate_geometry[0, 1, :4] = native[0, 0, :4]
        duplicate_geometry[0, 1, 4] = .4
        self.assertEqual(compare_outputs(duplicate_geometry, duplicate_geometry[:, ::-1].copy(), .25)[0], 2)
        reordered[0, 0, 0] += 20
        with self.assertRaises(AssertionError):
            compare_outputs(native, reordered, .25)

    def test_reviewed_groups_are_disjoint_and_keep_only_one_split(self):
        review = json.loads(REVIEW.read_text(encoding="utf-8"))
        seen = set()
        self.assertEqual(len(review["groups"]), 10)
        for group in review["groups"]:
            members = {(m["split"], m["family"]) for m in group["members"]}
            self.assertFalse(seen & members)
            self.assertIn(group["keep_split"], {s for s, _ in members})
            self.assertTrue(all(len(h) == 64 for h in group["source_images"].values()))
            seen |= members

    def test_preparation_preserves_pixels_pads_border_and_checks_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw"
            source.mkdir()
            (source / "data.yaml").write_text("names: [plate]\n")
            for i, split in enumerate(("train", "val", "test")):
                folder = source / split
                (folder / "images").mkdir(parents=True)
                (folder / "labels").mkdir()
                image = np.full((100, 100, 3), 40 + i * 60, np.uint8)
                cv2.imencode(".png", image)[1].tofile(str(folder / "images" / f"frame_{i}.png"))
                (folder / "labels" / f"frame_{i}.txt").write_text("0 0 .45 .13 .30 .26 .50 .08 .67 0 .45\n")
            cfg = {"raw_dataset": str(source), "dataset": str(root / "runs/plate_detection/dataset")}
            with patch("plate_detection.data.ROOT", root):
                result = prepare_dataset(cfg)
                self.assertTrue(result.is_file())
                checked = check_prepared(cfg)
                self.assertTrue(checked["ready"])
                self.assertTrue(checked["source_pixels_verified"])
                self.assertGreater(checked["padded_images"], 0)
                self.assertEqual(checked["images"], 3)
                label = next((Path(cfg["dataset"]) / "train/labels").glob("*.txt"))
                label.write_text("0 .1 .1 .3 .1 .3 .2 .1 .2\n")
                with self.assertRaises(ValueError):
                    check_prepared(cfg)


if __name__ == "__main__":
    unittest.main()

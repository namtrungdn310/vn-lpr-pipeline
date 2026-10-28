import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from experiments.config import load_config
from experiments.data import image, read_split
from experiments.evaluate import compare, validate_export
from experiments.runtime import Pipeline, decode_ctc, decode_boxes


class ProtocolTests(unittest.TestCase):
    def test_paddle_db_postprocess_runs_without_importing_paddle(self):
        cfg = load_config("paddleocr")
        probability = np.zeros((64, 64), np.float32)
        probability[8:18, 8:54] = 0.95
        probability[36:48, 8:54] = 0.95
        boxes = decode_boxes(probability, np.zeros((128, 128, 3), np.uint8), cfg)
        self.assertEqual(len(boxes), 2)
        self.assertLess(boxes[0][0][:, 1].mean(), boxes[1][0][:, 1].mean())

    def test_pipeline_batches_one_and_keeps_both_lines_in_order(self):
        cfg = load_config("vietlpr_v2")
        blank = np.zeros((1, 32, len(cfg["chars"]) + 1), np.float32)

        def sequence(text):
            data = blank.copy()
            data[:, :, 0] = 1
            for i, char in enumerate(text):
                data[0, i * 2, 0] = 0
                data[0, i * 2, cfg["chars"].index(char) + 1] = 1
            return data

        outputs = iter((sequence("16M9"), sequence("0826")))
        det = lambda data: np.zeros((1, 1, 64, 64), np.float32)
        rec = lambda data: next(outputs)
        boxes = [
            (np.float32([[5, 5], [75, 5], [75, 35], [5, 35]]), 1.0),
            (np.float32([[5, 45], [75, 45], [75, 75], [5, 75]]), 1.0),
        ]
        with patch("experiments.runtime.decode_boxes", return_value=boxes):
            result = Pipeline(cfg, det, rec)(np.zeros((80, 80, 3), np.uint8))
        self.assertEqual(result["text"], "16M90826")

    def test_ctc_collapses_repeats_but_blank_separates_identical_characters(self):
        data = np.zeros((1, 5, 3), np.float32)
        for i, index in enumerate([1, 1, 0, 1, 2]):
            data[0, i, index] = 1
        self.assertEqual(decode_ctc(data, "AB"), "AAB")

    def test_compare_requires_same_conditions_and_paddle_baseline(self):
        shared = {
            "test_signature": "same",
            "threads": 4,
            "opencv_threads": 1,
            "batch_size": 1,
            "precision": "FP32",
            "runtime": "1.20.1",
            "provider": "CPUExecutionProvider",
            "platform": "machine",
            "processor": "cpu",
            "repeats": 5,
            "warmup": 20,
            "timing_scope": "all",
            "code_signature": "same-code",
        }
        results = [
            {**shared, "Model": name, "ACC": 90.0, "CER": 1.0, "Speed": speed, "FPS": 1000 / speed}
            for name, speed in (("PaddleOCR", 60.0), ("VietLPR_v1", 30.0), ("VietLPR_v2", 15.0))
        ]
        with tempfile.TemporaryDirectory() as directory:
            rows = compare(results, Path(directory) / "comparison.csv")
            self.assertEqual([row["Speedup"] for row in rows], [1.0, 2.0, 4.0])
            results[-1]["threads"] = 1
            with self.assertRaises(ValueError):
                compare(results, Path(directory) / "invalid.csv")

    def test_smoke_weights_cannot_be_used_for_final_test(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = load_config("vietlpr_v1")
            cfg["run_dir"] = directory
            (Path(directory) / "onnx").mkdir()
            (Path(directory) / "onnx/export_check.json").write_text(
                json.dumps({"det": {"smoke_only": True}, "rec": {"smoke_only": True}})
            )
            with self.assertRaises(ValueError):
                validate_export(cfg)

    def test_split_reader_does_not_resplit_and_preserves_full_plate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "val/det"
            (base / "images").mkdir(parents=True)
            pixels = np.zeros((80, 80, 3), np.uint8)
            cv2.imwrite(str(base / "images/a.jpg"), pixels)
            objects = [
                {"transcription": "0826", "points": [[1, 45], [79, 45], [79, 79], [1, 79]]},
                {"transcription": "16M9", "points": [[1, 1], [79, 1], [79, 35], [1, 35]]},
            ]
            (base / "det_labels.txt").write_text(
                "images/a.jpg\t" + json.dumps(objects), encoding="utf-8"
            )
            rows = read_split(root, "val", "det")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["text"], "16M90826")


if __name__ == "__main__":
    unittest.main()

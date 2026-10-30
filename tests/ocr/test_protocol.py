import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from ocr.config import load_config
from ocr.data import read_split
from ocr.evaluate import validate_export
from ocr.runtime import Pipeline, decode_ctc


class ProtocolTests(unittest.TestCase):
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

        def rec(data):
            self.assertEqual(len(data), 2)
            return np.concatenate([next(outputs), next(outputs)])

        boxes = [
            (np.float32([[5, 5], [75, 5], [75, 35], [5, 35]]), 1.0),
            (np.float32([[5, 45], [75, 45], [75, 75], [5, 75]]), 1.0),
        ]
        with patch("ocr.runtime.SourceV2.detect", return_value=boxes):
            result = Pipeline(cfg, det, rec)(np.zeros((80, 80, 3), np.uint8))
        self.assertEqual(result["text"], "16M90826")

    def test_ctc_collapses_repeats_but_blank_separates_identical_characters(self):
        data = np.zeros((1, 5, 3), np.float32)
        for i, index in enumerate([1, 1, 0, 1, 2]):
            data[0, i, index] = 1
        self.assertEqual(decode_ctc(data, "AB"), "AAB")

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

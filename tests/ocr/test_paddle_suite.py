"""Kiểm bộ model, decoder, khoá trọng số và điều kiện so sánh."""

import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ocr.config import ROOT
from ocr import paddle_suite as suite
from ocr.paddle_suite_state import BenchmarkState, model_lock
from third_party.PaddleOCR_suite.processors import CTCLabelDecode, DetResizeForTest, SortQuadBoxes


class PaddleSuiteTests(unittest.TestCase):
    def test_resume_keeps_every_recorded_measurement_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            state = BenchmarkState(path, {"signature": "same"}, 2, 5)
            state.record(1, 0, "59C1", 25.0)
            resumed = BenchmarkState(path, {"signature": "same"}, 2, 5)
            self.assertEqual(resumed.completed, 1)
            self.assertEqual(resumed.times, [[], [25.0]])
            self.assertEqual(resumed.predictions, [None, "59C1"])
            with self.assertRaisesRegex(ValueError, "lặp"):
                resumed.record(1, 0, "59C1", 30)
            with self.assertRaisesRegex(ValueError, "khác"):
                BenchmarkState(path, {"signature": "changed"}, 2, 5)
            resumed.record(0, 0, "16M9", 20)
            resumed.record(1, 1, "59C1", 24)
            with self.assertRaisesRegex(ValueError, "ổn định"):
                resumed.record(0, 1, "16M8", 20)
            self.assertEqual(BenchmarkState(path, {"signature": "same"}, 2, 5).completed, 3)

    def test_live_model_lock_rejects_concurrent_measurement_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            with model_lock(directory):
                with self.assertRaisesRegex(RuntimeError, "tiến trình khác"):
                    with model_lock(directory):
                        self.fail("Lock must reject another writer")
            with model_lock(directory):
                pass

    def test_catalog_uses_seven_real_paired_variants_and_pinned_sources(self):
        catalog = suite.load_catalog()
        self.assertEqual(len(catalog["variants"]), 7)
        for variant in catalog["variants"]:
            for task, model in variant["models"].items():
                self.assertEqual(model["name"], variant["name"] + "_" + task)
                self.assertTrue(model["repository"].startswith("PaddlePaddle/"))
                self.assertEqual(len(model["revision"]), 40)
                self.assertIn(model["revision"], model["config"]["url"])
                info = model.get("onnx", {})
                self.assertEqual(len(info.get("sha256", model.get("archive_sha256", ""))), 64)

    def test_processors_match_reviewed_source_ast(self):
        directory = ROOT / "third_party/PaddleOCR_suite"
        manifest = json.loads((directory / "source_manifest.json").read_text(encoding="utf-8"))
        actual = {n.name: n for n in ast.parse((directory / "processors.py").read_text(encoding="utf-8")).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
        self.assertEqual(set(actual), set(manifest["definitions"]))
        for name, origin in manifest["definitions"].items():
            self.assertEqual(hashlib.sha256(ast.dump(actual[name]).encode()).hexdigest(), origin["ast_sha256"], name)

    def test_ctc_preserves_token_order_blank_repeat_and_space(self):
        decode = CTCLabelDecode(character_list=["A", "12", "B"], use_space_char=True)
        ids = [1, 1, 0, 1, 2, 4, 3, 3]
        scores = np.zeros((1, len(ids), 5), np.float32)
        for i, value in enumerate(ids):
            scores[0, i, value] = 1
        self.assertEqual(decode([scores])[0], ["AA12 B"])

    def test_det_keeps_aspect_and_does_not_upscale_small_plate(self):
        resize = DetResizeForTest(limit_side_len=960, limit_type="max")
        pixels, shapes = resize([np.zeros((64, 160, 3), np.uint8)])
        self.assertEqual(pixels[0].shape, (64, 160, 3))
        np.testing.assert_array_equal(shapes[0], [64, 160, 1, 1])
        pixels, _ = resize([np.zeros((1200, 2400, 3), np.uint8)])
        self.assertEqual(pixels[0].shape, (480, 960, 3))

    def test_sort_keeps_two_plate_lines_in_order(self):
        top = np.float32([[8, 10], [50, 10], [50, 25], [8, 25]])
        bottom = top + [0, 40]
        boxes = SortQuadBoxes()(np.array([bottom, top]))
        np.testing.assert_array_equal(boxes[0], top)

    def test_artifact_guard_catches_dictionary_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("onnx/det.onnx", "onnx/rec.onnx", "source/det/inference.yml", "source/rec/inference.yml"):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"original")
            hashes = suite.artifact_hashes(root)
            suite.check_artifacts(root, hashes)
            (root / "source/rec/inference.yml").write_bytes(b"different tokens")
            with self.assertRaisesRegex(ValueError, "dictionary"):
                suite.check_artifacts(root, hashes)

    def comparison_rows(self):
        common = {key: "same" for key in (
            "test_signature", "val_signature", "threads", "opencv_threads", "batch_size", "batch_unit",
            "rec_batching", "precision", "runtime", "provider", "platform", "processor", "warmup",
            "repeats", "timing_scope", "det_resize",
        )}
        return [{**common, "Model": name, "ACC": 80, "CER": 4, "Speed": 10 + i, "FPS": 1000 / (10 + i), "code_signature": suite.code_signature()} for i, name in enumerate(suite.MODEL_NAMES)]

    def test_speedup_uses_v4_mobile_and_missing_or_mixed_results_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = suite.load_catalog()
            catalog["run_dir"] = directory
            results = self.comparison_rows()
            table = suite.compare(catalog, results)
            self.assertEqual(table[1]["Speedup"], 1)
            self.assertAlmostEqual(table[0]["Speedup"], 11 / 10)
            self.assertEqual(len(table), 7)
            with self.assertRaisesRegex(ValueError, "bảy"):
                suite.compare(catalog, results[:-1])
            results[0]["threads"] = "different"
            with self.assertRaisesRegex(ValueError, "threads"):
                suite.compare(catalog, results)

    def test_tune_reads_only_val_and_does_not_change_models(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = suite.load_catalog()
            catalog["run_dir"] = directory
            name = suite.MODEL_NAMES[0]
            rows = [{"path": "val-only.jpg", "text": "59C1"}]

            class Pipe:
                def __init__(self, *args):
                    self.postprocess = {"thresh": .3, "unclip_ratio": 1.5}

                def detect_raw(self, pixels):
                    return None

                def recognize(self, pixels, raw):
                    return {"text": "59C1"}

            with patch.object(suite, "prepared_state"), patch.object(suite, "read_split", return_value=rows) as read, patch.object(suite, "split_signature", return_value="val-sig"), patch.object(suite, "image", return_value=np.zeros((32, 64, 3), np.uint8)), patch.object(suite, "PaddlePipeline", Pipe), patch.object(suite, "artifact_hashes", return_value={"det": "same", "rec": "same"}):
                result = suite.tune(catalog, name, search=False)
            read.assert_called_once_with(catalog["dataset"], "val", "det")
            self.assertEqual(result["ACC"], 100)
            frozen = suite.read_json(Path(directory) / name / "onnx/frozen.json")
            self.assertEqual(frozen["val_signature"], "val-sig")
            self.assertEqual(frozen["artifacts"], {"det": "same", "rec": "same"})


if __name__ == "__main__":
    unittest.main()

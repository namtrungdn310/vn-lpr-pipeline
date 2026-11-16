"""Kiểm luồng PC mới: split, FP32, model giữ nguyên và số đo gốc."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto, helper

from ocr.adaptation import benchmark as b
from ocr.adaptation import common, data, preprocessing
from ocr.adaptation import sources


class AdaptationTests(unittest.TestCase):
    def test_catalog_has_seven_models_and_only_six_are_trained(self):
        cfg = common.settings()
        self.assertEqual(
            [m["name"] for m in cfg["models"]],
            [
                "VietLPR_v1",
                "CCT_S_VN",
                "PP-OCRv4_mobile",
                "PP-OCRv6_small",
                "PP-OCRv5_mobile",
                "PP-OCRv6_tiny",
                "LPRNet_VN",
            ],
        )
        self.assertEqual(sum(m["train"] for m in cfg["models"]), 6)
        self.assertFalse(cfg["models"][0]["train"])
        self.assertEqual(cfg["precision"], "FP32")

    def test_changed_dataset_is_rejected_before_training(self):
        with patch.object(data, "read_json", return_value={"content_fingerprint": "wrong"}):
            with self.assertRaisesRegex(ValueError, "Baseline"):
                data.verify_dataset()

    def test_small_cct_image_is_retained_and_preprocessing_matches_train(self):
        pixels = np.full((14, 40, 3), 120, dtype=np.uint8)
        prepared = preprocessing.cct_image(pixels)
        actual = preprocessing.cct_input(pixels)
        import cv2

        expected = cv2.resize(cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB), (128, 64))[None].astype(
            np.float32
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, (1, 64, 128, 3))

    def test_quantized_graph_is_rejected_from_fp32_suite(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "model.onnx"
            node = helper.make_node("DynamicQuantizeLinear", ["input"], ["q", "scale", "zero"])
            graph = helper.make_graph(
                [node],
                "bad",
                [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
                [
                    helper.make_tensor_value_info("q", TensorProto.UINT8, [1]),
                    helper.make_tensor_value_info("scale", TensorProto.FLOAT, []),
                    helper.make_tensor_value_info("zero", TensorProto.UINT8, []),
                ],
            )
            onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), p)
            with self.assertRaisesRegex(ValueError, "lượng tử"):
                common.check_fp32(p)

    def test_desktop_without_battery_is_allowed_but_battery_power_is_not(self):
        state = {"ac": None, "power_scheme": "Balanced"}
        with patch.object(b, "power", return_value=state):
            self.assertEqual(b.check_power(state), state)
        with patch.object(b, "power", return_value={"ac": False, "power_scheme": "Balanced"}):
            with self.assertRaises(RuntimeError):
                b.check_power(state)

    def test_table_ranks_accuracy_then_cer_not_fps_and_uses_slowest_baseline(self):
        rows = [
            {"Model": "fast", "ACC": 90, "CER": 2, "Speed": 1, "Speed_std": 0.1, "FPS": 1000},
            {"Model": "accurate", "ACC": 95, "CER": 1, "Speed": 10, "Speed_std": 0.2, "FPS": 100},
        ]
        original = copy.deepcopy(rows)
        table = b.render(rows)
        self.assertLess(table.index("accurate"), table.index("fast"))
        self.assertIn("| 1.00 |", table)
        self.assertIn("| 10.00 |", table)
        self.assertEqual(rows, original)

    def test_corrupted_pass_times_or_ground_truth_are_rejected(self):
        context = {"cores": 4}
        power = {"ac": None}
        model = {"files": {"model.onnx": "hash"}}
        manifest = {
            "code_signature": "code",
            "cpu_protocol": context,
            "power": power,
            "models": {"A": model},
        }
        record = {
            "name": "A",
            "round": 0,
            "manifest_signature": "code",
            "cpu_protocol": context,
            "model": model,
            "before": {"power": power},
            "after": {"power": power},
            "details": [
                {"image": "test.jpg", "GT": "51A12345", "prediction": "51A12345", "time_ms": 1}
            ],
        }
        with patch.object(b, "rows", return_value=[{"relative": "test.jpg", "text": "51A12345"}]):
            b.validate_record(record, manifest, "A", 0)
            bad = copy.deepcopy(record)
            bad["details"][0]["time_ms"] = float("nan")
            with self.assertRaisesRegex(ValueError, "Thời gian"):
                b.validate_record(bad, manifest, "A", 0)
            bad = copy.deepcopy(record)
            bad["details"][0]["GT"] = "other"
            with self.assertRaisesRegex(ValueError, "test"):
                b.validate_record(bad, manifest, "A", 0)

    def test_training_module_refuses_to_retrain_frozen_vietlpr(self):
        from ocr.adaptation.suite import run

        with self.assertRaisesRegex(ValueError, "không train lại"):
            run(["VietLPR_v1"])


if __name__ == "__main__":
    unittest.main()

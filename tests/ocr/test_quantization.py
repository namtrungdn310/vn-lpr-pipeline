"""Kiểm INT4 thực, Conv giữ FP32 và calibration chỉ dùng train."""

import ast
import inspect
import random
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

from ocr import quantization
from ocr import evaluate


class QuantizationTests(unittest.TestCase):
    def test_timed_statements_are_identical_to_notebook03(self):
        """Khóa cùng phạm vi đo, gồm tạo kết quả pipeline trước khi dừng clock."""
        functions = [ast.parse(inspect.getsource(f)) for f in (evaluate.benchmark, quantization.measure_in_memory)]
        blocks = []
        for tree in functions:
            inner = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                         and isinstance(n.target, ast.Name) and n.target.id == "index")
            blocks.append([ast.dump(n, include_attributes=False) for n in inner.body])
        self.assertEqual(blocks[0], blocks[1])

    def test_ram_measurement_has_no_file_io_or_progress_writes(self):
        calls = []
        def pipeline(pixel):
            calls.append(pixel)
            return {"text": str(pixel)}

        params = {"warmup": 2, "repeats": 3}
        with patch.object(quantization, "save_json", side_effect=AssertionError("write during measurement")), \
                patch("builtins.open", side_effect=AssertionError("file IO during measurement")), \
                patch("builtins.print", side_effect=AssertionError("print during measurement")):
            predictions, times = quantization.measure_in_memory(pipeline, [0, 1, 2], "warm", params, 42)
        expected = ["warm", "warm"]
        for repeat in range(3):
            indices = list(range(3))
            random.Random(42 + repeat).shuffle(indices)
            expected.extend(indices)
        self.assertEqual(calls, expected)
        self.assertEqual(predictions, ["0", "1", "2"])
        self.assertTrue(all(len(t) == 3 and all(v > 0 for v in t) for t in times))

    def test_ram_measurement_rejects_unstable_predictions(self):
        calls = iter([{"text": "51A"}, {"text": "51A"}, {"text": "51B"}])
        with self.assertRaisesRegex(RuntimeError, "Dự đoán thay đổi"):
            quantization.measure_in_memory(lambda _: next(calls), [0], 0, {"warmup": 1, "repeats": 2}, 42)

    def test_int4_uses_packed_operator_and_preserves_conv(self):
        rng = np.random.default_rng(42)
        weights = rng.normal(0, .1, (32, 16)).astype(np.float32)
        graph = helper.make_graph([
            helper.make_node("MatMul", ["input", "weight"], ["output"], name="linear"),
            helper.make_node("Conv", ["pixels", "kernel"], ["conv_output"], name="conv")], "mixed",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 32]),
             helper.make_tensor_value_info("pixels", TensorProto.FLOAT, [1, 1, 3, 3])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16]),
             helper.make_tensor_value_info("conv_output", TensorProto.FLOAT, [1, 1, 3, 3])],
            [numpy_helper.from_array(weights, "weight"),
             numpy_helper.from_array(np.ones((1, 1, 1, 1), np.float32), "kernel")])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "source.onnx", Path(folder) / "int4.onnx"
            onnx.save(model, source)
            quantization.weight_int4(source, target, 32)
            inv = quantization.inventory(target)
            self.assertEqual(inv["ops"].get("MatMulNBits"), 1)
            self.assertEqual(inv["ops"].get("Conv"), 1)
            actual = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"]).run(None, {
                "input": np.ones((1, 32), np.float32), "pixels": np.ones((1, 1, 3, 3), np.float32)})
            self.assertTrue(all(np.isfinite(o).all() for o in actual))
            np.testing.assert_array_equal(actual[1], np.ones((1, 1, 3, 3), np.float32))

    def test_calibration_never_reads_val_or_test(self):
        class Source:
            def detect(self, pixels):
                return [(np.float32([[0, 0], [3, 0], [3, 3], [0, 3]]), 1)]

            def crop(self, pixels, quad):
                return np.zeros((32, 128, 3), np.uint8)

        class Pipe:
            source = Source()

        rows = [{"path": Path("train.jpg"), "relative": "train.jpg", "text": "51A1"}]
        with patch.object(quantization, "read_split", return_value=rows) as reader, \
                patch.object(quantization, "onnx_pipeline", return_value=Pipe()), \
                patch.object(quantization, "image", return_value=np.zeros((32, 128, 3), np.uint8)), \
                patch.object(quantization, "split_signature", return_value="train_signature"):
            data, recipe = quantization.calibration({"seed": 42, "calibration_train_images": 1,
                "calibration_rec_images": 1}, {"dataset": "dataset", "model": {"det_size": 128}})
        self.assertTrue(all(call.args[1] == "train" for call in reader.call_args_list))
        self.assertEqual(recipe["split"], "train")
        self.assertEqual(data["det"][0].shape, (1, 3, 128, 128))
        self.assertEqual(recipe["rec_samples"], 2)


if __name__ == "__main__":
    unittest.main()

"""Kiểm phép đổi Conv, INT4 thực và tính công bằng của lượt đo."""

from copy import deepcopy
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

from plate_detection import quantization as q


def small_model(path):
    rng = np.random.default_rng(42)
    point = rng.normal(0, .05, (16, 32, 1, 1)).astype(np.float32)
    spatial = rng.normal(0, .05, (4, 16, 3, 3)).astype(np.float32)
    bias = rng.normal(0, .01, (16,)).astype(np.float32)
    graph = helper.make_graph([
        helper.make_node("Conv", ["input", "point", "bias"], ["hidden"], name="point", kernel_shape=[1, 1]),
        helper.make_node("Conv", ["hidden", "spatial"], ["output"], name="spatial", kernel_shape=[3, 3])],
        "pointwise", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 32, 7, 7])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 5, 5])],
        [numpy_helper.from_array(point, "point"), numpy_helper.from_array(bias, "bias"),
         numpy_helper.from_array(spatial, "spatial")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.save(model, path)


def run(path, tensor):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"]).run(None, {"input": tensor})[0]


class QuantizationTests(unittest.TestCase):
    def test_pointwise_rewrite_preserves_float_math_bias_and_spatial_conv(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp) / "fp32.onnx", Path(temp) / "pointwise.onnx"
            small_model(source)
            original = source.read_bytes()
            recipe = q.rewrite_pointwise(source, target)
            self.assertEqual(recipe["converted_convs"], 1)
            self.assertEqual(recipe["converted_weight_elements"], 512)
            tensor = np.random.default_rng(3).normal(size=(1, 32, 7, 7)).astype(np.float32)
            np.testing.assert_allclose(run(source, tensor), run(target, tensor), rtol=1e-5, atol=1e-6)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(q.inventory(target)["ops"]["Conv"], 1)

    def test_int4_is_packed_matmul_with_unchanged_spatial_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = [Path(temp) / (p + ".onnx") for p in ("source", "rewrite", "int4")]
            small_model(paths[0])
            q.rewrite_pointwise(paths[0], paths[1])
            q.int4_model(paths[1], paths[2], 32)
            ops = q.inventory(paths[2])["ops"]
            self.assertEqual(ops["MatMulNBits"], 1)
            self.assertEqual(ops["Conv"], 1)
            kernels = [{w.name: numpy_helper.to_array(w) for w in onnx.load(p).graph.initializer} for p in paths[::2]]
            np.testing.assert_array_equal(kernels[0]["spatial"], kernels[1]["spatial"])
            data = np.random.default_rng(8).normal(size=(1, 32, 7, 7)).astype(np.float32)
            self.assertTrue(np.isfinite(run(paths[2], data)).all())
            self.assertLess(float(np.mean(np.abs(run(paths[0], data) - run(paths[2], data)))), .03)

    def test_unsupported_conv_does_not_produce_fake_int4(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp) / "source.onnx", Path(temp) / "out.onnx"
            small_model(source)
            model = onnx.load(source)
            # Conv 1x1 stride 2 không có cùng phép nhân từng vị trí.
            model.graph.node[0].attribute.append(helper.make_attribute("strides", [2, 2]))
            onnx.save(model, source)
            with self.assertRaisesRegex(ValueError, "Không có Conv"):
                q.rewrite_pointwise(source, target)
            self.assertFalse(target.exists())

    def test_calibration_reads_train_only_and_is_reproducible(self):
        rows = [{"image": str(i)} for i in range(20)]
        with patch.object(q, "split_rows", side_effect=lambda *args: deepcopy(rows)) as loader:
            a = q.calibration_rows({"seed": 42, "calibration_train_images": 5}, {})
            b = q.calibration_rows({"seed": 42, "calibration_train_images": 5}, {})
        self.assertEqual(a, b)
        self.assertEqual(len(a), 5)
        self.assertTrue(all(call.args[1] == "train" for call in loader.call_args_list))

    def test_static_int8_calibrates_all_images_and_executes(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp) / "source.onnx", Path(temp) / "int8.onnx"
            small_model(source)
            model = onnx.load(source)
            model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 3
            weight = model.graph.initializer[0]
            weight.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(weight)[:, :3].copy(), "point"))
            onnx.save(model, source)
            images = [np.full((7, 7, 3), v, np.uint8) for v in (10, 100, 240)]
            with patch.object(q, "read_image", side_effect=images) as reader:
                q.int8_model(source, target, [{"image": str(i)} for i in range(3)], 7)
            self.assertEqual(reader.call_count, 3)
            self.assertGreater(q.inventory(target)["ops"]["QuantizeLinear"], 0)
            self.assertTrue(any(w.data_type == TensorProto.INT8 for w in onnx.load(target).graph.initializer))
            data = np.full((1, 3, 7, 7), .5, np.float32)
            self.assertTrue(np.isfinite(run(target, data)).all())
            np.testing.assert_allclose(run(source, data), run(target, data), atol=.02, rtol=.1)

    def test_measurement_has_no_file_writes_and_each_image_once(self):
        calls = []
        def pipeline(item):
            calls.append(item)
            return [{"id": item}]
        with patch.object(q, "save_json", side_effect=AssertionError("IO")), \
                patch("builtins.open", side_effect=AssertionError("IO")), \
                patch("builtins.print", side_effect=AssertionError("print")):
            predictions, times = q.measure_in_memory(pipeline, [0, 1, 2], "warm", {"warmup": 2, "seed": 42}, 0)
        self.assertEqual(calls[:2], ["warm", "warm"])
        self.assertEqual(sorted(calls[2:]), [0, 1, 2])
        self.assertEqual(predictions, [[{"id": i}] for i in range(3)])
        self.assertTrue(all(t > 0 for t in times))

    def test_sd_is_across_round_means_and_tampered_labels_rejected(self):
        rows = [{"image": "test.jpg"}]
        quad = [[0, 0], [20, 0], [20, 10], [0, 10]]
        prediction = [{"quad": quad, "confidence": .9}]
        gt = [[quad]]
        scores = {"Precision": 100., "Recall": 100., "TP": 1, "FP": 0, "FN": 0}
        records = [{"name": "fp32", "round": i, "model_sha256": "model", "code_signature": "code",
                    "dataset_signature": "dataset", "scores": scores,
                    "power_before": {"ac": True}, "power_after": {"ac": True},
                    "details": [{"image": "test.jpg", "GT": gt[0], "prediction": prediction, "time_ms": t}]}
                   for i, t in enumerate((10., 20.))]
        state = {"protocol": {"dataset_signature": "dataset"}, "models": {"fp32": {"graph": {"sha256": "model", "bytes": 100}, "scope": "FP32"}}}
        with patch.object(q, "code_signature", return_value="code"), patch.object(q, "read_image"), \
                patch.object(q, "ground_truth", return_value=gt):
            result = q.aggregate(records, "fp32", state, rows, {"rounds": 2, "require_ac": True}, {"benchmark": {"match_iou": .5}})
            self.assertEqual(result["Speed"], 15.)
            self.assertEqual(result["SD"], statistics.stdev([10., 20.]))
            records[1]["details"][0]["GT"] = []
            with self.assertRaisesRegex(ValueError, "GT"):
                q.aggregate(records, "fp32", state, rows, {"rounds": 2, "require_ac": True}, {"benchmark": {"match_iou": .5}})


if __name__ == "__main__":
    unittest.main()

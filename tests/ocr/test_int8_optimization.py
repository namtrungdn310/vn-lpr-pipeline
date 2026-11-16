"""Kiểm lựa chọn bằng val, INT8 thật và dùng chung bản ghi FP32."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

from ocr import int8_optimization as opt
from ocr import int8_refinement as refine
from ocr import optimized_benchmark as bench
from ocr import yolo_int8_refinement as yref
from ocr import cpu_protocol


class OptimizationTests(unittest.TestCase):
    def test_selection_rejects_fast_but_inaccurate_ocr(self):
        cfg = opt.settings()
        baseline = {"scores": {"correct": 210, "CER": .8}}
        slow = {"scores": {"correct": 210, "CER": .85}, "timing": {"Speed": 2}}
        fast = {"scores": {"correct": 208, "CER": .9}, "timing": {"Speed": 1}}
        selected, accepted = opt.choose("vietlpr", [fast, slow], baseline, cfg)
        self.assertTrue(accepted)
        self.assertIs(selected, slow)
        selected, accepted = opt.choose("vietlpr", [fast], baseline, cfg)
        self.assertFalse(accepted)

    def test_yolo_gate_uses_map_and_fixed_threshold_recall(self):
        cfg = opt.settings()
        baseline = {"scores": {"mAP50-95": 96., "Recall": 97.}}
        self.assertTrue(opt.eligible("yolo", {"scores": {"mAP50-95": 95.2, "Recall": 96.5}}, baseline, cfg))
        self.assertFalse(opt.eligible("yolo", {"scores": {"mAP50-95": 94., "Recall": 97.}}, baseline, cfg))
        self.assertFalse(opt.eligible("yolo", {"scores": {"mAP50-95": 96., "Recall": 95.}}, baseline, cfg))

    def test_unsigned_relu_graph_quantizes_and_runs_batch_one_and_two(self):
        rng = np.random.default_rng(42)
        graph = helper.make_graph([
            helper.make_node("Conv", ["input", "weight"], ["hidden"], name="conv", kernel_shape=[1, 1]),
            helper.make_node("Relu", ["hidden"], ["output"], name="relu")], "small",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 8, 8])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 4, 8, 8])],
            [numpy_helper.from_array(rng.normal(0, .1, (4, 3, 1, 1)).astype(np.float32), "weight")])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
        tensors = [rng.normal(size=(2, 3, 8, 8)).astype(np.float32) for _ in range(3)]
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp) / "source.onnx", Path(temp) / "target.onnx"
            onnx.save(model, source)
            spec = next(s for s in opt.settings()["candidates"]["vietlpr"] if s["name"] == "u8s8_relu")
            with patch.object(opt, "read_split", side_effect=AssertionError("No dataset IO here")):
                opt.quantize(source, target, lambda: iter(tensors), spec)
            quantized = onnx.load(target)
            self.assertTrue(any(w.data_type == TensorProto.INT8 for w in quantized.graph.initializer))
            opts = ort.SessionOptions(); opts.intra_op_num_threads = 1
            session = ort.InferenceSession(str(target), opts, providers=["CPUExecutionProvider"])
            for batch in (1, 2):
                out = session.run(None, {"input": tensors[0][:batch]})[0]
                self.assertEqual(out.shape, (batch, 4, 8, 8))
                self.assertTrue(np.isfinite(out).all())
            self.assertGreater(opt.optimized_inventory(target).get("QLinearConv", 0), 0)

    def test_shared_catalog_has_one_fp32_per_task(self):
        names = bench.names()
        self.assertEqual(len(names), 21)
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names.count("VietLPR_v2"), 1)
        self.assertEqual(names.count("YOLO26n-OBB_FP32"), 1)
        groups = bench.groups()
        self.assertIn("VietLPR_v2", groups["ocr"])
        self.assertIn("VietLPR_v2", groups["quantization"])
        self.assertEqual(len(groups["quantization"]), 4)

    def test_optimized_ocr_uses_original_pipeline_and_frozen_selected_files(self):
        frozen = {"folder": "folder", "scope": {"quantized_part": "rec"}, "artifacts": {"det.onnx": "a", "rec.onnx": "b"}}
        with patch.object(bench.cpu_selection, "validate_selected", return_value=frozen), \
                patch.object(bench.quantization, "source_model", return_value=({"native": True}, None, None)), \
                patch.object(bench.quantization, "settings", return_value={}), \
                patch.object(bench, "onnx_pipeline", return_value="pipeline") as factory, \
                patch.object(Path, "stat") as stat:
            stat.return_value.st_size = 100
            pipeline, state = bench.assets(bench.NEW_OCR, create=True)
        self.assertEqual(pipeline, "pipeline")
        factory.assert_called_once_with({"native": True}, "folder")
        self.assertEqual(state["precision"], "INT8 mixed")

    def test_val_measurement_keeps_io_outside_loop(self):
        pixels = [1, 2, 3]
        with patch.object(opt, "save_json", side_effect=AssertionError("IO")), \
                patch("builtins.open", side_effect=AssertionError("IO")):
            predictions, timing = opt.val_measure(lambda x: {"text": str(x)}, pixels, 0,
                {"val_warmup": 2, "val_rounds": 3, "seed": 42})
        self.assertEqual([p["text"] for p in predictions], ["1", "2", "3"])
        self.assertEqual(len(timing["round_means_ms"]), 3)
        self.assertGreater(timing["Speed"], 0)

    def test_validation_worker_only_loads_val_not_test(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "det.onnx").write_bytes(b"det")
            (folder / "rec.onnx").write_bytes(b"rec")
            row = {"path": Path("val.jpg"), "text": "51A1"}
            with patch.object(opt.old, "source_model", return_value=({"dataset": "dataset"}, None, None)), \
                    patch.object(opt.old, "settings", return_value={}), \
                    patch.object(opt, "read_split", return_value=[row]) as loader, \
                    patch.object(opt, "image", return_value=np.zeros((8, 8, 3), np.uint8)), \
                    patch.object(opt, "onnx_pipeline", return_value=lambda _: {"text": "51A1"}), \
                    patch.object(opt, "optimized_inventory", return_value={}):
                opt.assess("vietlpr", folder)
            self.assertTrue(all(call.args[1] == "val" for call in loader.call_args_list))
            self.assertEqual(opt.read_json(folder / "assessment.json")["scores"]["ACC"], 100.)

    def test_shared_benchmark_rejects_partial_tampered_and_power_changed_passes(self):
        name = "VietLPR_v2"
        rows = [{"relative": "test.jpg", "text": "51A1"}]
        model = {"size_bytes": 100, "precision": "FP32"}
        power = {"ac": True, "power_scheme": "balanced"}
        manifest = {"config": {"rounds": 2}, "models": {name: model}, "code_signature": "code",
                    "power": power, "datasets": {name: {"test": "data"}}}
        records = []
        for index, elapsed in enumerate((1., 3.)):
            scores = bench.text_scores(["51A1"], ["51A1"])
            scores.update(bench.speed_scores([elapsed]))
            records.append({"Model": name, "round": index, "manifest_code": "code", "test_signature": "data",
                            "model": model, "before": {"power": power}, "after": {"power": power}, "scores": scores,
                            "details": [{"image": "test.jpg", "GT": "51A1", "prediction": "51A1", "time_ms": elapsed}]})
        with patch.object(bench, "rows_for", return_value=rows), patch.object(bench, "rows_signature", return_value="data"):
            result = bench.aggregate(records, manifest, name)
            self.assertEqual(result["Speed"], 2.)
            self.assertEqual(result["FPS"], 500.)
            self.assertAlmostEqual(result["Speed_std"], 2 ** .5)
            with self.assertRaisesRegex(ValueError, "Chưa đủ"):
                bench.aggregate(records[:1], manifest, name)
            corrupt = copy.deepcopy(records)
            corrupt[1]["details"][0]["time_ms"] = 10
            with self.assertRaisesRegex(ValueError, "Chỉ số"):
                bench.aggregate(corrupt, manifest, name)
            corrupt = copy.deepcopy(records)
            corrupt[1]["after"]["power"]["power_scheme"] = "other"
            with self.assertRaisesRegex(ValueError, "nguồn"):
                bench.aggregate(corrupt, manifest, name)

    def test_both_ocr_tables_take_exact_same_fp32_record(self):
        result = {"Model": "VietLPR_v2", "ACC": 94., "CER": .8, "Speed": 2., "Speed_std": .1,
                  "FPS": 500., "Size_MB": 1.23}
        records = {"VietLPR_v2": result}
        a, _, _ = bench.baseline.make_table(records, ["VietLPR_v2"])
        b, _, _ = bench.baseline.make_table(records, ["VietLPR_v2"], with_size=True)
        self.assertEqual(a[0], {k: v for k, v in b[0].items() if k != "Size_MB"})

    def test_yolo_silu_quantization_preserves_cpu_execution(self):
        graph = helper.make_graph([
            helper.make_node("Conv", ["input", "weight"], ["features"], name="/model.0/conv/Conv", kernel_shape=[1, 1]),
            helper.make_node("Sigmoid", ["features"], ["gate"], name="/model.0/act/Sigmoid"),
            helper.make_node("Mul", ["features", "gate"], ["output"], name="/model.0/act/Mul")], "silu",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 8, 8])],
            [numpy_helper.from_array(np.random.default_rng(4).normal(0, .1, (4, 3, 1, 1)).astype(np.float32), "weight")])
        m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
        x = np.random.default_rng(8).normal(size=(1, 3, 8, 8)).astype(np.float32)
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "source.onnx", Path(temp) / "quant.onnx"
            onnx.save(m, a)
            with patch.object(yref, "read_image", return_value=x), patch.object(yref.yq, "letterbox", return_value=(x, 1, (0, 0))):
                yref.quantize(a, b, [{"image": "train"}], yref.SILU_VARIANTS[0])
            opts = ort.SessionOptions(); opts.intra_op_num_threads = 1
            session = ort.InferenceSession(str(b), opts, providers=["CPUExecutionProvider"])
            out = session.run(None, {"input": x})[0]
            self.assertEqual(out.shape, (1, 4, 8, 8))
            self.assertTrue(np.isfinite(out).all())
            counts = opt.optimized_inventory(b)
            self.assertGreater(counts.get("QLinearConv", 0), 0)

    def test_cpu_selection_uses_four_distinct_performance_cores_without_smt_siblings(self):
        rows = [{"efficiency": 1, "logical": [i, i + 1]} for i in (0, 2, 4, 6)]
        rows += [{"efficiency": 0, "logical": [i]} for i in range(8, 16)]
        self.assertEqual(cpu_protocol.select(rows), [0, 2, 4, 6])
        with self.assertRaisesRegex(RuntimeError, "Không đủ"):
            cpu_protocol.select(rows, 5)

    def test_common_session_policy_disables_idle_spinning_without_changing_thread_count(self):
        original = ort.SessionOptions
        try:
            cpu_protocol.configure_sessions()
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            self.assertEqual(options.get_session_config_entry("session.intra_op.allow_spinning"), "0")
            self.assertEqual(options.get_session_config_entry("session.inter_op.allow_spinning"), "0")
            self.assertEqual(options.intra_op_num_threads, 4)
            factory = ort.SessionOptions
            cpu_protocol.configure_sessions()
            self.assertIs(ort.SessionOptions, factory)
        finally:
            ort.SessionOptions = original

    def test_rec_static_quantization_keeps_dynamic_attention_matmul_float(self):
        weight = np.random.default_rng(2).normal(0, .1, (32, 16)).astype(np.float32)
        graph = helper.make_graph([
            helper.make_node("MatMul", ["input", "weight"], ["hidden"], name="linear"),
            helper.make_node("Transpose", ["hidden"], ["transposed"], name="transpose", perm=[1, 0]),
            helper.make_node("MatMul", ["hidden", "transposed"], ["output"], name="attention")], "attention",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 32])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])],
            [numpy_helper.from_array(weight, "weight")])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
        x = np.ones((1, 32), np.float32)
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "source.onnx", Path(temp) / "quant.onnx"
            onnx.save(model, a)
            refine.quantize(a, b, [x, x * .5], {"ops": ["MatMul"], "method": "static"})
            m = onnx.load(b)
            attention = next(n for n in m.graph.node if n.name == "attention")
            self.assertEqual(attention.op_type, "MatMul")
            # Không đặt Q/DQ riêng ở đầu ra attention động.
            self.assertFalse(any(n.op_type == "QuantizeLinear" and "output" in n.input for n in m.graph.node))
            opts = ort.SessionOptions(); opts.intra_op_num_threads = 1
            out = ort.InferenceSession(str(b), opts, providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
            self.assertTrue(np.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()

"""Đối chiếu adapter với hàm nguồn, không dùng ACC test để chọn logic."""

import ast
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from ocr.config import ROOT, SOURCE_CHARS, load_config
from ocr.runtime import (
    Pipeline,
    crop_line,
    rec_input,
    source_v2_runtime,
    postprocess_candidates,
)
from ocr.train_torch import v1_dataset, task_rows
from ocr.upstream import v1_modules, v2_source
from ocr import evaluate


class SourceParityTests(unittest.TestCase):
    def test_v2_tune_preserves_default_on_acc_tie_and_updates_native_config(self):
        cfg = load_config("vietlpr_v2")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg["run_dir"] = str(root)
            (root / "onnx").mkdir()
            for name in ("det.onnx", "rec.onnx", "lpr_infer.py"):
                (root / "onnx" / name).write_bytes(b"test artifact")
            (root / "onnx/export_check.json").write_text(
                json.dumps({t: {"smoke_only": False} for t in ("det", "rec")}), encoding="utf-8"
            )
            path = root / "image.jpg"
            path.write_bytes(b"image")
            rows = [{"path": path, "relative": "val/det/image.jpg", "text": "16M9"}]
            default = cfg["postprocess"].copy()
            other = {**default, "threshold": 0.5}

            class FakePipeline:
                def __init__(self):
                    self.cfg = copy.deepcopy(cfg)

                def __call__(self, image):
                    return {
                        "text": (
                            ""
                            if self.cfg["postprocess"]["threshold"] == default["threshold"]
                            else "16M"
                        )
                    }

            with patch.object(evaluate, "read_split", return_value=rows), patch.object(
                evaluate, "image", return_value=np.zeros((20, 40, 3), np.uint8)
            ), patch.object(evaluate, "onnx_pipeline", return_value=FakePipeline()), patch.object(
                evaluate, "postprocess_candidates", return_value=[default, other]
            ):
                result = evaluate.tune(cfg)
            self.assertEqual(result["postprocess"], default)
            native = json.loads((root / "onnx/lpr_config.json").read_text(encoding="utf-8"))
            self.assertEqual(native["det_thresh"], default["threshold"])
            self.assertEqual(native["rec_resize"], "stretch")
            frozen = json.loads((root / "onnx/frozen.json").read_text(encoding="utf-8"))
            self.assertIn("lpr_config.json", frozen["files"])
            self.assertIn("lpr_infer.py", frozen["files"])

    def test_model_architecture_matches_notebook(self):
        root = ROOT / "third_party/VietLPR_v2"
        manifest = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))

        class DropDocs(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                self.generic_visit(node)
                if ast.get_docstring(node):
                    node.body = node.body[1:]
                return node

            def visit_ClassDef(self, node):
                self.generic_visit(node)
                if ast.get_docstring(node):
                    node.body = node.body[1:]
                return node

        for node in ast.parse((root / "models.py").read_text(encoding="utf-8")).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                digest = hashlib.sha256(
                    ast.dump(DropDocs().visit(node), include_attributes=False).encode()
                ).hexdigest()
                self.assertEqual(digest, manifest["models"][node.name], node.name)

    def test_all_v2_definitions_match_notebook_manifest(self):
        root = ROOT / "third_party/VietLPR_v2"
        manifest = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
        tree = ast.parse((root / "source.py").read_text(encoding="utf-8"))
        definitions = {
            n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }
        self.assertEqual(set(manifest["definitions"]), set(definitions))
        for name, node in definitions.items():
            digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            self.assertEqual(digest, manifest["definitions"][name]["ast_sha256"], name)
        self.assertEqual(
            hashlib.sha256(
                (root / "inference.py").read_text(encoding="utf-8").encode()
            ).hexdigest(),
            manifest["inference_sha256"],
        )

    def test_v1_mask_and_inputs_come_from_original_dataset(self):
        cfg = load_config("vietlpr_v1")
        source, _ = v1_modules()
        with tempfile.TemporaryDirectory() as temp:
            cfg["dataset"] = temp
            base = Path(temp) / "val/det"
            (base / "images").mkdir(parents=True)
            cv2.imwrite(str(base / "images/a.jpg"), np.full((50, 90, 3), 180, np.uint8))
            objects = [{"points": [[2, 3], [85, 2], [84, 25], [2, 26]], "transcription": "16M9"}]
            (base / "det_labels.txt").write_text(
                "images/a.jpg\t" + json.dumps(objects) + "\n", encoding="utf-8"
            )
            actual = v1_dataset(cfg, "val", "det", 2)[0]
            expected = source.DetDataset(
                str(base / "det_labels.txt"), str(base / "images"), cfg["model"]["det_size"], False
            )[0]
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(cfg["chars"], "".join(source.CHAR2LABEL))

    def test_v1_crop_matches_original_warp_and_letterbox(self):
        cfg = load_config("vietlpr_v1")
        v1_modules()
        from vietlpr.utils import warp_crop, letterbox_resize

        image = np.random.default_rng(3).integers(0, 256, (61, 130, 3), np.uint8)
        quad = np.float32([[6, 8], [120, 4], [121, 49], [5, 55]])
        expected = (
            letterbox_resize(warp_crop(image, quad), (128, 32))
            .transpose(2, 0, 1)[None]
            .astype(np.float32)
            / 255
        )
        np.testing.assert_array_equal(crop_line(image, quad, cfg), expected)

    def test_v2_full_quad_and_stretch_match_training_source(self):
        cfg = load_config("vietlpr_v2")
        self.assertEqual(cfg["model"]["rec_resize"], "stretch")
        source = v2_source(cfg)
        pixels = np.random.default_rng(9).integers(0, 256, (29, 173, 3), np.uint8)
        expected = (
            source.crop_for_rec(pixels, source.full_quad(pixels))
            .transpose(2, 0, 1)[None]
            .astype(np.float32)
            / 255
        )
        np.testing.assert_array_equal(rec_input(pixels, cfg), expected)
        quad = np.float32([[5, 3], [155, 1], [160, 25], [4, 27]])
        expected = (
            source.crop_for_rec(pixels, quad).transpose(2, 0, 1)[None].astype(np.float32) / 255
        )
        np.testing.assert_array_equal(crop_line(pixels, quad, cfg), expected)

    def test_v2_pipeline_keeps_original_batched_rec_and_reading_order(self):
        cfg = load_config("vietlpr_v2")
        probability = np.zeros((1, 1, 64, 64), np.float32)
        probability[0, 0, 6:20, 8:55] = 0.95
        probability[0, 0, 35:50, 8:55] = 0.95
        batches = []

        def recognize(x):
            batches.append(len(x))
            out = np.zeros((len(x), 32, len(cfg["chars"]) + 1), np.float32)
            out[:, :, 0] = 1
            for i, text in enumerate(("16M9", "0826")):
                for j, c in enumerate(text):
                    out[i, j * 2, 0] = 0
                    out[i, j * 2, cfg["chars"].index(c) + 1] = 1
            return out

        pipeline = Pipeline(cfg, lambda x: probability, recognize)
        result = pipeline(np.full((100, 100, 3), 200, np.uint8))
        self.assertEqual(result["text"], "16M90826")
        self.assertEqual(batches, [2])
        self.assertLess(
            np.mean(result["boxes"][0], axis=0)[1], np.mean(result["boxes"][1], axis=0)[1]
        )

    def test_v2_ema_and_parameter_groups_follow_source(self):
        cfg = load_config("vietlpr_v2")
        source = v2_source(cfg)
        model = torch.nn.Linear(2, 1)
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
        ema = source.ModelEMA(model)
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(1)
        ema.update(model)
        for p in ema.ema.parameters():
            torch.testing.assert_close(p, torch.full_like(p, 9 / 11))
        groups = source.param_groups(model, 0.0005)
        self.assertTrue(any(p is model.bias for p in groups[1]["params"]))
        self.assertEqual(groups[1]["weight_decay"], 0)

    def test_v2_predicted_crop_branch_is_kept(self):
        cfg = load_config("vietlpr_v2")
        source = v2_source(cfg)
        gt = np.float32([[2, 2], [30, 2], [30, 20], [2, 20]])
        predicted = gt + 1
        items = [("image", [{"poly": gt, "text": "16M9", "ignore": False}])]
        ds = source.RecFromDetDataset(items, True, {("image", 0): predicted})
        with patch.object(
            source, "load_img", return_value=np.zeros((32, 40, 3), np.uint8)
        ), patch.object(source.random, "random", return_value=0.1), patch.object(
            source, "jitter_quad", side_effect=lambda q, **kwargs: q
        ) as jitter, patch.object(
            source, "photo_aug", side_effect=lambda x: x
        ):
            ds[0]
        np.testing.assert_array_equal(jitter.call_args.args[0], predicted)
        self.assertEqual(jitter.call_args.kwargs["p"], 0.5)

    def test_source_tune_grid_does_not_add_v1_search_or_last_checkpoints(self):
        self.assertEqual(len(postprocess_candidates(load_config("vietlpr_v1"))), 1)
        grid = postprocess_candidates(load_config("vietlpr_v2"))
        expected = {
            (th, uc, bt, px)
            for th in (0.3, 0.4, 0.5)
            for uc in (1.4, 1.6, 1.8, 2.0)
            for bt in (0.5, 0.7)
            for px in (0.0, 0.1)
        }
        self.assertEqual(
            {(p["threshold"], p["unclip"], p["box_threshold"], p["pad_x"]) for p in grid[1:]},
            expected,
        )

    def test_train_adapter_never_reads_test(self):
        cfg = load_config("vietlpr_v2")
        from ocr import train_torch

        rows = [{"text": "16M9"}]
        with patch.object(train_torch, "read_split", return_value=rows) as read, patch.object(
            train_torch, "split_signature", return_value="signature"
        ):
            task_rows(cfg, "rec")
        self.assertEqual({c.args[1] for c in read.call_args_list}, {"train", "val"})

    def test_stale_runtime_blocks_test_before_reading_data(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "onnx").mkdir()
            (run / "onnx/export_check.json").write_text(
                json.dumps({t: {"smoke_only": False} for t in ("det", "rec")}), encoding="utf-8"
            )
            (run / "onnx/frozen.json").write_text(
                json.dumps({"inference_signature": "old-code"}), encoding="utf-8"
            )
            cfg = load_config("vietlpr_v1")
            cfg["run_dir"] = str(run)
            with patch.object(evaluate, "read_split") as read:
                with self.assertRaisesRegex(ValueError, "Code suy luận đã đổi"):
                    evaluate.benchmark(cfg)
            read.assert_not_called()

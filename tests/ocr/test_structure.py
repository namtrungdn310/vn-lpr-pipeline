"""Kiểm cấu hình model và nguồn đã chốt."""

import json
from pathlib import Path
import tempfile
import unittest

from ocr.config import ROOT, load_config, same_training_config
from ocr.paddle_suite import load_catalog


class StructureTests(unittest.TestCase):
    def test_vietlpr_loader_rejects_legacy_paddle_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            path.write_text(json.dumps({"name": "paddleocr", "chars": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"}))
            with self.assertRaisesRegex(ValueError, "Tên hệ thống"):
                load_config(path)

    def test_paddle_sources_point_to_catalog_and_pinned_processors(self):
        sources = json.loads((ROOT / "configs/ocr/sources.json").read_text(encoding="utf-8"))
        self.assertEqual(set(sources), {"vietlpr_v1", "adaptation", "paddle_suite", "public_vn"})
        paddle = sources["paddle_suite"]
        self.assertTrue((ROOT / paddle["models_config"]).is_file())
        self.assertEqual(len(load_catalog()["variants"]), 7)
        manifest = json.loads((ROOT / paddle["processors"]["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(paddle["processors"]["url"], manifest["repository"])
        self.assertEqual(paddle["processors"]["commit"], manifest["revision"])

    def test_config_origin_may_move_but_training_values_must_match(self):
        cfg = {"name": "vietlpr_v2", "model": {"width": 1}, "seed": 42, "config_path": "old.json"}
        self.assertTrue(same_training_config(cfg, {**cfg, "config_path": "new.json"}))
        self.assertFalse(same_training_config(cfg, {**cfg, "seed": 43}))
        self.assertFalse(same_training_config(cfg, {**cfg, "model": {"width": 2}}))

"""Kiểm cấu hình sau khi di chuyển và quy tắc bảng tổng hợp."""

import json
from pathlib import Path
import tempfile
import unittest

from ocr.config import ROOT, load_config, same_training_config
from ocr.paddle_suite import load_catalog
from ocr.comparison import CONDITIONS, check_conditions, rank_results


class StructureTests(unittest.TestCase):
    def test_vietlpr_loader_rejects_legacy_paddle_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            path.write_text(json.dumps({"name": "paddleocr", "chars": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"}))
            with self.assertRaisesRegex(ValueError, "Tên hệ thống"):
                load_config(path)

    def test_paddle_sources_point_to_catalog_and_pinned_processors(self):
        sources = json.loads((ROOT / "configs/ocr/sources.json").read_text(encoding="utf-8"))
        self.assertEqual(set(sources), {"vietlpr_v1", "vietlpr_v2", "paddle_suite"})
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

    def samples(self):
        base = {key: "same" for key in CONDITIONS}
        return [{**base, "Model": name, "ACC": acc, "CER": cer, "Speed": speed, "FPS": 1000 / speed}
                for name, acc, cer, speed in [("A", 90, 2, 10), ("B", 95, 1, 100), ("C", 90, 1, 20)]]

    def test_rank_prefers_accuracy_then_cer_and_slowest_is_one(self):
        rows = rank_results(self.samples())
        self.assertEqual([row["Model"] for row in rows], ["B", "C", "A"])
        self.assertEqual([row["Rank"] for row in rows], [1, 2, 3])
        self.assertEqual([row["Speedup"] for row in rows], [1, 5, 10])

    def test_different_measurement_settings_are_rejected(self):
        rows = self.samples()
        rows[1]["threads"] = 8
        with self.assertRaisesRegex(ValueError, "threads"):
            check_conditions(rows)

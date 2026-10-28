"""Kiểm split, nhãn và cấu hình Paddle; không cần import PaddlePaddle."""

import json
from pathlib import Path
import tempfile
import unittest

import yaml

from experiments.config import load_config
from experiments.paddle import SOURCE, prepare, prepared_state


@unittest.skipUnless((SOURCE / "tools/train.py").is_file(), "Cần chạy experiments.bootstrap")
class PaddlePreparationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.run = Path(self.directory.name)
        self.cfg = load_config("paddleocr")
        self.cfg["run_dir"] = str(self.run)
        prepare(self.cfg, device="cpu", epochs=1, limit=4)

    def test_uses_existing_train_val_and_common_charset(self):
        state = prepared_state(self.cfg)
        self.assertTrue(state["smoke_only"])
        self.assertEqual(set(state["signatures"]), {"det_train", "det_val", "rec_train", "rec_val"})
        self.assertEqual(
            "".join((self.run / "chars.txt").read_text().splitlines()), self.cfg["chars"]
        )
        for task in ("det", "rec"):
            config = yaml.safe_load((self.run / f"{task}.yml").read_text(encoding="utf-8"))
            self.assertIsNone(config["Global"]["pretrained_model"])
            self.assertFalse(config["Global"]["use_gpu"])
            for split, section in (("train", "Train"), ("val", "Eval")):
                label = Path(config[section]["dataset"]["label_file_list"][0])
                lines = label.read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(lines), 4)
                self.assertTrue(
                    all(f"/{split}/{task}/images/" in line.split("\t")[0] for line in lines)
                )
        first = (self.run / "det_train.txt").read_text().splitlines()[0]
        self.assertTrue(json.loads(first.split("\t")[1])[0]["transcription"])

    def test_modified_yaml_and_config_are_rejected(self):
        path = self.run / "rec.yml"
        path.write_text(path.read_text(encoding="utf-8") + "\n# Đã thay đổi\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            prepared_state(self.cfg)
        self.cfg["model"]["rec_w"] = 128
        with self.assertRaises(ValueError):
            prepared_state(self.cfg)

    def test_prepare_cannot_relabel_existing_weights(self):
        before = (self.run / "prepared.json").read_bytes()
        (self.run / "rec").mkdir()
        (self.run / "rec/best_accuracy.pdparams").write_bytes(b"checkpoint")
        with self.assertRaises(FileExistsError):
            prepare(self.cfg, device="cpu")
        self.assertEqual((self.run / "prepared.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

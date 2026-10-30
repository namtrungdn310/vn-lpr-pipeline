"""Không trộn checkpoint của hai phiên bản dữ liệu."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from ocr import run_state


class RunStateTests(unittest.TestCase):
    def test_native_sidecar_allows_resume_before_canonical_checkpoint_exists(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "runs/vietlpr_v2"
            run.mkdir(parents=True)
            cfg = {
                "name": "vietlpr_v2",
                "run_dir": str(run),
                "dataset": "data",
                "training_revision": 3,
            }
            state = {
                "cfg": cfg,
                "tasks": {
                    "det": {
                        "signatures": {"train": "current", "val": "current"},
                        "smoke_only": False,
                        "epochs": 200,
                    }
                },
            }
            (run / "source_protocol.json").write_text(json.dumps(state), encoding="utf-8")
            with patch.object(run_state, "ROOT", root), patch.object(
                run_state, "read_split", return_value=[]
            ), patch.object(run_state, "split_signature", return_value="current"):
                self.assertIsNone(run_state.prepare_vietlpr_run(cfg))
            self.assertTrue((run / "source_protocol.json").exists())

    def prepare(self, root, *, changed=False, config_changed=False):
        run = root / "runs/vietlpr_v1"
        run.mkdir(parents=True)
        cfg = {"name": "vietlpr_v1", "run_dir": str(run), "dataset": "data"}
        for task in ("det", "rec"):
            checkpoint = {
                "cfg": cfg if not config_changed else {**cfg, "seed": 123},
                "task": task,
                "signatures": {
                    "train": "old" if changed and task == "rec" else "current",
                    "val": "current",
                },
            }
            torch.save(checkpoint, run / f"{task}_last.pth")
            torch.save(checkpoint, run / f"{task}_best.pth")
        (run / "onnx").mkdir()
        (run / "onnx/rec.onnx").write_bytes(b"old-export")
        return run, cfg

    def test_changed_rec_archives_both_tasks_and_exports_without_reading_test(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run, cfg = self.prepare(root, changed=True)
            comparison = root / "runs/comparison_ranked.csv"
            comparison.write_text("previous table", encoding="utf-8")
            before = {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}
            with patch.object(run_state, "ROOT", root), patch.object(
                run_state, "read_split", return_value=[]
            ) as read, patch.object(
                run_state, "split_signature", return_value="current"
            ), redirect_stdout(
                io.StringIO()
            ):
                archive = run_state.prepare_vietlpr_run(cfg)
            self.assertFalse(run.exists())
            self.assertFalse(comparison.exists())
            self.assertEqual(
                (archive / "previous_comparison_ranked.csv").read_text(encoding="utf-8"), "previous table"
            )
            self.assertTrue(archive.is_relative_to((root / "runs/archive").resolve()))
            for name, content in before.items():
                self.assertEqual((archive / name).read_bytes(), content)
            self.assertEqual({c.args[1] for c in read.call_args_list}, {"train", "val"})
            reason = json.loads((archive / "archive_reason.json").read_text(encoding="utf-8"))
            self.assertEqual(set(reason["changed_checkpoints"]), {"rec_last.pth", "rec_best.pth"})

    def test_same_dataset_keeps_resume_files_in_place(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run, cfg = self.prepare(root)
            with patch.object(run_state, "ROOT", root), patch.object(
                run_state, "read_split", return_value=[]
            ), patch.object(run_state, "split_signature", return_value="current"):
                self.assertIsNone(run_state.prepare_vietlpr_run(cfg))
            self.assertTrue((run / "det_last.pth").exists())
            self.assertFalse((root / "runs/archive").exists())

    def test_config_mismatch_does_not_archive_or_discard_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run, cfg = self.prepare(root, changed=True, config_changed=True)
            with patch.object(run_state, "ROOT", root), patch.object(
                run_state, "read_split", return_value=[]
            ), patch.object(
                run_state, "split_signature", return_value="current"
            ), self.assertRaisesRegex(
                ValueError, "Cấu hình khác"
            ):
                run_state.prepare_vietlpr_run(cfg)
            self.assertTrue((run / "det_last.pth").exists())
            self.assertFalse((root / "runs/archive").exists())

    def test_run_outside_runs_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(run_state, "ROOT", root), self.assertRaisesRegex(ValueError, "runs/"):
                run_state.prepare_vietlpr_run({"run_dir": str(root / "outside")})

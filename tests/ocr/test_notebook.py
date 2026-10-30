"""Kiểm log tiến trình con và lệnh tiếp tục train khi chạy lại notebook."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from ocr import notebook


class NotebookTests(unittest.TestCase):
    def test_resume_uses_native_checkpoint_after_interrupted_v2_task(self):
        with tempfile.TemporaryDirectory() as directory:
            native = Path(directory) / "native"
            native.mkdir()
            (native / "rec_last.pth").touch()
            with patch.object(
                notebook, "load_config", return_value={"run_dir": directory}
            ), patch.object(notebook, "prepare_vietlpr_run"), patch.object(notebook, "run") as run:
                notebook.train_vietlpr("python.exe", "vietlpr_v2")
            self.assertEqual(run.call_count, 2)
            self.assertNotIn("--resume", run.call_args_list[0].args)
            self.assertIn("--resume", run.call_args_list[1].args)

    def test_child_stdout_stderr_and_unicode_reach_notebook_and_log(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            notebook, "ROOT", Path(directory)
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                log = notebook.run(
                    sys.executable,
                    "-c",
                    "import sys; print('Đang train GPU'); print('stderr visible', file=sys.stderr)",
                )
            self.assertIn("Đang train GPU", output.getvalue())
            self.assertIn("stderr visible", output.getvalue())
            self.assertIn("Đang train GPU", log.read_text(encoding="utf-8"))

    def test_child_error_exposes_traceback_and_log_path(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            notebook, "ROOT", Path(directory)
        ):
            with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError) as error:
                notebook.run(sys.executable, "-c", "raise ValueError('broken training')")
            self.assertIn("broken training", str(error.exception))
            self.assertIn(".log", str(error.exception))

    def test_vietlpr_resumes_only_existing_task(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "det_last.pth").touch()
            with patch.object(
                notebook, "load_config", return_value={"run_dir": directory}
            ), patch.object(notebook, "prepare_vietlpr_run") as prepare, patch.object(
                notebook, "run"
            ) as run:
                notebook.train_vietlpr("python.exe", "vietlpr_v1")
            prepare.assert_called_once_with({"run_dir": directory})
            self.assertEqual(run.call_count, 2)
            self.assertIn("--resume", run.call_args_list[0].args)
            self.assertNotIn("--resume", run.call_args_list[1].args)

    def test_missing_inputs_have_actionable_message(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError) as error:
                notebook.require_files([Path(directory) / "det_best.pth"], "Chạy bước train trước")
            self.assertIn("Chạy bước train trước", str(error.exception))
            self.assertIn("det_best.pth", str(error.exception))


if __name__ == "__main__":
    unittest.main()

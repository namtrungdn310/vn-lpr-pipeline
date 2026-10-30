"""Kiểm giải nén gói pretrained V4 không ghi file ngoài phạm vi."""

import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from ocr.paddle_pretrained import extract_inference


class PretrainedTests(unittest.TestCase):
    def test_archive_never_extracts_arbitrary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "model.tar"
            with tarfile.open(archive, "w") as bundle:
                for name in (
                    "model/inference.pdmodel",
                    "model/inference.pdiparams",
                    "../other.txt",
                ):
                    member = tarfile.TarInfo(name)
                    member.size = 3
                    bundle.addfile(member, io.BytesIO(b"abc"))
            model = extract_inference(archive, root / "output")
            self.assertEqual(model.read_bytes(), b"abc")
            self.assertEqual(
                {p.name for p in model.parent.iterdir()},
                {"inference.pdmodel", "inference.pdiparams"},
            )
            self.assertFalse((root / "other.txt").exists())

if __name__ == "__main__":
    unittest.main()

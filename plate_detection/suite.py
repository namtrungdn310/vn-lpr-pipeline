"""Train và đo riêng YOLO26n-OBB; không gọi OCR."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

from ocr.config import ROOT, save_json
from ocr.notebook import run
from ocr.paddle_suite_state import model_lock
from .data import load_config
from .evaluate import comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", choices=tuple(load_config()["variants"]), default=list(load_config()["variants"]))
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    directory = ROOT / "runs/plate_detection/suite"
    with model_lock(directory):
        state = {"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
                 "variants": args.variants, "finished": [], "status": "running", "scope": "detection_only"}
        path = directory / "state.json"
        try:
            for variant in args.variants:
                for action in ("train", "val", "export", "test"):
                    if args.train_only and action == "test":
                        continue
                    state.update(variant=variant, action=action)
                    save_json(path, state)
                    module = "plate_detection.evaluate" if action == "test" else "plate_detection.train"
                    run(sys.executable, "-m", module, action, "--variant", variant)
                state["finished"].append(variant)
                save_json(path, state)
                if not args.train_only:
                    print(comparison(allow_partial=True)[1], flush=True)
            state["status"] = "complete"
            state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        except BaseException as error:
            state["status"] = "failed"
            state["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            save_json(path, state)


if __name__ == "__main__":
    main()

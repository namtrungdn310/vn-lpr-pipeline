"""Chạy bảy cấu hình Paddle bằng ONNX và tiền/hậu xử lý PaddleX gốc."""

import copy
from pathlib import Path

import numpy as np
import yaml

from third_party.PaddleOCR_suite.processors import (
    CTCLabelDecode,
    CropByPolys,
    DBPostProcess,
    DetResizeForTest,
    NormalizeImage,
    OCRReisizeNormImg,
    SortQuadBoxes,
)
from .metrics import normalize


def model_config(directory, task):
    path = Path(directory) / "source" / task / "inference.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class Preprocessor:
    def __init__(self, directory, resize):
        det, rec = (model_config(directory, task) for task in ("det", "rec"))
        self.det_resize = DetResizeForTest(**resize)
        transforms = det["PreProcess"]["transform_ops"]
        norm = copy.deepcopy(next(t["NormalizeImage"] for t in transforms if "NormalizeImage" in t))
        # Truyền số trực tiếp; không eval chuỗi trong YAML tải về.
        if norm.get("scale") not in ("1./255.", "1/255", 1 / 255):
            raise ValueError("Scale DET khác cấu hình đã kiểm tra")
        norm["scale"] = 1 / 255
        self.det_normalize = NormalizeImage(**norm)
        if norm.get("order") != "hwc":
            raise ValueError("Cần NormalizeImage theo HWC")
        rec_transforms = rec["PreProcess"]["transform_ops"]
        shape = next(t["RecResizeImg"]["image_shape"] for t in rec_transforms if "RecResizeImg" in t)
        self.rec_resize = OCRReisizeNormImg(rec_image_shape=shape)
        post = rec["PostProcess"]
        if post["name"] != "CTCLabelDecode":
            raise ValueError("Model REC không dùng CTC")
        self.decoder = CTCLabelDecode(
            character_list=post["character_dict"], use_space_char=post.get("use_space_char", True)
        )
        self.default_postprocess = {k: v for k, v in det["PostProcess"].items() if k != "name"}

    def det_input(self, pixels):
        resized, shapes = self.det_resize([pixels])
        normalized = self.det_normalize(resized)[0]
        return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None]), shapes

    def rec_input(self, pixels):
        return np.ascontiguousarray(self.rec_resize([pixels])[0][None])


def session(path, threads):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    result = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    if len(result.get_inputs()) != 1 or result.get_inputs()[0].type != "tensor(float)":
        raise ValueError("Model cần một input FP32")
    return result


class PaddlePipeline:
    def __init__(self, directory, resize, threads, postprocess=None):
        self.pre = Preprocessor(directory, resize)
        self.postprocess = copy.deepcopy(postprocess or self.pre.default_postprocess)
        self.det, self.rec = (session(Path(directory) / "onnx" / f"{t}.onnx", threads) for t in ("det", "rec"))
        self.sort = SortQuadBoxes()
        self.cropper = CropByPolys()

    def detect_raw(self, pixels):
        inputs, shapes = self.pre.det_input(pixels)
        output = self.det.run(None, {self.det.get_inputs()[0].name: inputs})
        if output[0].ndim != 4 or output[0].shape[:2] != (1, 1) or not np.isfinite(output[0]).all():
            raise ValueError("Output DET không hợp lệ")
        return output, shapes

    def recognize(self, pixels, raw):
        outputs, shapes = raw
        boxes, _ = DBPostProcess(**self.postprocess)(outputs, shapes)
        boxes = self.sort(boxes[0])
        texts = []
        for box in boxes:
            crop = self.cropper.get_minarea_rect_crop(pixels, box)
            if not crop.size:
                continue
            inputs = self.pre.rec_input(crop)
            output = self.rec.run(None, {self.rec.get_inputs()[0].name: inputs})
            if output[0].ndim != 3 or output[0].shape[-1] != len(self.pre.decoder.character):
                raise ValueError("Dictionary REC không khớp output model")
            if not np.isfinite(output[0]).all():
                raise ValueError("Output REC chứa NaN/Inf")
            decoded, _ = self.pre.decoder(output)
            texts.append(decoded[0])
        return {"text": normalize("".join(texts)), "lines": texts, "boxes": np.asarray(boxes).tolist()}

    def __call__(self, pixels):
        return self.recognize(pixels, self.detect_raw(pixels))

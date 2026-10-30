"""Kiểm decoder, NMS và đầu vào các model biển Việt Nam pretrained."""

import unittest
from unittest.mock import patch

import numpy as np

from ocr.public_vn import load_catalog, variant
from ocr.public_vn_runtime import PublicVNPipeline, decode_lprnet, model_input, yolo_boxes, yolo_input, yolo_text


class PublicVNTests(unittest.TestCase):
    def test_sources_pin_vietnamese_training_evidence_and_weights(self):
        catalog = load_catalog()
        self.assertEqual(len(catalog["variants"]), 5)
        for v in catalog["variants"]:
            self.assertEqual(len(v["revision"]), 40)
            self.assertIn(v["revision"], v["training_evidence"])
            self.assertIn(v["revision"], v["weights"]["url"])
            self.assertEqual(len(v["weights"]["sha256"]), 64)
            self.assertEqual(len(v["chars"]), len(set(v["chars"])))

    def test_lprnet_blank_is_last_and_blank_separates_repeated_digit(self):
        output = np.zeros((1, 4, 6), np.float32)
        for t, i in enumerate([0, 0, 3, 0, 1, 3]):
            output[0, i, t] = 1
        self.assertEqual(decode_lprnet(output, "01A-"), "001")

    def test_lprnet_keeps_bgr_and_native_scale(self):
        cfg = variant(load_catalog(), "LPRNet_VN")
        pixels = np.full((30, 110, 3), [0, 127, 255], dtype=np.uint8)
        actual = model_input(pixels, cfg)
        self.assertEqual(actual.shape, (1, 3, 24, 94))
        np.testing.assert_array_equal(actual[0, :, 0, 0], np.float32([-127.5, -0.5, 127.5]) / 128)

    def test_yolo_letterbox_is_rectangular_rgb_and_stride_aligned(self):
        pixels = np.full((40, 200, 3), [0, 0, 255], dtype=np.uint8)
        inputs, ratio, padding = yolo_input(pixels)
        self.assertEqual(inputs.shape, (1, 3, 128, 640))
        self.assertEqual(ratio, 3.2)
        self.assertEqual(padding, (0, 0))
        np.testing.assert_array_equal(inputs[0, :, 10, 10], [1, 0, 0])

    def test_nms_suppresses_same_class_but_keeps_different_character(self):
        cfg = {"family": "yolov8", "chars": "AB", "confidence": 0.4, "iou": 0.7, "max_det": 300}
        output = np.float32([[[50, 50, 50], [30, 30, 30], [20, 20, 20], [30, 30, 30],
                              [0.95, 0.90, 0.01], [0.01, 0.01, 0.94]]])
        boxes = yolo_boxes(output, np.zeros((80, 100, 3), np.uint8), cfg, 1, (0, 0))
        self.assertEqual([b[1] for b in boxes], ["A", "B"])

    def test_v5_uses_checkpoint_dictionary_and_top_then_bottom(self):
        boxes = [(np.array([10 + i * 20, y, 20 + i * 20, y + 20]), char, 0.9)
                 for y, text in ((10, "59C1"), (45, "12345")) for i, char in enumerate(text)]
        self.assertEqual(yolo_text(boxes[::-1], "yolov5"), "59C112345")
        self.assertEqual(yolo_text(boxes[:3], "yolov5"), "")

    def test_v8_preserves_single_line_order_and_separates_two_lines(self):
        one = [(np.array([x, y, x + 8, y + 20]), text, .9)
               for x, y, text in ((10, 10, "5"), (30, 12, "9"), (50, 11, "A"))]
        self.assertEqual(yolo_text(one[::-1], "yolov8"), "59A")
        two = one + [(np.array([5, 40, 12, 60]), "1", .9)]
        self.assertEqual(yolo_text(two[::-1], "yolov8"), "59A1")

    def test_cct_is_slot_decoder_and_does_not_collapse_repeated_digits(self):
        cfg = variant(load_catalog(), "CCT_S_VN")
        scores = np.zeros((1, 9, 37), np.float32)
        for t, char in enumerate("51A11111_"):
            scores[0, t, cfg["chars"].index(char)] = 1

        class Engine:
            def get_inputs(self):
                return [type("Input", (), {"name": "input"})()]

            def run(self, outputs, feeds):
                self.dtype = feeds["input"].dtype
                return [scores]

        engine = Engine()
        with patch("ocr.public_vn_runtime.session", return_value=engine):
            pipe = PublicVNPipeline("unused", cfg)
            result = pipe(np.zeros((64, 128, 3), np.uint8))
            self.assertEqual(result["text"], "51A11111")
            self.assertEqual(engine.dtype, np.uint8)
            self.assertEqual(pipe(np.zeros((10, 30, 3), np.uint8))["text"], "")


if __name__ == "__main__":
    unittest.main()

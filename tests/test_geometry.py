import math
import unittest

from PIL import Image, ImageDraw

from src.geometry import intersection_area, rectify_crop, validate_quad
from src.rec_preprocessing import prepare_rec_crop, resize_pad


class GeometryTests(unittest.TestCase):
    def test_rejects_degenerate_crossed_and_out_of_bounds_quads(self):
        cases = [
            [[0, 0]] * 4,
            [[0, 0], [40, 30], [40, 0], [0, 30]],
            [[-1, 0], [40, 0], [40, 30], [0, 30]],
            [[0, 0], [math.nan, 0], [40, 30], [0, 30]],
        ]
        for points in cases:
            with self.subTest(points=points), self.assertRaises(ValueError):
                validate_quad(points, 50, 40)

    def test_continuous_bounds_allow_width_and_height(self):
        self.assertEqual(
            validate_quad([[0, 0], [50, 0], [50, 40], [0, 40]], 50, 40)[2],
            [50.0, 40.0],
        )

    def test_narrow_text_axis_does_not_rotate_or_mirror(self):
        image = Image.new("RGB", (60, 90), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((12, 12, 20, 70), fill="red")
        draw.rectangle((30, 12, 38, 70), fill="blue")
        points = [[10, 10], [40, 10], [40, 75], [10, 75]]
        crop = prepare_rec_crop(image, points)
        self.assertEqual(crop.size, (30, 65))
        self.assertGreater(crop.getpixel((6, 30))[0], 200)
        self.assertGreater(crop.getpixel((25, 30))[2], 200)

    def test_crop_at_image_edge_does_not_add_black_border(self):
        image = Image.new("RGB", (50, 30), "white")
        crop = rectify_crop(image, [[0, 0], [50, 0], [50, 30], [0, 30]])
        self.assertEqual(crop.getpixel((crop.width - 1, crop.height - 1)), (255, 255, 255))

    def test_intersection_uses_polygon_instead_of_axis_aligned_box(self):
        first = [[0, 0], [10, 10], [10, 12], [0, 2]]
        separate = [[0, 4], [10, 14], [10, 16], [0, 6]]
        overlapping = [[0, 1], [10, 11], [10, 13], [0, 3]]
        self.assertEqual(intersection_area(first, separate), 0.0)
        self.assertGreater(intersection_area(first, overlapping), 0.1)

    def test_long_rec_keeps_last_character_after_resize(self):
        image = Image.new("RGB", (400, 40), "white")
        ImageDraw.Draw(image).rectangle((380, 0, 399, 39), fill="blue")
        result = resize_pad(image, height=48, width=320)
        self.assertEqual(result.size, (320, 48))
        self.assertGreater(result.getpixel((310, 15))[2], 200)
        self.assertLess(result.getpixel((310, 15))[0], 30)


if __name__ == "__main__":
    unittest.main()

import unittest

from experiments.metrics import normalize, speed_scores, text_scores


class MetricsTests(unittest.TestCase):
    def test_acc_is_exact_match_and_cer_uses_total_reference_characters(self):
        score = text_scores(["51F88271", "0826"], ["51F88270", "0826"])
        self.assertEqual(score["ACC"], 50)
        self.assertAlmostEqual(score["CER"], 100 / 12)

    def test_missing_detection_is_counted_instead_of_removed(self):
        score = text_scores(["", "0826"], ["51F88270", "0826"])
        self.assertEqual(score["ACC"], 50)
        self.assertAlmostEqual(score["CER"], 800 / 12)

    def test_no_automatic_character_correction_or_loss_of_leading_zero(self):
        self.assertEqual(normalize(" 51f-882.70 "), "51F88270")
        self.assertEqual(normalize("O826"), "O826")
        self.assertEqual(normalize("0826"), "0826")

    def test_speed_uses_mean_and_total_throughput(self):
        score = speed_scores([10.0, 30.0])
        self.assertEqual(score["Speed"], 20.0)
        self.assertEqual(score["FPS"], 50.0)
        for times in ([], [0], [-1], [float("nan")]):
            with self.assertRaises(ValueError):
                speed_scores(times)

    def test_incomplete_predictions_are_rejected(self):
        with self.assertRaises(ValueError):
            text_scores(["A"], ["A", "B"])


if __name__ == "__main__":
    unittest.main()

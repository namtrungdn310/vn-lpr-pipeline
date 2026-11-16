"""Kiểm lượt đo chung, thống kê giữa lượt và FP32 dùng chung hai bảng."""

import copy
import contextlib
import io
import runpy
import statistics
import unittest
from unittest.mock import patch

from ocr import benchmark as b
from ocr import benchmark_display as display
from ocr.metrics import speed_scores, text_scores


class BenchmarkTests(unittest.TestCase):
    def test_sixteen_unique_models_with_single_fp32(self):
        names = b.names()
        self.assertEqual(len(names), 16)
        self.assertEqual(len(set(names)), 16)
        self.assertEqual(names.count("VietLPR_v2"), 1)
        self.assertEqual(len([n for n in names if n not in b.QUANT_NAMES]), 14)
        self.assertEqual(b.QUANT_TABLE, ("VietLPR_v2", "VietLPR_v2_INT8", "VietLPR_v2_INT4"))

    def test_independent_rounds_have_deterministic_rotated_model_order(self):
        names = b.names()
        orders = [b.round_order(names, i, 42) for i in range(5)]
        self.assertEqual(orders, [b.round_order(names, i, 42) for i in range(5)])
        self.assertTrue(all(sorted(order) == sorted(names) for order in orders))
        self.assertEqual(len({tuple(order) for order in orders}), 5)

    def fixture(self):
        m = {"config": {"rounds": 3}, "models": {"A": {"size_bytes": 100, "precision": "FP32"}},
             "test_signature": "test", "code_signature": "code"}
        rows = [{"relative": "one.jpg", "text": "51A1"}, {"relative": "two.jpg", "text": "1234"}]
        records = []
        for index, times in enumerate(((1, 3), (3, 5), (5, 7))):
            score = text_scores([r["text"] for r in rows], [r["text"] for r in rows])
            score.update(speed_scores(list(times)))
            records.append({"Model": "A", "round": index, "manifest_code": "code", "test_signature": "test",
                "model": m["models"]["A"], "scores": score, "details": [
                    {"image": r["relative"], "GT": r["text"], "prediction": r["text"], "time_ms": t}
                    for r, t in zip(rows, times)]})
        return m, rows, records

    def test_standard_deviation_is_across_round_means_not_different_images(self):
        m, rows, records = self.fixture()
        result = b.aggregate_passes(records, m, "A", rows)
        self.assertEqual(result["Speed"], 4)
        self.assertEqual(result["FPS"], 250)
        self.assertEqual(result["round_mean_ms"], [2, 4, 6])
        self.assertEqual(result["Speed_std"], statistics.stdev([2, 4, 6]))
        self.assertEqual([r["times_ms"] for r in result["details"]], [[1, 3, 5], [3, 5, 7]])

    def test_partial_and_tampered_passes_are_rejected(self):
        m, rows, records = self.fixture()
        with self.assertRaisesRegex(ValueError, "Chưa đủ"):
            b.aggregate_passes(records[:2], m, "A", rows)
        corrupt = copy.deepcopy(records)
        corrupt[1]["details"][0]["time_ms"] = 100
        with self.assertRaisesRegex(ValueError, "Chỉ số"):
            b.aggregate_passes(corrupt, m, "A", rows)

    def test_invalid_ac_and_power_scheme_are_rejected(self):
        expected = {"ac": True, "power_scheme": "balanced"}
        with patch.object(b, "power_context", return_value={"ac": False, "power_scheme": "balanced"}):
            with self.assertRaisesRegex(RuntimeError, "Cần cắm sạc"):
                b.check_power(expected, {"require_ac": True})
        with patch.object(b, "power_context", return_value={"ac": True, "power_scheme": "other"}):
            with self.assertRaisesRegex(RuntimeError, "Chế độ nguồn"):
                b.check_power(expected, {"require_ac": True})

    def test_both_tables_reuse_same_fp32_result(self):
        result = {"Model": "VietLPR_v2", "ACC": 94, "CER": 1, "Speed": 2, "Speed_std": .1,
                  "FPS": 500, "Size_MB": 1}
        results = {"VietLPR_v2": result}
        ocr, _, _ = b.make_table(results, ["VietLPR_v2"])
        quant, _, _ = b.make_table(results, ["VietLPR_v2"], with_size=True)
        self.assertEqual({k: v for k, v in quant[0].items() if k != "Size_MB"}, ocr[0])

    def test_current_quantization_view_hides_old_int8_and_ranks_by_fps(self):
        names = ["VietLPR_v2", display.measured.NEW_OCR, "VietLPR_v2_INT4", "VietLPR_v2_INT8"]
        results = {name: {"Model": name, "ACC": 95 - i, "CER": i, "Speed": speed,
                          "Speed_std": .1, "FPS": 1000 / speed, "Size_MB": 1}
                   for i, (name, speed) in enumerate(zip(names, (2, 1, 3, 10)))}
        original = copy.deepcopy(results)
        rows, _, _ = display.render("quantization", results)
        self.assertEqual([row["Model"] for row in rows],
                         ["VietLPR_v1_INT8_mixed", "VietLPR_v1_FP32", "VietLPR_v1_INT4_mixed"])
        self.assertEqual([row["Rank"] for row in rows], [1, 2, 3])
        self.assertEqual([row["Speedup"] for row in rows], [3, 1.5, 1])
        self.assertEqual(results, original)

    def test_full_ocr_view_keeps_fourteen_models_ranked_by_accuracy(self):
        results = {name: {"Model": name, "ACC": i, "CER": 1, "Speed": i + 1,
                          "Speed_std": .1, "FPS": 1000 / (i + 1)}
                   for i, name in enumerate(display.GROUPS["ocr"])}
        rows, _, _ = display.render("ocr", results)
        self.assertEqual(len(rows), 14)
        self.assertEqual([row["ACC"] for row in rows], list(reversed(range(14))))

    def test_comparison_cli_reads_current_benchmark_instead_of_legacy_results(self):
        from ocr.adaptation import benchmark as adapted
        with patch.object(adapted, "table", return_value="current table") as table:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                runpy.run_module("ocr.comparison", run_name="__main__")
        table.assert_called_once_with()
        self.assertEqual(output.getvalue().strip(), "current table")


if __name__ == "__main__":
    unittest.main()

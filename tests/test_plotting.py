"""Check result parsing without requiring the optional plotting dependency."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from plot_benchmark import METRICS, read_results


class PlotInputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def row(self, dataset="nr", method="random"):
        row = dict(dataset=dataset, method=method, runs=5, top_n=10)
        for metric in METRICS:
            row[f"{metric}_mean"] = 0.25
            row[f"{metric}_sd"] = 0.1
        return row

    def save(self, rows, fields=None):
        with (self.root / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_reads_summary_and_preserves_undefined_metrics(self):
        row = self.row()
        row["ap_mean"], row["ap_sd"] = "", ""
        self.save([row])
        tables = read_results(self.root)
        self.assertIsNone(tables["nr"]["random"]["ap_mean"])
        self.assertEqual(tables["nr"]["random"]["runs"], 5)
        self.assertEqual(tables["nr"]["random"]["auroc_sd"], 0.1)

    def test_rejects_nan_infinite_and_out_of_range_means(self):
        for value in ("nan", "inf", "-0.1", "1.01"):
            with self.subTest(value=value):
                row = self.row()
                row["ap_mean"] = value
                self.save([row])
                with self.assertRaises(ValueError):
                    read_results(self.root)

    def test_rejects_missing_columns_duplicate_rows_and_inconsistent_methods(self):
        row = self.row()
        del row["ap_sd"]
        self.save([row])
        with self.assertRaises(ValueError):
            read_results(self.root)
        self.save([self.row(), self.row()])
        with self.assertRaises(ValueError):
            read_results(self.root)
        self.save([self.row(), self.row("e", "two-sided")])
        with self.assertRaises(ValueError):
            read_results(self.root)

    def test_rejects_cutoff_mismatch_and_negative_time(self):
        row = self.row("e")
        row["top_n"] = 5
        self.save([self.row(), row])
        with self.assertRaises(ValueError):
            read_results(self.root)
        row = self.row()
        row["score_seconds_mean"] = -1
        self.save([row])
        with self.assertRaises(ValueError):
            read_results(self.root)

    def test_legacy_single_seed_is_supported(self):
        directory = self.root / "nr_seed42"
        directory.mkdir()
        with (directory / "test.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["method", "top_n", *METRICS])
            writer.writeheader()
            writer.writerow(dict(method="random", top_n=10, ap=0.25, auroc=0.5,
                                 macro_top_precision=0.1, macro_top_recall=0.2))
        tables = read_results(self.root, seed=42)
        self.assertEqual(tables["nr"]["random"]["ap_sd"], 0)
        self.assertIsNone(tables["nr"]["random"]["score_seconds_mean"])


if __name__ == "__main__":
    unittest.main()

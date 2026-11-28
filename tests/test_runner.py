"""A complete small run checks the benchmark/prediction file contract."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from predict import main as predict_main
from run import aggregate, main as run_main


def write_labelled(path, rows, columns, matrix):
    content = ["\t".join(columns)]
    content.extend("\t".join([identifier, *map(str, values)]) for identifier, values in zip(rows, matrix))
    path.write_text("\n".join(content) + "\n")


class RunnerTests(unittest.TestCase):
    def test_benchmark_to_predictions_and_hash_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            drugs, proteins = tuple(f"d{i}" for i in range(5)), tuple(f"p{i}" for i in range(5))
            interactions = np.eye(5, dtype=int) + np.roll(np.eye(5, dtype=int), 1, axis=1)
            write_labelled(root / "tiny_admat_dgc.txt", drugs, proteins, interactions)
            write_labelled(root / "tiny_simmat_dc.txt", drugs, drugs, np.ones((5, 5)))
            write_labelled(root / "tiny_simmat_dg.txt", proteins, proteins, np.ones((5, 5)))
            output = root / "benchmark"
            args = ["--dataset", "tiny", "--seed", "0", "--input", str(root), "--output", str(output),
                    "--ks", "2", "--powers", "1", "--alphas", "0.5", "--timing-repeats", "1", "--compare-backends"]
            with contextlib.redirect_stdout(io.StringIO()):
                run_main(args)
            receipt = json.loads((output / "tiny_seed0" / "run.json").read_text())
            self.assertEqual(receipt["counts"]["train"], 8)
            self.assertEqual(receipt["counts"]["validation"], 1)
            self.assertEqual(receipt["counts"]["test"], 1)
            self.assertEqual(receipt["counts"]["test_candidates"], 16)
            self.assertEqual(len(receipt["data"]["input_sha256"]), 3)
            with (output / "all_runs.csv").open(newline="") as handle:
                results = list(csv.DictReader(handle))
            self.assertEqual(len(results), 7)
            self.assertTrue(all(int(r["positives"]) == 1 and int(r["background"]) == 15 for r in results))
            prediction_args = ["--dataset", "tiny", "--input", str(root), "--settings",
                               str(output / "tiny_seed0" / "run.json"), "--output", str(root / "predictions")]
            with contextlib.redirect_stdout(io.StringIO()):
                predict_main(prediction_args)
            with (root / "predictions" / "predictions.csv").open(newline="") as handle:
                predictions = list(csv.DictReader(handle))
            self.assertEqual(len(predictions), 15)
            for row in predictions:
                self.assertEqual(interactions[drugs.index(row["drug"]), proteins.index(row["protein"])], 0)
            with self.assertRaises(ValueError):
                run_main(args)
            # Even a harmless source-file edit must invalidate saved settings.
            path = root / "tiny_simmat_dc.txt"
            path.write_text(path.read_text() + "\n")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                predict_main(prediction_args)

    def test_aggregate_uses_sample_sd_and_counts_defined_values(self):
        common = dict(dataset="tiny", method="random", top_n=10,
                      auroc=None, macro_top_precision=.2, macro_top_recall=.4, score_seconds=.01)
        rows = [dict(common, ap=.2), dict(common, ap=.4)]
        result = aggregate(rows)[0]
        self.assertAlmostEqual(result["ap_mean"], .3)
        self.assertAlmostEqual(result["ap_sd"], .2 / np.sqrt(2))
        self.assertEqual(result["ap_n"], 2)
        self.assertIsNone(result["auroc_mean"])
        self.assertEqual(result["auroc_n"], 0)


if __name__ == "__main__":
    unittest.main()

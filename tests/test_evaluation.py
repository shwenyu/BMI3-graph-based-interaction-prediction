"""Ranking checks use hand-calculated values rather than another metric library."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from evaluation import evaluate, ranking_metrics


class EvaluationTests(unittest.TestCase):
    def test_hand_rankings_and_tied_groups(self):
        result = ranking_metrics([1, 0, 1, 0], [.9, .8, .7, .6])
        self.assertAlmostEqual(result["ap"], 5 / 6)
        self.assertAlmostEqual(result["auroc"], 3 / 4)
        tied = ranking_metrics([1, 0], [.5, .5])
        self.assertEqual(tied["ap"], .5)
        self.assertEqual(tied["auroc"], .5)
        reversed_order = ranking_metrics([0, 1], [2, 1])
        self.assertEqual(reversed_order["ap"], .5)
        self.assertEqual(reversed_order["auroc"], 0)

    def test_undefined_metrics_are_none(self):
        no_positive = ranking_metrics([0, 0], [2, 1])
        self.assertIsNone(no_positive["ap"])
        self.assertIsNone(no_positive["auroc"])
        only_positive = ranking_metrics([1, 1], [2, 1])
        self.assertEqual(only_positive["ap"], 1)
        self.assertIsNone(only_positive["auroc"])
        pooled, rows, _ = evaluate(np.zeros((2, 2)), np.zeros((2, 2), bool), np.ones((2, 2), bool), ("a", "b"))
        self.assertIsNone(pooled["macro_top_precision"])
        self.assertIsNone(pooled["macro_top_recall"])
        self.assertEqual(pooled["eligible_drugs"], 0)
        self.assertEqual(len(rows), 2)

    def test_top_ties_follow_ids_not_labels_or_column_order(self):
        scores = np.zeros((1, 3))
        hidden = np.array([[1, 0, 0]], bool)
        candidates = np.ones_like(hidden)
        metrics, rows, _ = evaluate(scores, hidden, candidates, ("b", "a", "c"), 1)
        self.assertEqual(rows[0]["hits"], 0)
        permutation = [2, 0, 1]
        repeated, _, _ = evaluate(scores[:, permutation], hidden[:, permutation], candidates[:, permutation], ("c", "b", "a"), 1)
        self.assertEqual(metrics, repeated)

    def test_precision_uses_available_candidates_and_macro_eligibility(self):
        metrics, rows, _ = evaluate([[.8, .2], [.8, .2]], [[1, 0], [0, 0]], [[1, 1], [1, 1]], ("a", "b"), 10)
        self.assertEqual(metrics["macro_top_precision"], .5)
        self.assertEqual(metrics["macro_top_recall"], 1)
        self.assertEqual(metrics["eligible_drugs"], 1)
        self.assertEqual(metrics["drugs_without_hidden"], 1)
        self.assertIsNone(rows[1]["recall"])

    def test_invalid_labels_scores_and_top_count_fail_clearly(self):
        for labels, scores in (([2, 0], [1, 0]), ([1], [float("nan")]), ([], []), ([1, 0], [1])):
            with self.assertRaises(ValueError):
                ranking_metrics(labels, scores)
        for top_n in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                evaluate([[1, 0]], [[1, 0]], [[1, 1]], ("a", "b"), top_n)
        with self.assertRaises(ValueError):
            evaluate([[1, float("inf")]], [[1, 0]], [[1, 0]], ("a", "b"))
        with self.assertRaises(ValueError):
            evaluate([[1, 0]], [[1, 0]], [[0, 1]], ("a", "b"))


if __name__ == "__main__":
    unittest.main()

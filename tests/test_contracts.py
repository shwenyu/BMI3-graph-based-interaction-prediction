"""Small independent examples verify evaluation contracts, not project accuracy."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from bmi3_dti.data import align_square, read_matrix
from bmi3_dti.benchmark import make_split
from bmi3_dti.evaluation import ranking_metrics, evaluate
from bmi3_dti.scoring import score


class Contracts(unittest.TestCase):
    def setUp(self):
        self.a = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8)

    def test_hiding_preserves_nodes_and_never_exposes_answers(self):
        for mode in ("random", "drug", "protein"):
            x = make_split(self.a, .34, 12, mode)
            y = make_split(self.a, .34, 12, mode)
            np.testing.assert_array_equal(x.hidden, y.hidden)
            np.testing.assert_array_equal(self.a, x.visible + x.hidden)
            self.assertFalse(np.any(x.visible[x.hidden]))
            self.assertTrue(np.all(x.candidates[x.hidden]))
            self.assertFalse(np.any(x.candidates & (x.visible == 1)))
            self.assertEqual(x.visible.shape, self.a.shape)
            if mode == "drug":
                self.assertFalse(np.any(x.visible[x.selected_entities]))
            if mode == "protein":
                self.assertFalse(np.any(x.visible[:, x.selected_entities]))

    def test_alignment_uses_both_axes_and_rejects_unknown_ids(self):
        a = np.array([[1, .3], [.2, 1]])
        out = align_square(("b", "a"), ("a", "b"), a, ("a", "b"))
        np.testing.assert_array_equal(out, [[.2, 1], [1, .3]])
        with self.assertRaises(ValueError):
            align_square(("a", "b"), ("a", "b"), a, ("a", "c"))

    def test_matrix_reader_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"bad.txt"
            p.write_text("\ta\tb\na\t1\t0\na\t0\t1\n")
            with self.assertRaises(ValueError):
                read_matrix(p)

    def test_metrics_against_hand_calculations(self):
        m = ranking_metrics([1, 0, 1, 0], [4, 3, 2, 1])
        self.assertAlmostEqual(m["ap"], 5/6)
        self.assertAlmostEqual(m["auroc"], .75)
        m = ranking_metrics([1, 0, 1, 0], [1, 1, 1, 1])
        self.assertEqual(m["ap"], .5)
        self.assertEqual(m["auroc"], .5)
        self.assertIsNone(ranking_metrics([1, 1], [2, 1])["auroc"])
        self.assertIsNone(ranking_metrics([0, 0], [2, 1])["ap"])

    def test_topn_ties_use_ids_not_correct_answers(self):
        scores = np.ones((2, 3))
        hidden = np.array([[0, 1, 0], [0, 0, 0]], dtype=bool)
        m, rows, _ = evaluate(scores, hidden, np.ones_like(hidden), ("c", "a", "b"), 1)
        self.assertEqual(rows[0]["hits"], 1)
        self.assertEqual(m["eligible_drugs"], 1)
        self.assertIsNone(rows[1]["recall"])

    def test_baseline_uses_visible_degrees(self):
        x = make_split(self.a, .34, 1)
        scores = score("target-degree", x.visible, None, None, 1)
        np.testing.assert_array_equal(scores[0], x.visible.sum(axis=0))
        with self.assertRaises(NotImplementedError):
            score("two-sided", x.visible, None, None, 1)


if __name__ == "__main__":
    unittest.main()

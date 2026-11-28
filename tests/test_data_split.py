"""Small, independently checked fixtures for data alignment and hidden edges."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from data import align_square, load_dataset, read_matrix
from split import candidate_masks, split_edges


class DataAndSplitTests(unittest.TestCase):
    def test_split_preserves_exactly_known_edges_without_mutation(self):
        original = np.ones((4, 5), dtype=np.uint8)
        before = original.copy()
        train, validation, test = split_edges(original, 42)
        self.assertEqual([int(x.sum()) for x in (train, validation, test)], [16, 2, 2])
        np.testing.assert_array_equal(original, before)
        np.testing.assert_array_equal(train.astype(int) + validation + test, original)
        self.assertFalse(np.any(train & validation | train & test | validation & test))
        for actual, repeated in zip((train, validation, test), split_edges(original, 42)):
            np.testing.assert_array_equal(actual, repeated)
        for invalid in (np.array([[2]]), np.array([]), np.array([1, 0])):
            with self.assertRaises(ValueError):
                split_edges(invalid, 42)
        for fractions in ((0, .1), (.5, .5), (float("nan"), .1)):
            with self.assertRaises(ValueError):
                split_edges(original, 42, *fractions)
        with self.assertRaises(ValueError):
            split_edges(np.eye(3), 42)

    def test_candidates_exclude_other_known_positives(self):
        original = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        validation, test = np.zeros((3, 3), bool), np.zeros((3, 3), bool)
        validation[0, 1], test[2, 2] = True, True
        val_candidates, test_candidates = candidate_masks(original, validation, test)
        self.assertEqual(int(val_candidates.sum()), 6)
        self.assertEqual(int(test_candidates.sum()), 6)
        self.assertFalse(val_candidates[2, 2])
        self.assertFalse(test_candidates[0, 1])
        np.testing.assert_array_equal(val_candidates, (original == 0) | validation)
        with self.assertRaises(ValueError):
            candidate_masks(original, validation, validation)
        invalid = np.zeros((3, 3), bool)
        invalid[0, 2] = True
        with self.assertRaises(ValueError):
            candidate_masks(original, invalid, test)

    def test_both_axes_align_by_identifier(self):
        # Source rows b,a and columns a,b represent the same symmetric matrix.
        matrix = np.array([[.25, 1], [1, .25]])
        expected = np.array([[1, .25], [.25, 1]])
        np.testing.assert_array_equal(align_square(("b", "a"), ("a", "b"), matrix, ("a", "b")), expected)
        with self.assertRaises(ValueError):
            align_square(("a", "a"), ("a", "b"), matrix, ("a", "b"))
        with self.assertRaises(ValueError):
            align_square(("b", "a"), ("a", "b"), matrix, ("a", "c"))

    def test_reader_rejects_empty_ragged_duplicate_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.txt"
            for content in ("", "a b\nr 1\n", "a a\nr 1 2\n", "a b\nr 1 nan\n"):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    read_matrix(path)

    def test_custom_prefix_loads_transposes_and_records_source_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "custom_admat_dgc.txt").write_text("d1 d2\np1 1 0\np2 0 1\n")
            (root / "custom_simmat_dc.txt").write_text("d2 d1\nd1 .2 1\nd2 1 .4\n")
            (root / "custom_simmat_dg.txt").write_text("p1 p2\np1 1 .5\np2 .5 1\n")
            data = load_dataset(root, "custom")
            self.assertEqual(data.drugs, ("d1", "d2"))
            self.assertEqual(data.proteins, ("p1", "p2"))
            np.testing.assert_array_equal(data.interactions, np.eye(2))
            np.testing.assert_allclose(data.drug_similarity, [[0, .3], [.3, 0]])
            self.assertEqual(len(data.audit["source_files"]), 3)
            self.assertTrue(all(len(h) == 64 for h in data.audit["source_files"].values()))
            json.dumps(data.audit)
            for prefix in ("../custom", "a/b", ".", ""):
                with self.assertRaises(ValueError):
                    load_dataset(root, prefix)


if __name__ == "__main__":
    unittest.main()

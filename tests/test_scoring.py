"""Independent hand examples for the graph scoring rules (AI-assisted)."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from scoring import (drug_only, protein_only, neighbour_scores, two_sided,
                     fuse_scores, dti_only, target_degree, random_scores)


class ScoringTests(unittest.TestCase):
    def test_baseline_and_active_neighbour_denominators(self):
        graph = np.array([[0], [1], [0]])
        similarity = np.array([[1, .8, .2], [.8, 1, .4], [.2, .4, 1]])
        self.assertAlmostEqual(drug_only(graph, similarity)[0, 0], .8)
        enhanced, available = neighbour_scores(graph, similarity)
        self.assertEqual(enhanced[0, 0], 1)
        np.testing.assert_array_equal(available, [True, False, True])
        self.assertEqual(similarity[0, 0], 1)  # no caller mutation
        np.testing.assert_allclose(protein_only(graph.T, similarity), drug_only(graph, similarity).T)

    def test_power_selection_and_permutation_ties(self):
        graph = np.array([[0, 0], [1, 0], [0, 1]])
        similarity = np.array([[0, .5, .5], [.5, 0, .1], [.5, .1, 0]])
        ids = ["query", "z", "a"]
        scores, _ = neighbour_scores(graph, similarity, ids, k=1)
        np.testing.assert_allclose(scores[0], [0, 1])
        order = [2, 0, 1]
        permuted, _ = neighbour_scores(graph[order], similarity[np.ix_(order, order)],
                                       [ids[j] for j in order], k=1)
        np.testing.assert_allclose(permuted, scores[order])
        similarity[0, 1] = .8
        similarity[0, 2] = .2
        scores, _ = neighbour_scores(graph, similarity, power=2)
        np.testing.assert_allclose(scores[0], [16/17, 1/17])

    def test_reference_and_vectorized_equivalence(self):
        rng = np.random.default_rng(17)
        graph = (rng.random((8, 5)) > .7).astype(int)
        similarity = rng.random((8, 8))
        for k in (None, 1, 3, 20):
            fast, available = neighbour_scores(graph, similarity, k=k, power=2)
            slow, other_available = neighbour_scores(graph, similarity, k=k,
                                                      power=2, backend="reference")
            np.testing.assert_allclose(fast, slow, atol=1e-14)
            np.testing.assert_array_equal(available, other_available)
        all_scores, _ = neighbour_scores(graph, similarity)
        large_k, _ = neighbour_scores(graph, similarity, k=100)
        np.testing.assert_allclose(all_scores, large_k)

    def test_fusion_availability_and_endpoints(self):
        drug = np.full((2, 2), .8)
        protein = np.full((2, 2), .2)
        np.testing.assert_allclose(two_sided(drug, protein, .75), .65)
        scores = fuse_scores(drug, protein, .5, [True, False], [False, True])
        np.testing.assert_allclose(scores, [[.8, .5], [0, .2]])
        scores = fuse_scores(drug, protein, 1, [True, False], [True, True])
        np.testing.assert_allclose(scores, [[.8, .8], [0, 0]])
        scores = fuse_scores(drug, protein, 0, [True, True], [False, True])
        np.testing.assert_allclose(scores, [[0, .2], [0, .2]])

    def test_length_three_paths(self):
        graph = np.array([[1, 0], [1, 1]])
        # Only the absent upper-right edge closes a four-node path.
        np.testing.assert_array_equal(dti_only(graph), [[0, 1], [0, 0]])
        np.testing.assert_array_equal(target_degree(graph), [[2, 1], [2, 1]])
        np.testing.assert_array_equal(random_scores(graph.shape, 4), random_scores(graph.shape, 4))

    def test_zero_support_and_validation(self):
        graph = np.zeros((2, 3))
        scores, available = neighbour_scores(graph, np.ones((2, 2)))
        np.testing.assert_array_equal(scores, np.zeros((2, 3)))
        self.assertFalse(available.any())
        np.testing.assert_array_equal(drug_only(graph, np.zeros((2, 2))), graph)
        for kwargs in ({"k": 0}, {"k": True}, {"power": 0}, {"power": float("inf")},
                       {"ids": ["same", "same"]}, {"backend": "bad"}):
            with self.assertRaises(ValueError):
                neighbour_scores(graph, np.ones((2, 2)), **kwargs)
        for bad in (np.ones((2, 3)), np.array([[0, -1], [1, 0]]),
                    np.array([[0, float("nan")], [1, 0]])):
            with self.assertRaises(ValueError):
                drug_only(graph, bad)
        with self.assertRaises(ValueError):
            dti_only([[.5]])
        with self.assertRaises(ValueError):
            two_sided(graph, graph, 1.1)
        with self.assertRaises(ValueError):
            fuse_scores(graph, graph, .5, [True], [True, True, True])


if __name__ == "__main__":
    unittest.main()

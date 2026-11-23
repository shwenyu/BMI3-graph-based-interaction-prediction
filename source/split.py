"""Split known edges only. Node indices retain the original matrix coordinates."""
import numpy as np


def split_edges(adjacency, seed):
    edges = np.argwhere(adjacency == 1)
    if len(edges) < 10:
        raise ValueError("Need at least ten known edges for the 8:1:1 split")
    shuffled = np.random.default_rng(seed).permutation(edges)
    n = len(edges) // 10
    train = np.zeros_like(adjacency)
    validation = np.zeros_like(adjacency, dtype=bool)
    test = np.zeros_like(validation)
    for matrix, pairs in ((validation, shuffled[:n]), (test, shuffled[n:2*n]),
                          (train, shuffled[2*n:])):
        for drug, protein in pairs:
            matrix[drug, protein] = 1
    return train, validation, test

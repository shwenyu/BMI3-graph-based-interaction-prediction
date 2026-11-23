"""Split known edges only. Node indices retain the original matrix coordinates."""
import numpy as np


def split_edges(adjacency, seed):
    edges = np.argwhere(adjacency == 1)
    if len(edges) < 10:
        raise ValueError("Need at least ten known edges for the 8:1:1 split")
    shuffled = np.random.default_rng(seed).permutation(edges)
    held_out_count = len(edges) // 10
    validation_edges = shuffled[:held_out_count]
    test_edges = shuffled[held_out_count:2 * held_out_count]
    train_edges = shuffled[2 * held_out_count:]

    train = np.zeros_like(adjacency)
    validation = np.zeros(adjacency.shape, dtype=bool)
    test = np.zeros(adjacency.shape, dtype=bool)
    for drug, protein in train_edges:
        train[drug, protein] = 1
    for drug, protein in validation_edges:
        validation[drug, protein] = True
    for drug, protein in test_edges:
        test[drug, protein] = True
    return train, validation, test

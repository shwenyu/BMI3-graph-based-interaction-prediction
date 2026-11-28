"""Split known edges only. Node indices retain the original matrix coordinates."""
import numpy as np


def binary_matrix(values, name):
    matrix = np.asarray(values)
    if matrix.ndim != 2 or not all(matrix.shape) or not np.isin(matrix, [0, 1]).all():
        raise ValueError(f"{name} must be a nonempty binary 2-D matrix")
    return matrix.astype(bool)


def split_edges(adjacency, seed, validation_fraction=0.1, test_fraction=0.1):
    adjacency = binary_matrix(adjacency, "Adjacency")
    fractions = np.asarray([validation_fraction, test_fraction], dtype=float)
    if (not np.isfinite(fractions).all() or (fractions <= 0).any()
            or fractions.sum() >= 1):
        raise ValueError("Validation and test fractions must be positive and sum to less than one")
    edges = np.argwhere(adjacency == 1)
    validation_count = int(len(edges) * validation_fraction)
    test_count = int(len(edges) * test_fraction)
    if min(validation_count, test_count, len(edges) - validation_count - test_count) < 1:
        raise ValueError("Too few known edges for nonempty train, validation and test splits")
    shuffled = np.random.default_rng(seed).permutation(edges)
    validation_edges = shuffled[:validation_count]
    test_edges = shuffled[validation_count:validation_count + test_count]
    train_edges = shuffled[validation_count + test_count:]

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


def candidate_masks(original, validation, test):
    """Use original zeros as unlabeled background, never other held-out positives."""
    original = binary_matrix(original, "Original adjacency")
    validation = binary_matrix(validation, "Validation labels")
    test = binary_matrix(test, "Test labels")
    if original.shape != validation.shape or original.shape != test.shape:
        raise ValueError("Original, validation and test matrices must have identical shapes")
    if np.any(validation & test):
        raise ValueError("Validation and test positives must be disjoint")
    if np.any((validation | test) & ~original):
        raise ValueError("Held-out positives must come from original known edges")
    unknown = ~original
    return unknown | validation, unknown | test

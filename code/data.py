"""Read the labelled matrices, then hide known edges for validation and testing."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import numpy as np


def read_matrix(path):
    lines = [line.split() for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
             if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"Expected a header and matrix rows: {path}")
    columns = tuple(lines[0])
    rows = tuple(line[0] for line in lines[1:])
    if any(len(line) != len(columns) + 1 for line in lines[1:]):
        raise ValueError(f"Wrong number of entries: {path}")
    if len(set(rows)) != len(rows) or len(set(columns)) != len(columns):
        raise ValueError(f"Duplicate identifiers: {path}")
    matrix = np.array([line[1:] for line in lines[1:]], dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Nonfinite entries: {path}")
    return rows, columns, matrix


def align_square(rows, columns, matrix, ids):
    matrix = np.asarray(matrix, dtype=float)
    if (len(set(rows)) != len(rows) or len(set(columns)) != len(columns)
            or len(set(ids)) != len(ids)):
        raise ValueError("Duplicate identifiers in similarity matrix")
    if matrix.shape != (len(rows), len(columns)):
        raise ValueError("Similarity dimensions disagree with identifiers")
    if set(rows) != set(ids) or set(columns) != set(ids):
        raise ValueError("Similarity and interaction identifiers differ")
    row_index = {name: i for i, name in enumerate(rows)}
    column_index = {name: i for i, name in enumerate(columns)}
    return matrix[np.ix_([row_index[name] for name in ids],
                         [column_index[name] for name in ids])]


@dataclass
class Dataset:
    drugs: tuple
    proteins: tuple
    interactions: np.ndarray
    drug_similarity: np.ndarray
    protein_similarity: np.ndarray
    audit: dict


def load_dataset(root, code):
    if not isinstance(code, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", code):
        raise ValueError("Use a dataset prefix containing letters, digits, '_' or '-'")
    root = Path(root)
    dr, dc, sd = read_matrix(root / f"{code}_simmat_dc.txt")
    pr, pc, sp = read_matrix(root / f"{code}_simmat_dg.txt")
    rows, columns, graph = read_matrix(root / f"{code}_admat_dgc.txt")
    if set(rows) == set(dr) and set(columns) == set(pr):
        drugs, proteins, orientation = rows, columns, "drug_rows"
    elif set(columns) == set(dr) and set(rows) == set(pr):
        drugs, proteins, graph = columns, rows, graph.T
        orientation = "protein_rows; transposed"
    else:
        raise ValueError("Cannot match interaction matrix IDs to drugs and proteins")
    binary_matrix(graph, "Interaction matrix")
    sd = align_square(dr, dc, sd, drugs)
    sp = align_square(pr, pc, sp, proteins)
    audit = dict(dataset=code, drugs=len(drugs), proteins=len(proteins),
                 edges=int(graph.sum()), density=float(graph.mean()),
                 source_orientation=orientation, similarities={}, input_sha256={},
                 similarity_transform="average with transpose; zero diagonal")
    for name, matrix in (("drug", sd), ("protein", sp)):
        if (matrix < 0).any():
            raise ValueError("Similarities must be nonnegative")
        audit["similarities"][name] = dict(minimum=float(matrix.min()), maximum=float(matrix.max()),
            max_asymmetry=float(np.abs(matrix - matrix.T).max()),
            nonunit_diagonal=int((np.abs(np.diag(matrix) - 1) > 1e-6).sum()))
    # Average asymmetric inputs and stop a node voting for itself.
    sd, sp = sd / 2 + sd.T / 2, sp / 2 + sp.T / 2
    np.fill_diagonal(sd, 0)
    np.fill_diagonal(sp, 0)
    paths = [root / f"{code}_{suffix}.txt"
             for suffix in ("simmat_dc", "simmat_dg", "admat_dgc")]
    edge_path = root / f"bind_orfhsa_drug_{code}.txt"
    audit["edge_list_crosscheck"] = "not available"
    if edge_path.exists():
        listed = set()
        for line in edge_path.read_text().splitlines():
            if not line.strip():
                continue
            pair = line.split()
            if len(pair) != 2:
                raise ValueError(f"Expected a protein and drug on each line: {edge_path}")
            protein, drug = pair
            listed.add((protein.replace("hsa:", "hsa", 1), drug))
        actual = {(proteins[j], drugs[i]) for i, j in np.argwhere(graph == 1)}
        if listed != actual:
            raise ValueError("Source edge list disagrees with interaction matrix")
        audit["edge_list_crosscheck"] = "passed"
        paths.append(edge_path)
    audit["input_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return Dataset(drugs, proteins, graph.astype(np.uint8), sd, sp, audit)


def binary_matrix(values, name):
    matrix = np.asarray(values)
    if matrix.ndim != 2 or not all(matrix.shape) or not np.isin(matrix, [0, 1]).all():
        raise ValueError(f"{name} must be a nonempty binary 2-D matrix")
    return matrix.astype(bool)


def split_edges(adjacency, seed, validation_fraction=0.1, test_fraction=0.1):
    original = binary_matrix(adjacency, "Adjacency")
    fractions = np.array([validation_fraction, test_fraction], dtype=float)
    if not np.isfinite(fractions).all() or (fractions <= 0).any() or fractions.sum() >= 1:
        raise ValueError("Held-out fractions must be positive and sum to less than one")
    edges = np.argwhere(original)
    n_validation = int(len(edges) * validation_fraction)
    n_test = int(len(edges) * test_fraction)
    if min(n_validation, n_test, len(edges) - n_validation - n_test) < 1:
        raise ValueError("Too few edges for nonempty train, validation and test splits")
    shuffled = np.random.default_rng(seed).permutation(edges)
    train, validation, test = (np.zeros_like(original) for _ in range(3))
    # Only known positive edges are split; original zeros stay unlabeled.
    for i, j in shuffled[:n_validation]:
        validation[i, j] = True
    for i, j in shuffled[n_validation:n_validation + n_test]:
        test[i, j] = True
    for i, j in shuffled[n_validation + n_test:]:
        train[i, j] = True
    return train, validation, test


def candidate_masks(original, validation, test):
    original = binary_matrix(original, "Original adjacency")
    validation = binary_matrix(validation, "Validation labels")
    test = binary_matrix(test, "Test labels")
    if original.shape != validation.shape or original.shape != test.shape:
        raise ValueError("Original, validation and test matrices must have identical shapes")
    if np.any(validation & test) or np.any((validation | test) & ~original):
        raise ValueError("Held-out positives must be disjoint subsets of the known edges")
    return ~original | validation, ~original | test

"""Read labelled source matrices and align by identifiers, never by position alone."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


def read_matrix(path):
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    columns = lines[0].split()
    rows, values = [], []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split()
        rows.append(parts[0])
        values.append([float(x) for x in parts[1:]])
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (len(rows), len(columns)):
        raise ValueError(f"Malformed labelled matrix: {path}")
    if len(set(rows)) != len(rows) or len(set(columns)) != len(columns):
        raise ValueError(f"Duplicate identifiers: {path}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"Nonfinite values: {path}")
    return tuple(rows), tuple(columns), matrix


def align_square(rows, columns, matrix, ids):
    if set(rows) != set(ids) or set(columns) != set(ids):
        raise ValueError("Similarity and interaction identifier sets differ")
    ri, ci = {x: i for i, x in enumerate(rows)}, {x: i for i, x in enumerate(columns)}
    return matrix[np.ix_([ri[x] for x in ids], [ci[x] for x in ids])]


@dataclass
class Dataset:
    drugs: tuple
    proteins: tuple
    interactions: np.ndarray
    drug_similarity: np.ndarray
    protein_similarity: np.ndarray
    audit: dict


def load_dataset(root, code):
    root = Path(root)
    dr, dc, sd = read_matrix(root / f"{code}_simmat_dc.txt")
    pr, pc, sp = read_matrix(root / f"{code}_simmat_dg.txt")
    ar, ac, a = read_matrix(root / f"{code}_admat_dgc.txt")
    if set(ar) == set(dr) and set(ac) == set(pr):
        drugs, proteins = ar, ac
        source_orientation = "drug_rows"
    elif set(ac) == set(dr) and set(ar) == set(pr):
        drugs, proteins, a = ac, ar, a.T
        source_orientation = "protein_rows; transposed"
    else:
        raise ValueError("Cannot align adjacency identifiers with similarity matrices")
    if not np.isin(a, [0, 1]).all():
        raise ValueError("Interaction matrix must be binary")
    sd = align_square(dr, dc, sd, drugs)
    sp = align_square(pr, pc, sp, proteins)
    audit = dict(dataset=code, drugs=len(drugs), proteins=len(proteins),
                 edges=int(a.sum()), density=float(a.mean()),
                 source_orientation=source_orientation, similarities={})
    # Report raw anomalies before making a documented copy for scoring.
    for name, matrix in (("drug", sd), ("protein", sp)):
        if (matrix < 0).any():
            raise ValueError("Similarity weights must be nonnegative")
        audit["similarities"][name] = dict(
            minimum=float(matrix.min()), maximum=float(matrix.max()),
            max_asymmetry=float(np.abs(matrix - matrix.T).max()),
            nonunit_diagonal=int((np.abs(np.diag(matrix) - 1) > 1e-6).sum()))
    sd, sp = (sd + sd.T) / 2, (sp + sp.T) / 2
    np.fill_diagonal(sd, 0)
    np.fill_diagonal(sp, 0)
    edge_path = root / f"bind_orfhsa_drug_{code}.txt"
    if edge_path.exists():
        listed = {(protein.replace("hsa:", "hsa", 1), drug)
                  for protein, drug in (line.split() for line in edge_path.read_text().splitlines()
                                        if line.strip())}
        actual = {(proteins[j], drugs[i]) for i, j in np.argwhere(a == 1)}
        if listed != actual:
            raise ValueError("Source edge list disagrees with adjacency matrix")
        audit["edge_list_crosscheck"] = "passed"
    else:
        audit["edge_list_crosscheck"] = "not available"
    return Dataset(drugs, proteins, a.astype(np.uint8), sd, sp, audit)

"""Read labelled source matrices and align by identifiers, never by position alone."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import numpy as np


def read_matrix(path):
    lines = [line for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
             if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"Expected a header and at least one matrix row: {path}")
    columns = lines[0].split()
    rows, values = [], []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != len(columns) + 1:
            raise ValueError(f"Wrong number of matrix entries: {path}")
        rows.append(parts[0])
        try:
            values.append([float(x) for x in parts[1:]])
        except ValueError as error:
            raise ValueError(f"Nonnumeric matrix entry: {path}") from error
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (len(rows), len(columns)):
        raise ValueError(f"Malformed labelled matrix: {path}")
    if len(set(rows)) != len(rows) or len(set(columns)) != len(columns):
        raise ValueError(f"Duplicate identifiers: {path}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"Nonfinite values: {path}")
    return tuple(rows), tuple(columns), matrix


def align_square(rows, columns, matrix, ids):
    matrix = np.asarray(matrix, dtype=float)
    if (len(set(rows)) != len(rows) or len(set(columns)) != len(columns)
            or len(set(ids)) != len(ids)):
        raise ValueError("Duplicate identifiers in similarity alignment")
    if matrix.shape != (len(rows), len(columns)):
        raise ValueError("Similarity dimensions disagree with identifiers")
    if set(rows) != set(ids) or set(columns) != set(ids):
        raise ValueError("Similarity and interaction identifier sets differ")
    row_index = {identifier: i for i, identifier in enumerate(rows)}
    column_index = {identifier: j for j, identifier in enumerate(columns)}
    row_order = [row_index[identifier] for identifier in ids]
    column_order = [column_index[identifier] for identifier in ids]
    return matrix[np.ix_(row_order, column_order)]


@dataclass
class Dataset:
    drugs: tuple
    proteins: tuple
    interactions: np.ndarray
    drug_similarity: np.ndarray
    protein_similarity: np.ndarray
    audit: dict


def load_dataset(root, code):
    # Custom data can use any safe prefix, with the same three labelled filenames.
    if not isinstance(code, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", code):
        raise ValueError("Dataset prefix must contain only letters, digits, '_' or '-'")
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
                 source_orientation=source_orientation, similarities={},
                 source_files={}, similarity_transform="average with transpose; zero diagonal")
    for suffix in ("simmat_dc", "simmat_dg", "admat_dgc"):
        path = root / f"{code}_{suffix}.txt"
        audit["source_files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    # Report raw anomalies before making a documented copy for scoring.
    for name, matrix in (("drug", sd), ("protein", sp)):
        if (matrix < 0).any():
            raise ValueError("Similarity weights must be nonnegative")
        audit["similarities"][name] = dict(
            minimum=float(matrix.min()), maximum=float(matrix.max()),
            max_asymmetry=float(np.abs(matrix - matrix.T).max()),
            nonunit_diagonal=int((np.abs(np.diag(matrix) - 1) > 1e-6).sum()))
    sd, sp = sd / 2 + sd.T / 2, sp / 2 + sp.T / 2
    np.fill_diagonal(sd, 0)
    np.fill_diagonal(sp, 0)
    edge_path = root / f"bind_orfhsa_drug_{code}.txt"
    if edge_path.exists():
        audit["source_files"][edge_path.name] = hashlib.sha256(edge_path.read_bytes()).hexdigest()
        listed = set()
        for line in edge_path.read_text().splitlines():
            if line.strip():
                fields = line.split()
                if len(fields) != 2:
                    raise ValueError(f"Expected protein and drug on each edge-list line: {edge_path}")
                protein, drug = fields
                listed.add((protein.replace("hsa:", "hsa", 1), drug))

        actual = set()
        for i, j in np.argwhere(a == 1):
            actual.add((proteins[j], drugs[i]))
        if listed != actual:
            raise ValueError("Source edge list disagrees with adjacency matrix")
        audit["edge_list_crosscheck"] = "passed"
    else:
        audit["edge_list_crosscheck"] = "not available"
    audit["input_sha256"] = dict(audit["source_files"])
    return Dataset(drugs, proteins, a.astype(np.uint8), sd, sp, audit)

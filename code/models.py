"""Graph scoring methods. Rows are drugs, columns are proteins.

AI-assisted implementation; sources and method definitions are in README.md.
"""
import numpy as np


def validate_graph(visible):
    graph = np.asarray(visible, dtype=float)
    if graph.ndim != 2 or 0 in graph.shape:
        raise ValueError("Expected a nonempty drug-by-protein matrix")
    if not np.isfinite(graph).all() or not np.isin(graph, [0, 1]).all():
        raise ValueError("Visible interactions must be finite binary values")
    return graph


def similarity_weights(similarity, count):
    weights = np.array(similarity, dtype=float, copy=True)
    if weights.shape != (count, count):
        raise ValueError("Similarity dimensions do not match the graph")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Similarities must be finite and nonnegative")
    np.fill_diagonal(weights, 0)
    return weights


def check_fusion(drug_scores, protein_scores, alpha):
    drug = np.asarray(drug_scores, dtype=float)
    protein = np.asarray(protein_scores, dtype=float)
    if drug.ndim != 2 or drug.shape != protein.shape:
        raise ValueError("Both score matrices must have the same 2-D shape")
    if not np.isfinite(drug).all() or not np.isfinite(protein).all():
        raise ValueError("Scores must be finite")
    if not np.isscalar(alpha) or not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be between zero and one")
    return drug, protein


def drug_only(visible, similarity):
    """Weight every non-self neighbour, including neighbours with no edges."""
    graph = validate_graph(visible)
    weights = similarity_weights(similarity, graph.shape[0])
    total = weights.sum(axis=1, keepdims=True)
    return np.divide(weights @ graph, total, out=np.zeros_like(graph), where=total > 0)


def protein_only(visible, similarity):
    return drug_only(validate_graph(visible).T, similarity).T


def two_sided(drug_scores, protein_scores, alpha):
    drug, protein = check_fusion(drug_scores, protein_scores, alpha)
    return alpha * drug + (1 - alpha) * protein


def neighbour_scores(visible, similarity, ids=None, k=None, power=1.0,
                     backend="vectorized"):
    """Top-k voting among positive-weight neighbours with visible edges.

    Returns scores and one availability flag per query. Transpose the graph
    to apply the same rule to proteins. k=None uses all eligible neighbours.
    """
    graph = validate_graph(visible)
    count = graph.shape[0]
    weights = similarity_weights(similarity, count)
    ids = tuple(range(count)) if ids is None else tuple(ids)
    if len(ids) != count or len(set(ids)) != count:
        raise ValueError("Expected a unique ID for each graph row")
    if k is not None and (isinstance(k, (bool, np.bool_)) or
                         not isinstance(k, (int, np.integer)) or k <= 0):
        raise ValueError("k must be a positive integer or None")
    if not np.isscalar(power) or not np.isfinite(power) or power <= 0:
        raise ValueError("power must be finite and positive")
    if backend not in ("vectorized", "reference"):
        raise ValueError("backend must be vectorized or reference")

    active = graph.sum(axis=1) > 0
    scores = np.zeros_like(graph)
    available = np.zeros(count, dtype=bool)
    for query in range(count):
        candidates = np.flatnonzero(active & (weights[query] > 0))
        neighbours = sorted(candidates, key=lambda j: (-weights[query, j], ids[j]))
        if k is not None:
            neighbours = neighbours[:k]
        if not neighbours:
            continue
        chosen = weights[query, neighbours]
        # Scaling before the power avoids overflow; normalised weights are unchanged.
        chosen = (chosen / chosen.max()) ** power
        chosen /= chosen.sum()
        available[query] = True
        if backend == "vectorized":
            scores[query] = chosen @ graph[neighbours]
        else:
            for position, neighbour in enumerate(neighbours):
                for target in range(graph.shape[1]):
                    scores[query, target] += chosen[position] * graph[neighbour, target]
    return scores, available


def fuse_scores(drug_scores, protein_scores, alpha, drug_available=None,
                protein_available=None):
    """Use the fusion weights only where that side has an eligible neighbour."""
    drug, protein = check_fusion(drug_scores, protein_scores, alpha)
    n_drugs, n_proteins = drug.shape
    drug_available = (np.ones(n_drugs, dtype=bool) if drug_available is None
                      else np.asarray(drug_available, dtype=bool))
    protein_available = (np.ones(n_proteins, dtype=bool) if protein_available is None
                         else np.asarray(protein_available, dtype=bool))
    if drug_available.shape != (n_drugs,) or protein_available.shape != (n_proteins,):
        raise ValueError("Availability needs one flag per drug/protein")
    drug_weight = alpha * drug_available[:, None]
    protein_weight = (1 - alpha) * protein_available[None, :]
    total = drug_weight + protein_weight
    return np.divide(drug_weight * drug + protein_weight * protein, total,
                     out=np.zeros_like(drug), where=total > 0)


def dti_only(visible):
    """Count drug-protein-drug-protein paths without immediate backtracking."""
    graph = validate_graph(visible)
    walks = graph @ graph.T @ graph
    repeated = graph * (graph.sum(axis=1, keepdims=True) +
                        graph.sum(axis=0, keepdims=True) - 1)
    return np.maximum(walks - repeated, 0)


def target_degree(visible):
    graph = validate_graph(visible)
    return np.broadcast_to(graph.sum(axis=0), graph.shape).copy()


def random_scores(shape, seed):
    if len(shape) != 2 or any(not isinstance(n, (int, np.integer)) or n <= 0 for n in shape):
        raise ValueError("Expected a positive drug-by-protein shape")
    return np.random.default_rng(seed).random(shape)

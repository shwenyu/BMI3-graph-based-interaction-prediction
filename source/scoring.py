"""Simple graph scores; rows are drugs and columns are proteins.

AI-assisted implementation, 2026-11-28. See the repository provenance notes.
The functions take only the visible graph, never held-out labels.
"""
import numpy as np


def _visible_array(visible):
    graph = np.asarray(visible, dtype=float)
    if graph.ndim != 2 or 0 in graph.shape:
        raise ValueError("Expected a nonempty 2-D visible interaction matrix")
    if not np.isfinite(graph).all() or not np.isin(graph, [0, 1]).all():
        raise ValueError("Visible interactions must be finite binary values")
    return graph


def _similarity_array(similarity, count):
    weights = np.array(similarity, dtype=float, copy=True)
    if weights.shape != (count, count):
        raise ValueError("Similarity matrix must match the visible graph axis")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Similarities must be finite and nonnegative")
    np.fill_diagonal(weights, 0)
    return weights


def _score_pair(drug_scores, protein_scores, alpha):
    drug = np.asarray(drug_scores, dtype=float)
    protein = np.asarray(protein_scores, dtype=float)
    if drug.ndim != 2 or drug.shape != protein.shape:
        raise ValueError("Drug and protein scores must have the same 2-D shape")
    if not np.isfinite(drug).all() or not np.isfinite(protein).all():
        raise ValueError("Scores must be finite")
    if not np.isscalar(alpha) or not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be between zero and one")
    return drug, protein


def drug_only(visible, similarity):
    """All-neighbour voting, normalised by every non-self similarity weight.

    Neighbours with no visible edges still contribute to the denominator; this
    deliberately retains the original baseline rule for a fair comparison.
    """
    graph = _visible_array(visible)
    weights = _similarity_array(similarity, graph.shape[0])
    totals = weights.sum(axis=1, keepdims=True)
    return np.divide(weights @ graph, totals, out=np.zeros_like(graph), where=totals > 0)


def protein_only(visible, similarity):
    """Apply the same baseline to protein neighbours by swapping the axes."""
    return drug_only(_visible_array(visible).T, similarity).T


def neighbour_scores(visible, similarity, ids=None, k=None, power=1.0,
                     backend="vectorized"):
    """Vote using the most similar neighbours with at least one visible edge.

    k=None uses all positive-weight active neighbours. Similarity ties use IDs,
    not labels; the power is applied after selection. Returns the score matrix
    and one availability flag per query row. For protein voting, transpose the
    graph and then transpose the scores back.
    """
    graph = _visible_array(visible)
    weights = _similarity_array(similarity, graph.shape[0])
    count = graph.shape[0]
    identifiers = tuple(range(count)) if ids is None else tuple(ids)
    if len(identifiers) != count or len(set(identifiers)) != count:
        raise ValueError("Expected one unique identifier per graph row")
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
        neighbours = sorted(candidates, key=lambda j: (-weights[query, j], identifiers[j]))
        if k is not None:
            neighbours = neighbours[:k]
        if not neighbours:
            continue
        # Scale first: the normalised answer is unchanged, and large powers
        # cannot overflow even when source similarities exceed one.
        chosen = weights[query, neighbours]
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


def two_sided(drug_scores, protein_scores, alpha):
    """A fixed weighted average without availability correction."""
    drug, protein = _score_pair(drug_scores, protein_scores, alpha)
    return alpha * drug + (1 - alpha) * protein


def fuse_scores(drug_scores, protein_scores, alpha, drug_available=None,
                protein_available=None):
    """Renormalise the fusion weights over the sides available for each pair.

    Drug availability has one flag per drug; protein availability has one flag
    per protein. A side assigned weight zero is never used, including at the
    alpha endpoints. Pairs with no positive available weight receive zero.
    """
    drug, protein = _score_pair(drug_scores, protein_scores, alpha)
    n_drugs, n_proteins = drug.shape
    d_available = np.ones(n_drugs, dtype=bool) if drug_available is None else np.asarray(drug_available, dtype=bool)
    p_available = np.ones(n_proteins, dtype=bool) if protein_available is None else np.asarray(protein_available, dtype=bool)
    if d_available.shape != (n_drugs,) or p_available.shape != (n_proteins,):
        raise ValueError("Availability must have one flag per drug/protein")
    d_weight = alpha * d_available[:, None]
    p_weight = (1 - alpha) * p_available[None, :]
    totals = d_weight + p_weight
    return np.divide(d_weight * drug + p_weight * protein, totals,
                     out=np.zeros_like(drug), where=totals > 0)


def dti_only(visible):
    """Count non-backtracking length-three paths in the bipartite graph.

    A @ A.T @ A counts drug -> protein -> drug -> protein walks. Subtracting
    A*(drug_degree + protein_degree - 1) removes walks whose middle drug or
    first protein repeats an endpoint. For unseen pairs A=0, so the correction
    vanishes. This baseline uses no chemical or sequence similarities.
    """
    graph = _visible_array(visible)
    walks = graph @ graph.T @ graph
    backtracking = graph * (graph.sum(axis=1, keepdims=True) +
                            graph.sum(axis=0, keepdims=True) - 1)
    return np.maximum(walks - backtracking, 0)


def target_degree(visible):
    graph = _visible_array(visible)
    return np.broadcast_to(graph.sum(axis=0), graph.shape).copy()


def random_scores(shape, seed):
    if len(shape) != 2 or any(not isinstance(n, (int, np.integer)) or n <= 0 for n in shape):
        raise ValueError("Expected a positive drug-by-protein shape")
    return np.random.default_rng(seed).random(shape)

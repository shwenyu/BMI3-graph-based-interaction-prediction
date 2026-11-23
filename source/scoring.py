"""Linear similarity voting. Rows of visible are drugs, columns are targets."""
import numpy as np


def drug_only(visible, similarity):
    scores = np.zeros(visible.shape, dtype=float)
    for i, weights in enumerate(similarity):
        total = weights.sum()
        if total > 0:
            scores[i] = (weights[:, None] * visible).sum(axis=0) / total
    return scores


def protein_only(visible, similarity):
    # Swap axes so the same neighbour-voting rule operates on proteins.
    return drug_only(visible.T, similarity).T


def two_sided(drug_scores, protein_scores, alpha):
    return alpha * drug_scores + (1 - alpha) * protein_scores


def target_degree(visible):
    return np.broadcast_to(visible.sum(axis=0), visible.shape)


def random_scores(shape, seed):
    return np.random.default_rng(seed).random(shape)

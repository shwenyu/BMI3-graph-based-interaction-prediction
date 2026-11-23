"""Linear similarity voting. Rows of visible are drugs, columns are targets."""
import numpy as np


def drug_only(visible, similarity):
    scores = np.zeros(visible.shape, dtype=float)
    for drug, weights in enumerate(similarity):
        total_weight = weights.sum()
        if total_weight > 0:
            # Give each neighbor's row its similarity weight, then sum by target.
            weighted_edges = weights[:, None] * visible
            votes_by_target = weighted_edges.sum(axis=0)
            scores[drug] = votes_by_target / total_weight
    return scores


def protein_only(visible, similarity):
    # Swap axes so the same neighbour-voting rule operates on proteins.
    return drug_only(visible.T, similarity).T


def two_sided(drug_scores, protein_scores, alpha):
    return alpha * drug_scores + (1 - alpha) * protein_scores


def target_degree(visible):
    targets = visible.sum(axis=0)
    return np.broadcast_to(targets, visible.shape)


def random_scores(shape, seed):
    return np.random.default_rng(seed).random(shape)

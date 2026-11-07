"""Baseline implementations and explicit student-owned algorithm interfaces.

Only visible adjacency is accepted; original adjacency and hidden labels must
never be passed to these functions. All scores have shape (drugs, proteins).
"""
import numpy as np

METHODS = ("random", "target-degree", "drug-only", "protein-only", "two-sided")


def score(method, visible, drug_similarity, protein_similarity, seed):
    if method == "random":
        return np.random.default_rng(seed).random(visible.shape)
    if method == "target-degree":
        return np.broadcast_to(visible.sum(axis=0), visible.shape).astype(float).copy()
    if method in ("drug-only", "protein-only", "two-sided"):
        raise NotImplementedError(
            f"{method}: student implementation required; see docs/METHOD.md. "
            "Do not report infrastructure baseline output as the project algorithm.")
    raise ValueError(f"Unknown method: {method}")

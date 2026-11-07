"""Evaluator-owned edge hiding. The scorer receives visible adjacency only."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Split:
    visible: np.ndarray
    hidden: np.ndarray
    candidates: np.ndarray
    selected_entities: np.ndarray


def make_split(a, fraction, seed, mode="random"):
    a = np.asarray(a)
    if a.ndim != 2 or not np.isin(a, [0, 1]).all():
        raise ValueError("Expected a binary 2-D adjacency matrix")
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    hidden = np.zeros_like(a, dtype=bool)
    domain = np.ones_like(hidden)
    selected = np.array([], dtype=int)
    if mode == "random":
        edges = np.argwhere(a == 1)
        count = max(1, int(np.floor(len(edges) * fraction)))
        if count >= len(edges):
            raise ValueError("Need at least one visible and one hidden edge")
        chosen = edges[rng.choice(len(edges), count, replace=False)]
        hidden[chosen[:, 0], chosen[:, 1]] = True
    elif mode in ("drug", "protein"):
        size = a.shape[0 if mode == "drug" else 1]
        count = max(1, int(np.floor(size * fraction)))
        if count >= size:
            raise ValueError("Need at least one retained entity")
        selected = np.sort(rng.choice(size, count, replace=False))
        domain[:] = False
        if mode == "drug":
            domain[selected, :] = True
        else:
            domain[:, selected] = True
        hidden = domain & (a == 1)
        if not hidden.any():
            raise ValueError("Selected entities have no known edges; use another documented seed")
    else:
        raise ValueError(f"Unknown split mode: {mode}")
    visible = a.copy()
    visible[hidden] = 0
    candidates = domain & (visible == 0)
    assert not np.any(hidden & (visible == 1))
    assert np.array_equal((visible == 1) | hidden, a == 1)
    return Split(visible, hidden, candidates, selected)

"""Ranking metrics against hidden recorded edges, not biological truth."""
import numpy as np


def ranking_metrics(labels, scores):
    labels, scores = np.asarray(labels, dtype=bool), np.asarray(scores, dtype=float)
    if labels.shape != scores.shape or labels.ndim != 1 or not len(labels):
        raise ValueError("Expected equally sized nonempty 1-D arrays")
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite prediction scores")
    order = np.argsort(-scores, kind="stable")
    y, s = labels[order], scores[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s) - 1]
    tp = np.cumsum(y)[ends].astype(float)
    fp = (ends + 1) - tp
    positives, negatives = int(labels.sum()), int((~labels).sum())
    precision = tp / (ends + 1)
    recall = tp / positives if positives else np.zeros_like(tp)
    # AP groups all equal scores at one threshold; never breaks ties using labels.
    ap = float(np.sum(np.diff(np.r_[0, recall]) * precision)) if positives else None
    fpr = fp / negatives if negatives else np.zeros_like(fp)
    auc = (float(np.sum(np.diff(np.r_[0, fpr]) *
           (np.r_[0, recall][:-1] + recall) / 2)) if positives and negatives else None)
    return dict(ap=ap, auroc=auc, positives=positives, background=negatives,
                precision=precision, recall=recall, fpr=fpr, thresholds=s[ends])


def evaluate(scores, hidden, candidates, proteins, top_n=10):
    if scores.shape != hidden.shape or candidates.shape != hidden.shape:
        raise ValueError("Scores, labels and candidates must have identical shapes")
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if not np.all(candidates[hidden]):
        raise ValueError("Every hidden edge must be in the candidate set")
    pooled = ranking_metrics(hidden[candidates], scores[candidates])
    curves = {k: pooled.pop(k) for k in ("precision", "recall", "fpr", "thresholds")}
    rows = []
    for i in range(scores.shape[0]):
        js = np.flatnonzero(candidates[i])
        # Deterministic identifier tie break for top-N only; independent of labels.
        ranked = sorted(js, key=lambda j: (-scores[i, j], proteins[j]))
        top = ranked[:top_n]
        positives = int(hidden[i].sum())
        hits = int(hidden[i, top].sum())
        rows.append(dict(drug_index=i, candidates=len(js), hidden=positives,
                         hits=hits, precision=hits / len(top) if top else None,
                         recall=hits / positives if positives else None))
    eligible = [row for row in rows if row["hidden"] > 0]
    pooled.update(macro_top_precision=float(np.mean([r["precision"] for r in eligible])),
                  macro_top_recall=float(np.mean([r["recall"] for r in eligible])),
                  eligible_drugs=len(eligible), drugs_without_hidden=len(rows)-len(eligible),
                  top_n=top_n)
    return pooled, rows, curves

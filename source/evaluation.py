"""Ranking metrics against hidden recorded edges, not biological truth."""
import numpy as np


def ranking_metrics(labels, scores):
    labels, scores = np.asarray(labels, dtype=bool), np.asarray(scores, dtype=float)
    if labels.shape != scores.shape or labels.ndim != 1 or not len(labels):
        raise ValueError("Expected equally sized nonempty 1-D arrays")
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite prediction scores")
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    # Evaluate only after including every candidate with the same score.
    score_changes = np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:])
    group_ends = np.append(score_changes, len(sorted_scores) - 1)
    predicted_count = group_ends + 1
    true_positives = np.cumsum(sorted_labels)[group_ends].astype(float)
    false_positives = predicted_count - true_positives
    positives, negatives = int(labels.sum()), int((~labels).sum())
    precision = true_positives / predicted_count
    recall = np.zeros_like(true_positives)
    ap = None
    if positives:
        recall = true_positives / positives
        recall_gains = np.diff(np.insert(recall, 0, 0))
        ap = float(np.sum(recall_gains * precision))

    fpr = np.zeros_like(false_positives)
    auc = None
    if negatives:
        fpr = false_positives / negatives
    if positives and negatives:
        # Each trapezoid has width delta FPR and height average recall.
        widths = np.diff(np.insert(fpr, 0, 0))
        previous_recall = np.insert(recall[:-1], 0, 0)
        auc = float(np.sum(widths * (previous_recall + recall) / 2))

    return dict(ap=ap, auroc=auc, positives=positives, background=negatives,
                precision=precision, recall=recall, fpr=fpr, thresholds=sorted_scores[group_ends])


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
        candidate_targets = np.flatnonzero(candidates[i])
        # Deterministic identifier tie break for top-N only; independent of labels.
        ranked = sorted(candidate_targets, key=lambda j: (-float(scores[i, j]), proteins[j]))
        top = ranked[:top_n]
        positives = int(hidden[i].sum())
        hits = int(hidden[i, top].sum())
        rows.append(dict(drug_index=i, candidates=len(candidate_targets), hidden=positives,
                         hits=hits, precision=hits / len(top) if top else None,
                         recall=hits / positives if positives else None))
    eligible = [row for row in rows if row["hidden"] > 0]
    pooled.update(macro_top_precision=float(np.mean([r["precision"] for r in eligible])),
                  macro_top_recall=float(np.mean([r["recall"] for r in eligible])),
                  eligible_drugs=len(eligible), drugs_without_hidden=len(rows)-len(eligible),
                  top_n=top_n)
    return pooled, rows, curves

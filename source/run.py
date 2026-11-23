"""Run from the repository root: python source/run.py --dataset nr --seed 42"""
import argparse
import csv
from pathlib import Path
import numpy as np
from data import load_dataset
from split import split_edges
from evaluation import evaluate, ranking_metrics
from scoring import drug_only, protein_only, two_sided, random_scores, target_degree


def save_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nr", help="Input filename prefix")
    parser.add_argument("--input", type=Path, default=Path("input"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data = load_dataset(args.input, args.dataset)
    train, validation, test = split_edges(data.interactions, args.seed)
    unknown = data.interactions == 0
    out = args.output / f"{args.dataset}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=False)
    edges = []
    for name, matrix in (("train", train), ("validation", validation), ("test", test)):
        for i, j in np.argwhere(matrix):
            edges.append(dict(split=name, drug=data.drugs[i], protein=data.proteins[j]))
    save_csv(out / "edges.csv", edges)

    # Tune the weight with the same visible graph and candidates for every trial.
    drug_scores = drug_only(train, data.drug_similarity)
    protein_scores = protein_only(train, data.protein_similarity)
    validation_candidates = unknown | validation
    validation_labels = validation[validation_candidates]
    trials = []
    # Prefer equal weights on an exact tie.
    for alpha in (0.5, 0.25, 0.75):
        scores = two_sided(drug_scores, protein_scores, alpha)
        ap = ranking_metrics(validation_labels, scores[validation_candidates])["ap"]
        trials.append(dict(alpha=alpha, validation_ap=ap))
    best = max(trials, key=lambda row: row["validation_ap"])["alpha"]
    save_csv(out / "validation.csv", trials)

    # Restore validation edges after choosing the weight. Test edges stay hidden.
    visible = train | validation
    test_candidates = unknown | test
    drug_scores = drug_only(visible, data.drug_similarity)
    protein_scores = protein_only(visible, data.protein_similarity)
    methods = {"random": random_scores(visible.shape, args.seed + 1),
               "target-degree": target_degree(visible), "drug-only": drug_scores,
               "protein-only": protein_scores,
               "two-sided": two_sided(drug_scores, protein_scores, best)}
    summary, details = [], []
    for method, scores in methods.items():
        metrics, rows, _ = evaluate(scores, test, test_candidates, data.proteins)
        summary.append(dict(method=method, alpha=best if method == "two-sided" else "", **metrics))
        for row in rows:
            details.append(dict(method=method, drug=data.drugs[row["drug_index"]], **row))
    save_csv(out / "test.csv", summary)
    save_csv(out / "per_drug.csv", details)
    print(f"{out}: train={train.sum()}, validation={validation.sum()}, test={test.sum()}, alpha={best}")


if __name__ == "__main__":
    main()

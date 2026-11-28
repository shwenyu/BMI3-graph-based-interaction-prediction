"""Rank original unrecorded pairs after benchmarking; these are not confirmed DTIs."""
import argparse
import json
from pathlib import Path

import numpy as np
from data import load_dataset
from scoring import neighbour_scores, fuse_scores
from run import ROOT, save_csv, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input", type=Path, default=ROOT / "input")
    parser.add_argument("--settings", type=Path, required=True, help="A benchmark run.json")
    parser.add_argument("--output", type=Path, required=True, help="A new output directory")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args(argv)
    if args.top_n <= 0:
        parser.error("top-n must be positive")
    receipt = json.loads(args.settings.read_text())
    if receipt["dataset"] != args.dataset:
        parser.error("Settings dataset does not match --dataset")
    data = load_dataset(args.input, args.dataset)
    if data.audit["input_sha256"] != receipt["data"]["input_sha256"]:
        parser.error("Data hashes differ from the benchmark; do not silently reuse settings")
    setting = receipt["selected"]["two-sided-topk"]
    drug, ds = neighbour_scores(data.interactions, data.drug_similarity, data.drugs,
                                setting["k"], setting["power"])
    protein, ps = neighbour_scores(data.interactions.T, data.protein_similarity, data.proteins,
                                   setting["k"], setting["power"])
    scores = fuse_scores(drug, protein.T, setting["alpha"], ds, ps)
    rows = []
    for i, drug_id in enumerate(data.drugs):
        candidates = np.flatnonzero(data.interactions[i] == 0)
        ranked = sorted(candidates, key=lambda j: (-float(scores[i, j]), data.proteins[j]))
        for rank, j in enumerate(ranked[:args.top_n], 1):
            rows.append(dict(drug=drug_id, protein=data.proteins[j], rank=rank,
                             score=float(scores[i, j]), status="unlabeled"))
    args.output.mkdir(parents=True, exist_ok=False)
    if rows:
        save_csv(args.output / "predictions.csv", rows)
    else:
        (args.output / "predictions.csv").write_text("drug,protein,rank,score,status\n")
    write_json(args.output / "prediction.json", dict(dataset=args.dataset, selected=setting,
        settings_file=str(args.settings.resolve()), input_sha256=data.audit["input_sha256"],
        pairs=len(rows), trained_on="all recorded positives, including benchmark holdouts",
        interpretation="Prioritised unlabeled pairs, not probabilities or confirmed interactions. Do not evaluate benchmark holdouts using this full graph."))
    print(f"Wrote {len(rows)} unrecorded candidate predictions to {args.output}")


if __name__ == "__main__":
    main()

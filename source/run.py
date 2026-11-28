"""Run a reproducible hidden-edge benchmark; original zero entries are unlabeled."""
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from data import load_dataset
from evaluation import evaluate, ranking_metrics
from scoring import (drug_only, protein_only, two_sided, neighbour_scores,
                     fuse_scores, dti_only, random_scores, target_degree)
from split import split_edges, candidate_masks

ROOT = Path(__file__).resolve().parent.parent
METHODS = ("random", "target-degree", "dti-only", "drug-only", "protein-only",
           "two-sided", "two-sided-topk")
METRICS = ("ap", "auroc", "macro_top_precision", "macro_top_recall", "score_seconds")


def save_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def git_receipt():
    def git(*args):
        try:
            result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
        except FileNotFoundError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None
    status = git("status", "--porcelain")
    return dict(commit=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"),
                dirty=None if status is None else bool(status))


def timed(function, repeats=1):
    durations = []
    for _ in range(repeats):
        start = perf_counter()
        value = function()
        durations.append(perf_counter() - start)
    return value, float(np.median(durations))


def tune(train, data, validation, candidates, alphas, ks, powers):
    """Every trial sees exactly the same training graph and validation candidates."""
    labels = validation[candidates]
    drug = drug_only(train, data.drug_similarity)
    protein = protein_only(train, data.protein_similarity)
    trials = []
    for alpha in alphas:
        scores = two_sided(drug, protein, alpha)
        ap = ranking_metrics(labels, scores[candidates])["ap"]
        trials.append(dict(method="two-sided", k="", power=1.0, alpha=alpha, validation_ap=ap))
    for k in ks:
        for power in powers:
            drug, ds = neighbour_scores(train, data.drug_similarity, data.drugs, k, power)
            protein_t, ps = neighbour_scores(train.T, data.protein_similarity, data.proteins, k, power)
            for alpha in alphas:
                scores = fuse_scores(drug, protein_t.T, alpha, ds, ps)
                ap = ranking_metrics(labels, scores[candidates])["ap"]
                trials.append(dict(method="two-sided-topk", k=k, power=power,
                                   alpha=alpha, validation_ap=ap))
    # Stable trial order is the tie rule, specified before opening test results.
    selected = {}
    for method in ("two-sided", "two-sided-topk"):
        choices = [row for row in trials if row["method"] == method]
        selected[method] = dict(max(choices, key=lambda row: row["validation_ap"]))
    return selected, trials


def score_methods(visible, data, selected, seed, repeats):
    """Times include each method's own scoring and neighbour preparation."""
    scores, times = {}, {}
    scores["random"], times["random"] = timed(lambda: random_scores(visible.shape, seed + 1), repeats)
    scores["target-degree"], times["target-degree"] = timed(lambda: target_degree(visible), repeats)
    scores["dti-only"], times["dti-only"] = timed(lambda: dti_only(visible), repeats)
    scores["drug-only"], times["drug-only"] = timed(lambda: drug_only(visible, data.drug_similarity), repeats)
    scores["protein-only"], times["protein-only"] = timed(lambda: protein_only(visible, data.protein_similarity), repeats)
    alpha = selected["two-sided"]["alpha"]
    scores["two-sided"], times["two-sided"] = timed(
        lambda: two_sided(drug_only(visible, data.drug_similarity),
                          protein_only(visible, data.protein_similarity), alpha), repeats)
    setting = selected["two-sided-topk"]
    def enhanced():
        drug, ds = neighbour_scores(visible, data.drug_similarity, data.drugs,
                                    setting["k"], setting["power"])
        protein, ps = neighbour_scores(visible.T, data.protein_similarity, data.proteins,
                                       setting["k"], setting["power"])
        return fuse_scores(drug, protein.T, setting["alpha"], ds, ps)
    scores["two-sided-topk"], times["two-sided-topk"] = timed(enhanced, repeats)
    return scores, times


def compare_backends(visible, data, setting, repeats):
    rows = []
    for side, matrix, similarity, ids in (
        ("drug", visible, data.drug_similarity, data.drugs),
        ("protein", visible.T, data.protein_similarity, data.proteins)):
        results = {}
        for backend in ("reference", "vectorized"):
            result, seconds = timed(lambda: neighbour_scores(matrix, similarity, ids,
                                      setting["k"], setting["power"], backend), repeats)
            results[backend] = result
            rows.append(dict(side=side, backend=backend, k=setting["k"],
                             power=setting["power"], seconds=seconds, repeats=repeats))
        np.testing.assert_allclose(results["reference"][0], results["vectorized"][0], rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(results["reference"][1], results["vectorized"][1])
    return rows


def benchmark_one(args, code, seed):
    directory = args.output / f"{code}_seed{seed}"
    data = load_dataset(args.input, code)
    train, validation, test = split_edges(data.interactions, seed,
                                         args.validation_fraction, args.test_fraction)
    vc, tc = candidate_masks(data.interactions, validation, test)
    directory.mkdir(parents=True, exist_ok=False)
    edges = []
    for name, mask in (("train", train), ("validation", validation), ("test", test)):
        for i, j in np.argwhere(mask):
            edges.append(dict(split=name, drug=data.drugs[i], protein=data.proteins[j]))
    save_csv(directory / "edges.csv", edges)
    start = perf_counter()
    selected, trials = tune(train, data, validation, vc, args.alphas, args.ks, args.powers)
    tuning_seconds = perf_counter() - start
    save_csv(directory / "validation.csv", trials)
    visible = (train.astype(bool) | validation).astype(np.uint8)
    scores, times = score_methods(visible, data, selected, seed, args.timing_repeats)
    summary, details = [], []
    cold_drugs = visible.sum(axis=1) == 0
    cold_proteins = visible.sum(axis=0) == 0
    for method in METHODS:
        start = perf_counter()
        metrics, rows, _ = evaluate(scores[method], test, tc, data.proteins, args.top_n)
        evaluation_seconds = perf_counter() - start
        setting = selected.get(method, {})
        summary.append(dict(dataset=code, seed=seed, method=method,
            alpha=setting.get("alpha", ""), k=setting.get("k", ""),
            power=setting.get("power", ""), **metrics,
            score_seconds=times[method], evaluation_seconds=evaluation_seconds))
        for row in rows:
            details.append(dict(method=method, drug=data.drugs[row["drug_index"]],
                                cold_drug=bool(cold_drugs[row["drug_index"]]), **row))
    save_csv(directory / "test.csv", summary)
    save_csv(directory / "per_drug.csv", details)
    if args.compare_backends:
        save_csv(directory / "backend_timings.csv", compare_backends(
            visible, data, selected["two-sided-topk"], args.timing_repeats))
    settings = {key: value for key, value in vars(args).items() if key not in ("input", "output")}
    settings.update(input=str(args.input.resolve()), output=str(args.output.resolve()))
    source_digest = hashlib.sha256()
    for path in sorted((ROOT / "source").glob("*.py")):
        source_digest.update(path.name.encode())
        source_digest.update(path.read_bytes())
    receipt = dict(dataset=code, seed=seed, selected=selected, config=settings,
        data=data.audit, git=git_receipt(), source_sha256=source_digest.hexdigest(),
        python=sys.version, numpy=np.__version__, platform=platform.platform(),
        thread_environment={name: os.environ.get(name) for name in
                            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                             "VECLIB_MAXIMUM_THREADS")},
        finished_utc=datetime.now(timezone.utc).isoformat(), tuning_seconds=tuning_seconds,
        counts=dict(train=int(train.sum()), validation=int(validation.sum()), test=int(test.sum()),
                    test_candidates=int(tc.sum()), background=int((data.interactions == 0).sum()),
                    cold_drugs=int(cold_drugs.sum()), cold_proteins=int(cold_proteins.sum()),
                    test_pairs_with_cold_drug=int(test[cold_drugs].sum()),
                    test_pairs_with_cold_protein=int(test[:, cold_proteins].sum())),
        interpretation="Known-edge recovery against unlabeled background; random edge hiding, not entity holdout.")
    write_json(directory / "run.json", receipt)
    print(f"{code} seed={seed}: train={train.sum()}, val={validation.sum()}, test={test.sum()}", flush=True)
    return summary


def aggregate(rows):
    summary = []
    groups = sorted({(row["dataset"], row["method"]) for row in rows})
    for dataset, method in groups:
        group = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
        result = dict(dataset=dataset, method=method, runs=len(group), top_n=group[0]["top_n"])
        for metric in METRICS:
            values = [r[metric] for r in group if r[metric] is not None]
            result[metric + "_mean"] = float(np.mean(values)) if values else None
            result[metric + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            result[metric + "_n"] = len(values)
        summary.append(result)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", default=["e", "ic", "gpcr", "nr"])
    parser.add_argument("--seed", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--input", type=Path, default=ROOT / "input")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "benchmark")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.5, 0.25, 0.75])
    parser.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20, 50])
    parser.add_argument("--powers", nargs="+", type=float, default=[1.0, 2.0])
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--compare-backends", action="store_true")
    args = parser.parse_args(argv)
    if any(not np.isfinite(a) or not 0 <= a <= 1 for a in args.alphas):
        parser.error("alphas must be finite and between zero and one")
    if any(k <= 0 for k in args.ks) or any(not np.isfinite(p) or p <= 0 for p in args.powers):
        parser.error("ks and powers must be positive and finite")
    if args.top_n <= 0 or args.timing_repeats <= 0 or any(s < 0 for s in args.seed):
        parser.error("top-n/repeats must be positive and seeds nonnegative")
    if (not 0 < args.validation_fraction < 1 or not 0 < args.test_fraction < 1
            or args.validation_fraction + args.test_fraction >= 1):
        parser.error("held-out fractions must be positive with sum less than one")
    if len(set(args.dataset)) != len(args.dataset) or len(set(args.seed)) != len(args.seed):
        parser.error("dataset and seed lists must not contain duplicates")
    return args


def main(argv=None):
    args = parse_args(argv)
    # Never overwrite a run or silently mix old and new experiment summaries.
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"Output must be new or empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for code in args.dataset:
        for seed in args.seed:
            rows.extend(benchmark_one(args, code, seed))
    save_csv(args.output / "all_runs.csv", rows)
    save_csv(args.output / "summary.csv", aggregate(rows))
    print(f"Saved {len(rows)} method evaluations to {args.output}")


if __name__ == "__main__":
    main()

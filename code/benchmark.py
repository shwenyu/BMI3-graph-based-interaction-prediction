"""One experiment: split known edges, tune on validation, test, then plot."""
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from data import ROOT, project_path, recorded_path, load_dataset, split_edges, candidate_masks
from models import (drug_only, protein_only, two_sided, neighbour_scores,
                    fuse_scores, dti_only, target_degree, random_scores)

METHODS = ("random", "target-degree", "dti-only", "drug-only", "protein-only",
           "two-sided", "two-sided-topk")
METRICS = ("ap", "auroc", "macro_top_precision", "macro_top_recall", "score_seconds")


# 1. Check a ranking. Equal scores are evaluated as a group.
def ranking_metrics(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or labels.shape != scores.shape or not labels.size:
        raise ValueError("Labels and scores must be equally sized nonempty vectors")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Labels must be binary and scores finite")
    order = np.argsort(-scores, kind="stable")
    labels, scores = labels[order].astype(bool), scores[order]
    ends = np.append(np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1)
    tp = np.cumsum(labels)[ends].astype(float)
    fp = ends + 1 - tp
    positives, background = int(labels.sum()), int((~labels).sum())
    ap, auroc = None, None
    if positives:
        recall = tp / positives
        ap = float(np.sum(np.diff(np.r_[0, recall]) * (tp / (ends + 1))))
        if background:
            auroc = float(np.sum(np.diff(np.r_[0, fp / background]) *
                                 (np.r_[0, recall[:-1]] + recall) / 2))
    return dict(ap=ap, auroc=auroc, positives=positives, background=background)


def evaluate(scores, hidden, candidates, proteins, top_n=10):
    scores, hidden, candidates = np.asarray(scores), np.asarray(hidden), np.asarray(candidates)
    if scores.ndim != 2 or not all(scores.shape) or scores.shape != hidden.shape or scores.shape != candidates.shape:
        raise ValueError("Scores, labels and candidates must be matching nonempty matrices")
    if not np.isfinite(scores).all() or not np.isin(hidden, [0, 1]).all() or not np.isin(candidates, [0, 1]).all():
        raise ValueError("Scores must be finite; labels and candidates must be binary")
    if isinstance(top_n, (bool, np.bool_)) or not isinstance(top_n, (int, np.integer)) or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    if len(proteins) != scores.shape[1] or len(set(proteins)) != len(proteins):
        raise ValueError("Protein IDs must be unique and match score columns")
    hidden, candidates = hidden.astype(bool), candidates.astype(bool)
    if np.any(hidden & ~candidates):
        raise ValueError("Every hidden positive must be a candidate")
    result = ranking_metrics(hidden[candidates], scores[candidates])
    details = []
    for i in range(len(scores)):
        targets = np.flatnonzero(candidates[i])
        top = sorted(targets, key=lambda j: (-float(scores[i, j]), proteins[j]))[:top_n]
        positives, hits = int(hidden[i].sum()), int(hidden[i, top].sum())
        details.append(dict(drug_index=i, candidates=len(targets), hidden=positives,
                            hits=hits, precision=hits / len(top) if top else None,
                            recall=hits / positives if positives else None))
    eligible = [r for r in details if r["hidden"] > 0]
    result.update(macro_top_precision=float(np.mean([r["precision"] for r in eligible])) if eligible else None,
                  macro_top_recall=float(np.mean([r["recall"] for r in eligible])) if eligible else None,
                  eligible_drugs=len(eligible), drugs_without_hidden=len(details) - len(eligible), top_n=top_n)
    return result, details


# 2. Choose settings using training scores and validation labels only.
def tune(train, data, validation, candidates, alphas, ks, powers):
    trials = []
    drug = drug_only(train, data.drug_similarity)
    protein = protein_only(train, data.protein_similarity)
    for alpha in alphas:
        scores = two_sided(drug, protein, alpha)
        ap = ranking_metrics(validation[candidates], scores[candidates])["ap"]
        trials.append(dict(method="two-sided", k="", power=1.0, alpha=alpha, validation_ap=ap))
    for k in ks:
        for power in powers:
            drug, ds = neighbour_scores(train, data.drug_similarity, data.drugs, k, power)
            protein, ps = neighbour_scores(train.T, data.protein_similarity, data.proteins, k, power)
            for alpha in alphas:
                scores = fuse_scores(drug, protein.T, alpha, ds, ps)
                ap = ranking_metrics(validation[candidates], scores[candidates])["ap"]
                trials.append(dict(method="two-sided-topk", k=k, power=power, alpha=alpha, validation_ap=ap))
    # max keeps the first trial on an exact tie: the CLI grid order is the rule.
    selected = {method: max((r for r in trials if r["method"] == method), key=lambda r: r["validation_ap"])
                for method in ("two-sided", "two-sided-topk")}
    return selected, trials


def topk_scores(visible, data, setting, backend="vectorized"):
    drug, ds = neighbour_scores(visible, data.drug_similarity, data.drugs,
                                setting["k"], setting["power"], backend)
    protein, ps = neighbour_scores(visible.T, data.protein_similarity, data.proteins,
                                   setting["k"], setting["power"], backend)
    return fuse_scores(drug, protein.T, setting["alpha"], ds, ps)


def timed(function, repeats):
    seconds = []
    for _ in range(repeats):
        start = perf_counter()
        value = function()
        seconds.append(perf_counter() - start)
    return value, float(np.median(seconds))


# 3. All methods use the same visible graph and test candidates.
def score_methods(visible, data, selected, seed, repeats):
    alpha = selected["two-sided"]["alpha"]
    functions = (
        lambda: random_scores(visible.shape, seed + 1),
        lambda: target_degree(visible),
        lambda: dti_only(visible),
        lambda: drug_only(visible, data.drug_similarity),
        lambda: protein_only(visible, data.protein_similarity),
        lambda: two_sided(drug_only(visible, data.drug_similarity),
                          protein_only(visible, data.protein_similarity), alpha),
        lambda: topk_scores(visible, data, selected["two-sided-topk"]),
    )
    scores, durations = {}, {}
    for method, function in zip(METHODS, functions):
        scores[method], durations[method] = timed(function, repeats)
    return scores, durations


def compare_backends(visible, data, setting, repeats):
    results, rows = {}, []
    for backend in ("reference", "vectorized"):
        results[backend], seconds = timed(lambda: topk_scores(visible, data, setting, backend), repeats)
        rows.append(dict(backend=backend, seconds=seconds, repeats=repeats))
    np.testing.assert_allclose(results["reference"], results["vectorized"], rtol=1e-12, atol=1e-12)
    return rows


def aggregate(rows):
    summary = []
    for dataset, method in sorted({(r["dataset"], r["method"]) for r in rows}):
        group = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
        item = dict(dataset=dataset, method=method, runs=len(group), top_n=group[0]["top_n"])
        for metric in METRICS:
            values = [r[metric] for r in group if r[metric] is not None]
            item[metric + "_mean"] = float(np.mean(values)) if values else None
            item[metric + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            item[metric + "_n"] = len(values)
        summary.append(item)
    return summary


def save_csv(path, rows, fields=None):
    with project_path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git_state():
    def read(*args):
        try:
            p = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
            return p.stdout.strip() if p.returncode == 0 else None
        except FileNotFoundError:
            return None
    status = read("status", "--porcelain")
    return dict(commit=read("rev-parse", "HEAD"), branch=read("branch", "--show-current"),
                dirty=None if status is None else bool(status))


# Optional checks use the same model functions and leave the original benchmark alone.
STAGES = ("baseline", "active", "topk", "power", "availability")


def ablation_scores(visible, data, setting):
    stage, alpha = setting["stage"], setting["alpha"]
    if stage == "baseline":
        return two_sided(drug_only(visible, data.drug_similarity),
                         protein_only(visible, data.protein_similarity), alpha)
    k = None if stage == "active" else setting["k"]
    drug, ds = neighbour_scores(visible, data.drug_similarity, data.drugs, k, setting["power"])
    protein, ps = neighbour_scores(visible.T, data.protein_similarity, data.proteins, k, setting["power"])
    if stage == "availability":
        return fuse_scores(drug, protein.T, alpha, ds, ps)
    return two_sided(drug, protein.T, alpha)


def tune_ablation(train, data, validation, candidates, alphas, ks, powers):
    trials = []
    def try_weights(stage, k, power, drug, protein, ds=None, ps=None):
        for alpha in alphas:
            scores = (fuse_scores(drug, protein, alpha, ds, ps) if stage == "availability"
                      else two_sided(drug, protein, alpha))
            ap = ranking_metrics(validation[candidates], scores[candidates])["ap"]
            trials.append(dict(stage=stage, k=k, power=power, alpha=alpha, validation_ap=ap))
    try_weights("baseline", "", 1., drug_only(train, data.drug_similarity),
                protein_only(train, data.protein_similarity))
    drug, ds = neighbour_scores(train, data.drug_similarity, data.drugs)
    protein, ps = neighbour_scores(train.T, data.protein_similarity, data.proteins)
    try_weights("active", "", 1., drug, protein.T)
    # Keep power=1 in this grid as the no-power control.
    for k in ks:
        for power in dict.fromkeys([1., *powers]):
            drug, ds = neighbour_scores(train, data.drug_similarity, data.drugs, k, power)
            protein, ps = neighbour_scores(train.T, data.protein_similarity, data.proteins, k, power)
            if power == 1:
                try_weights("topk", k, power, drug, protein.T)
            try_weights("power", k, power, drug, protein.T)
            try_weights("availability", k, power, drug, protein.T, ds, ps)
    selected = {stage: max((r for r in trials if r["stage"] == stage), key=lambda r: r["validation_ap"])
                for stage in STAGES}
    return selected, trials


def paired_deltas(rows):
    """Descriptive changes on the same splits; overlapping splits are not independent studies."""
    result = []
    for dataset in sorted({r["dataset"] for r in rows}):
        for previous, stage in zip(STAGES, STAGES[1:]):
            before = {r["seed"]: r for r in rows if r["dataset"] == dataset and r["stage"] == previous}
            after = {r["seed"]: r for r in rows if r["dataset"] == dataset and r["stage"] == stage}
            for metric in METRICS[:4]:
                changes = [after[s][metric] - before[s][metric] for s in before
                           if before[s][metric] is not None and after[s][metric] is not None]
                result.append(dict(dataset=dataset, metric=metric, previous=previous, stage=stage,
                    runs=len(changes), mean_delta=float(np.mean(changes)) if changes else None,
                    sd_delta=float(np.std(changes, ddof=1)) if len(changes) > 1 else None,
                    min_delta=min(changes) if changes else None, max_delta=max(changes) if changes else None,
                    wins=sum(d > 1e-12 for d in changes), ties=sum(abs(d) <= 1e-12 for d in changes),
                    losses=sum(d < -1e-12 for d in changes)))
    return result


def run_study(args):
    if args.output.exists() and any(p.name not in (".gitkeep", ".DS_Store") for p in args.output.iterdir()):
        raise ValueError(f"Use a new or empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    common = ("random", "target-degree", "drug-only", "protein-only", "two-sided")
    core = ("drug-only", "protein-only", "two-sided", "two-sided-topk")
    alignment, stability, ablation, parameters, records = [], [], [], [], []
    for dataset in args.dataset:
        data = load_dataset(args.input, dataset)
        seeds = set(args.seed) | {42}
        if dataset == "nr":
            seeds.update(range(args.stability_seeds))
        for seed in sorted(seeds):
            train, validation, test = split_edges(data.interactions, seed, args.validation_fraction, args.test_fraction)
            vc, tc = candidate_masks(data.interactions, validation, test)
            visible = (train | validation).astype(np.uint8)
            context = dict(dataset=dataset, seed=seed)
            record = dict(**context, train=int(train.sum()), validation=int(validation.sum()),
                          test=int(test.sum()), candidates=int(tc.sum()), background=int((data.interactions == 0).sum()),
                          data=data.audit)
            if seed == 42 or (dataset == "nr" and seed < args.stability_seeds):
                selected, _ = tune(train, data, validation, vc, args.alphas, args.ks, args.powers)
                scores, seconds = score_methods(visible, data, selected, seed, args.timing_repeats)
                record["selected"] = selected
                for method in set(common if seed == 42 else ()) | set(core if dataset == "nr" and seed < args.stability_seeds else ()):
                    values, _ = evaluate(scores[method], test, tc, data.proteins, args.top_n)
                    setting = selected.get(method, {})
                    row = dict(**context, method=method, alpha=setting.get("alpha", ""),
                               k=setting.get("k", ""), power=setting.get("power", ""), **values, score_seconds=seconds[method])
                    if seed == 42 and method in common:
                        alignment.append(row)
                    if dataset == "nr" and seed < args.stability_seeds and method in core:
                        stability.append(row)
            if seed in args.seed:
                chosen, trials = tune_ablation(train, data, validation, vc, args.alphas, args.ks, args.powers)
                record["ablation_selected"] = chosen
                parameters.extend(dict(**context, **r, selected=r is chosen[r["stage"]]) for r in trials)
                unsupported_drugs = int((~(data.drug_similarity[:, visible.sum(axis=1) > 0] > 0).any(axis=1)).sum())
                unsupported_proteins = int((~(data.protein_similarity[:, visible.sum(axis=0) > 0] > 0).any(axis=1)).sum())
                for stage in STAGES:
                    setting = chosen[stage]
                    scores, seconds = timed(lambda: ablation_scores(visible, data, setting), args.timing_repeats)
                    values, _ = evaluate(scores, test, tc, data.proteins, args.top_n)
                    ablation.append(dict(**context, **setting, **values, score_seconds=seconds,
                                         unsupported_drugs=unsupported_drugs, unsupported_proteins=unsupported_proteins))
            records.append(record)
            print(f"Study {dataset} seed {seed} complete", flush=True)
    fields = ["dataset", "seed", "method", "alpha", "k", "power", "ap", "auroc", "positives", "background",
              "macro_top_precision", "macro_top_recall", "eligible_drugs", "drugs_without_hidden", "top_n", "score_seconds"]
    for name, rows in (("seed42", alignment), ("stability", stability)):
        rows.sort(key=lambda r: (r["dataset"], r["seed"], r["method"]))
        save_csv(args.output / f"{name}.csv", rows, fields)
    save_csv(args.output / "ablation.csv", ablation)
    save_csv(args.output / "ablation_parameters.csv", parameters)
    save_csv(args.output / "paired.csv", paired_deltas(ablation))
    if stability:
        save_csv(args.output / "stability_summary.csv", aggregate(stability))
        frequencies = {}
        for r in stability:
            if r["method"] in ("two-sided", "two-sided-topk"):
                key = (r["method"], r["alpha"], r["k"], r["power"])
                frequencies[key] = frequencies.get(key, 0) + 1
        save_csv(args.output / "parameter_frequency.csv", [dict(method=m, alpha=a, k=k, power=p, count=n)
                 for (m, a, k, p), n in frequencies.items()])
    digest = hashlib.sha256()
    for path in sorted((ROOT / "code").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    config = {k: recorded_path(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    receipt = dict(config=config, path_base="project_root", runs=records, git=git_state(), source_sha256=digest.hexdigest(),
        python=sys.version, numpy=np.__version__, matplotlib=version("matplotlib") if not args.no_plots else None,
        thread_environment={n: os.environ.get(n) for n in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        finished_utc=datetime.now(timezone.utc).isoformat(), effective_ablation_powers=list(dict.fromkeys([1., *args.powers])),
        note="Seed42 is alignment only. NR splits overlap; summaries are descriptive. Every ablation stage retunes validation AP; differences include parameter selection, not pure fixed-parameter causal effects.")
    (args.output / "study.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    if not args.no_plots:
        from plot import plot_study
        plot_study(args.output)
    print(f"Saved alignment, stability and ablation checks to {args.output}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", default=["e", "ic", "gpcr", "nr"])
    parser.add_argument("--seed", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--input", type=Path, default=Path("input"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--study", action="store_true", help="Run additional checks without changing the main benchmark")
    parser.add_argument("--stability-seeds", type=int, default=20, help="Number of NR repeat splits in --study mode")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.5, 0.25, 0.75])
    parser.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20, 50])
    parser.add_argument("--powers", nargs="+", type=float, default=[1., 2.])
    parser.add_argument("--validation-fraction", type=float, default=.1)
    parser.add_argument("--test-fraction", type=float, default=.1)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--compare-backends", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-predictions", action="store_true")
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = Path("output/validation") if args.study else Path("output")
    args.input, args.output = project_path(args.input), project_path(args.output)
    if args.study and args.stability_seeds < 2:
        parser.error("stability-seeds must be at least two")
    if any(not np.isfinite(a) or not 0 <= a <= 1 for a in args.alphas):
        parser.error("alphas must be finite and between 0 and 1")
    if any(k < 1 for k in args.ks) or any(not np.isfinite(p) or p <= 0 for p in args.powers):
        parser.error("ks and powers must be positive")
    if args.top_n < 1 or args.timing_repeats < 1 or any(s < 0 for s in args.seed):
        parser.error("top-n/repeats must be positive; seeds must be nonnegative")
    if (not 0 < args.validation_fraction < 1 or not 0 < args.test_fraction < 1
            or args.validation_fraction + args.test_fraction >= 1):
        parser.error("held-out fractions must be positive and sum to less than one")
    if len(set(args.dataset)) != len(args.dataset) or len(set(args.seed)) != len(args.seed):
        parser.error("dataset and seed lists must not contain duplicates")
    return args


# 4. Run in order, then write shared tables instead of one directory per seed.
def main(argv=None):
    args = parse_args(argv)
    if args.study:
        return run_study(args)
    if args.output.exists() and any(p.name not in (".gitkeep", ".DS_Store") for p in args.output.iterdir()):
        raise ValueError(f"Use a new or empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    metrics, parameters, splits, per_drug, timing, predictions, records = [], [], [], [], [], [], []
    for dataset in args.dataset:
        data = load_dataset(args.input, dataset)
        for seed in args.seed:
            train, validation, test = split_edges(data.interactions, seed, args.validation_fraction, args.test_fraction)
            vc, tc = candidate_masks(data.interactions, validation, test)
            context = dict(dataset=dataset, seed=seed)
            for name, mask in (("train", train), ("validation", validation), ("test", test)):
                for i, j in np.argwhere(mask):
                    splits.append(dict(**context, split=name, drug=data.drugs[i], protein=data.proteins[j]))
            start = perf_counter()
            selected, trials = tune(train, data, validation, vc, args.alphas, args.ks, args.powers)
            tuning_seconds = perf_counter() - start
            parameters.extend(dict(**context, **r, selected=r is selected[r["method"]]) for r in trials)
            visible = (train | validation).astype(np.uint8)
            scores, durations = score_methods(visible, data, selected, seed, args.timing_repeats)
            cold_drugs, cold_proteins = visible.sum(axis=1) == 0, visible.sum(axis=0) == 0
            for method in METHODS:
                start = perf_counter()
                values, details = evaluate(scores[method], test, tc, data.proteins, args.top_n)
                evaluation_seconds = perf_counter() - start
                setting = selected.get(method, {})
                metrics.append(dict(**context, method=method, alpha=setting.get("alpha", ""),
                    k=setting.get("k", ""), power=setting.get("power", ""), **values,
                    score_seconds=durations[method], evaluation_seconds=evaluation_seconds))
                per_drug.extend(dict(**context, method=method, drug=data.drugs[r["drug_index"]],
                                     cold_drug=bool(cold_drugs[r["drug_index"]]), **r) for r in details)
            if args.compare_backends:
                timing.extend(dict(**context, k=selected["two-sided-topk"]["k"],
                    power=selected["two-sided-topk"]["power"], alpha=selected["two-sided-topk"]["alpha"], **r)
                    for r in compare_backends(visible, data, selected["two-sided-topk"], args.timing_repeats))
            records.append(dict(**context, selected=selected, tuning_seconds=tuning_seconds,
                train=int(train.sum()), validation=int(validation.sum()), test=int(test.sum()),
                background=int((data.interactions == 0).sum()), test_candidates=int(tc.sum()),
                cold_drugs=int(cold_drugs.sum()), cold_proteins=int(cold_proteins.sum()),
                test_pairs_with_cold_drug=int(test[cold_drugs].sum()),
                test_pairs_with_cold_protein=int(test[:, cold_proteins].sum()), data=data.audit))
            print(f"{dataset} seed {seed}: train {train.sum()}, validation {validation.sum()}, test {test.sum()}", flush=True)
            # Full-graph predictions use the FIRST requested seed's validation settings.
            # This choice is fixed in advance, never based on the final test results.
            if seed == args.seed[0] and not args.no_predictions:
                full_scores = topk_scores(data.interactions, data, selected["two-sided-topk"])
                for i, drug in enumerate(data.drugs):
                    unknown = np.flatnonzero(data.interactions[i] == 0)
                    top = sorted(unknown, key=lambda j: (-float(full_scores[i, j]), data.proteins[j]))[:args.top_n]
                    for rank, j in enumerate(top, 1):
                        predictions.append(dict(**context, drug=drug, protein=data.proteins[j],
                                                rank=rank, score=float(full_scores[i, j]), status="unlabeled"))
    for name, rows in (("metrics", metrics), ("summary", aggregate(metrics)),
                       ("parameters", parameters), ("splits", splits), ("per_drug", per_drug)):
        save_csv(args.output / f"{name}.csv", rows)
    if timing:
        save_csv(args.output / "timing.csv", timing)
    if not args.no_predictions:
        save_csv(args.output / "predictions.csv", predictions,
                 ["dataset", "seed", "drug", "protein", "rank", "score", "status"])
    config = {key: recorded_path(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    digest = hashlib.sha256()
    for path in sorted((ROOT / "code").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    receipt = dict(config=config, path_base="project_root", runs=records, git=git_state(), source_sha256=digest.hexdigest(),
        python=sys.version, numpy=np.__version__,
        matplotlib=version("matplotlib") if not args.no_plots else None, platform=platform.platform(),
        thread_environment={name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        finished_utc=datetime.now(timezone.utc).isoformat(), prediction_settings_seed=args.seed[0],
        note="Zeros are unlabeled; test measures random hidden-edge recovery. Predictions use all recorded positives.")
    (args.output / "experiment.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    if not args.no_plots:
        from plot import read_results, plot_results
        plot_results(read_results(args.output), args.output)
    print(f"Saved {len(metrics)} evaluations to {args.output}")


if __name__ == "__main__":
    main()

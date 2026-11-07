"""CLI for data checks and evaluator infrastructure, with saved experiment evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter
import tracemalloc
import numpy as np
from .data import load_dataset
from .benchmark import make_split
from .scoring import METHODS, score
from .evaluation import evaluate


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_rows(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_hashes(root, code):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(root).glob(f"{code}_*.txt"))}


def run(args):
    config = json.loads(args.config.read_text())
    methods = config["methods"]
    if not methods or any(m not in METHODS for m in methods):
        raise ValueError("Choose registered methods")
    if any(m in ("drug-only", "protein-only", "two-sided") for m in methods):
        raise NotImplementedError("Implement and verify student scoring before enabling these methods")
    # Existing experiment directories must never be overwritten silently.
    args.output.mkdir(parents=True, exist_ok=False)
    dump(args.output / "config.json", config)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    dump(args.output / "environment.json", dict(python=platform.python_version(),
         numpy=np.__version__, platform=platform.platform(),
         git_commit=commit.stdout.strip() or None, git_dirty=bool(status.stdout)))
    all_rows = []
    for code in config["datasets"]:
        start = perf_counter()
        data = load_dataset(args.data, code)
        load_seconds = perf_counter() - start
        dump(args.output / f"{code}_audit.json", data.audit)
        dump(args.output / f"{code}_hashes.json", source_hashes(args.data, code))
        dump(args.output / f"{code}_identifiers.json", dict(drugs=data.drugs, proteins=data.proteins))
        for mode in config["modes"]:
            for fraction in config["fractions"]:
                for seed in config["seeds"]:
                    directory = args.output / code / f"{mode}_f{fraction:g}_s{seed}"
                    directory.mkdir(parents=True)
                    start = perf_counter()
                    split = make_split(data.interactions, fraction, seed, mode)
                    split_seconds = perf_counter() - start
                    np.savez_compressed(directory / "split.npz", visible=split.visible,
                         hidden=split.hidden, candidates=split.candidates,
                         selected_entities=split.selected_entities)
                    for method in methods:
                        tracemalloc.start()
                        start = perf_counter()
                        scores = score(method, split.visible.copy(), data.drug_similarity.copy(),
                                       data.protein_similarity.copy(), seed)
                        score_seconds = perf_counter() - start
                        _, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        start = perf_counter()
                        metrics, per_drug, curves = evaluate(scores, split.hidden,
                            split.candidates, data.proteins, config["top_n"])
                        evaluation_seconds = perf_counter() - start
                        for row in per_drug:
                            row["drug_id"] = data.drugs[row["drug_index"]]
                        write_rows(directory / f"{method}_per_drug.csv", per_drug)
                        np.savez_compressed(directory / f"{method}_predictions.npz", scores=scores)
                        np.savez_compressed(directory / f"{method}_curves.npz", **curves)
                        row = dict(dataset=code, mode=mode, fraction=fraction, seed=seed,
                            method=method, visible_edges=int(split.visible.sum()),
                            hidden_edges=int(split.hidden.sum()), candidates=int(split.candidates.sum()),
                            actual_hidden_fraction=float(split.hidden.sum()/data.interactions.sum()),
                            zero_degree_drugs=int((split.visible.sum(axis=1)==0).sum()),
                            zero_degree_proteins=int((split.visible.sum(axis=0)==0).sum()),
                            load_seconds=load_seconds, split_seconds=split_seconds,
                            score_seconds=score_seconds, evaluation_seconds=evaluation_seconds,
                            score_traced_peak_bytes=peak, **metrics)
                        all_rows.append(row)
                        dump(directory / f"{method}_metrics.json", row)
    write_rows(args.output / "metrics.csv", all_rows)
    print(f"Saved {len(all_rows)} evaluations to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Validate orientation, IDs, weights and edge-list agreement")
    inspect.add_argument("--data", type=Path, default=Path("data/raw"))
    inspect.add_argument("--dataset", required=True, help="File prefix, e.g. e or a compatible custom dataset")
    inspect.add_argument("--output", type=Path)
    bench = sub.add_parser("benchmark", help="Run the registered methods on shared hidden-edge splits")
    bench.add_argument("--data", type=Path, default=Path("data/raw"))
    bench.add_argument("--config", type=Path, default=Path("configs/benchmark.json"))
    bench.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inspect":
        result = load_dataset(args.data, args.dataset).audit
        if args.output:
            dump(args.output, result)
        print(json.dumps(result, indent=2))
    else:
        run(args)

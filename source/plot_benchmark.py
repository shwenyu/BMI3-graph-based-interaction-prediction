"""Plot seed-averaged benchmark metrics and scoring time from summary.csv."""
import argparse
import csv
from pathlib import Path

import numpy as np


DATASETS = {
    "e": ("Enzyme", "#0072B2", "o", "-"),
    "ic": ("Ion channel", "#E69F00", "^", "--"),
    "gpcr": ("GPCR", "#CC79A7", "s", ":"),
    "nr": ("Nuclear receptor", "#606060", "D", "-."),
}
METHODS = {
    "random": "Random",
    "target-degree": "Target degree",
    "dti-only": "DTI paths",
    "drug-only": "Drug only",
    "protein-only": "Protein only",
    "two-sided": "Two-sided",
    "two-sided-topk": "Top-k fusion",
}
METRICS = ("ap", "auroc", "macro_top_precision", "macro_top_recall", "score_seconds")


def optional_number(value, label, bounded=False):
    """An empty cell is undefined; explicit NaN/Inf is malformed input."""
    if value is None or value.strip() in ("", "None", "null"):
        return None
    number = float(value)
    if not np.isfinite(number) or number < 0 or (bounded and number > 1):
        raise ValueError(f"Invalid {label}: {value}")
    return number


def read_results(results, seed=None):
    """Return dataset -> method -> numeric row. Optional seed reads old run CSVs."""
    results = Path(results)
    rows = []
    if seed is None:
        path = results / "summary.csv"
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"dataset", "method", "runs", "top_n"}
            required.update(f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "sd"))
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing summary columns in {path}")
            rows = list(reader)
    else:
        # Compatibility with the prototype's dataset_seedN/test.csv layout.
        paths = sorted(results.glob(f"*_seed{seed}/test.csv"))
        if not paths:
            raise ValueError(f"No results found for seed {seed}")
        for path in paths:
            code = path.parent.name.rsplit("_seed", 1)[0]
            with path.open(newline="") as handle:
                for original in csv.DictReader(handle):
                    row = dict(dataset=code, method=original["method"], runs="1",
                               top_n=original["top_n"])
                    for metric in METRICS:
                        row[f"{metric}_mean"] = original.get(metric, "")
                        row[f"{metric}_sd"] = "0" if original.get(metric, "") else ""
                    rows.append(row)
    if not rows:
        raise ValueError("Result table is empty")
    tables = {}
    top_counts = set()
    for original in rows:
        code, method = original["dataset"].strip(), original["method"].strip()
        if not code or method not in METHODS:
            raise ValueError(f"Unknown dataset/method: {code}/{method}")
        table = tables.setdefault(code, {})
        if method in table:
            raise ValueError(f"Duplicate result for {code}/{method}")
        row = {"runs": int(original["runs"]), "top_n": int(original["top_n"])}
        if row["runs"] < 1 or row["top_n"] < 1:
            raise ValueError("runs and top_n must be positive integers")
        top_counts.add(row["top_n"])
        for metric in METRICS:
            mean = optional_number(original[f"{metric}_mean"], metric, metric != "score_seconds")
            sd = optional_number(original[f"{metric}_sd"], metric + " standard deviation")
            if mean is None and sd is not None:
                raise ValueError(f"Standard deviation without a mean: {code}/{method}/{metric}")
            row[f"{metric}_mean"], row[f"{metric}_sd"] = mean, sd
        table[method] = row
    if len(top_counts) != 1:
        raise ValueError("All rows must use the same top-N cutoff")
    method_sets = {tuple(sorted(table)) for table in tables.values()}
    if len(method_sets) != 1:
        raise ValueError("All datasets must contain the same methods")
    return tables


def draw_metric(ax, tables, metric, label, methods):
    positions = np.arange(len(methods))
    for index, (code, table) in enumerate(tables.items()):
        name, color, marker, style = DATASETS.get(
            code, (code, f"C{index % 10}", "o", "-"))
        values = [table[method][f"{metric}_mean"] for method in methods]
        errors = [table[method][f"{metric}_sd"] for method in methods]
        values = np.array([np.nan if v is None else v for v in values])
        errors = np.array([0 if e is None else e for e in errors])
        ax.errorbar(positions, values, yerr=errors, label=name, color=color,
                    marker=marker, linestyle=style, linewidth=1.6, markersize=5,
                    capsize=3, elinewidth=1)
    ax.set_xticks(positions, [METHODS[m] for m in methods], rotation=25, ha="right")
    ax.set_ylabel(label)
    ax.set_xlim(-0.3, len(methods) - 0.7)
    ax.set_ylim(bottom=0)
    if metric != "score_seconds":
        # Seed SD bars can extend past the valid metric range; show their full extent.
        top = ax.get_ylim()[1]
        ax.set_ylim(top=min(1.1, max(0.05, top)))
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def plot_results(tables, output):
    # Matplotlib is optional for running the prediction benchmark itself.
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    methods = [m for m in METHODS if m in next(iter(tables.values()))]
    top_n = next(iter(next(iter(tables.values())).values()))["top_n"]
    labels = {
        "ap": "Average precision",
        "auroc": "AUROC",
        "macro_top_precision": f"Macro precision@min({top_n}, candidates)",
        "macro_top_recall": f"Macro recall@{top_n}",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (metric, label) in zip(axes.flat, labels.items()):
        draw_metric(ax, tables, metric, label, methods)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=min(4, len(tables)),
               frameon=False, bbox_to_anchor=(0.5, 0.03))
    fig.suptitle("Held-out interaction recovery: mean and seed standard deviation", fontsize=13)
    fig.text(0.5, 0.005, "Top-N macro metrics include only drugs with held-out positives.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    for extension in ("png", "pdf"):
        fig.savefig(output / f"benchmark.{extension}", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    draw_metric(ax, tables, "score_seconds", "Scoring time (seconds)", methods)
    ax.set_title("Scoring time: mean and seed standard deviation")
    ax.legend(frameon=False, ncol=min(4, len(tables)))
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"runtime.{extension}", dpi=220)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=root / "output" / "benchmark")
    parser.add_argument("--output", type=Path, help="Directory for PNG and PDF figures")
    parser.add_argument("--seed", type=int, help="Read legacy dataset_seedN/test.csv files")
    args = parser.parse_args()
    tables = read_results(args.results, args.seed)
    output = args.output or args.results / "figures"
    plot_results(tables, output)
    print(f"Saved benchmark and runtime figures (PNG/PDF) in {output}")


if __name__ == "__main__":
    main()

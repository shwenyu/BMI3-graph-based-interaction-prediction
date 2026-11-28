"""Draw the benchmark summary. AI-assisted; sources are listed in README.md."""
import argparse
import csv
from pathlib import Path

import numpy as np

METHODS = {
    "random": "Random", "target-degree": "Target degree", "dti-only": "DTI paths",
    "drug-only": "Drug only", "protein-only": "Protein only",
    "two-sided": "Two-sided", "two-sided-topk": "Top-k fusion",
}
DATASETS = {
    "e": ("Enzyme", "#0072B2", "o", "-"),
    "ic": ("Ion channel", "#E69F00", "^", "--"),
    "gpcr": ("GPCR", "#CC79A7", "s", ":"),
    "nr": ("Nuclear receptor", "#606060", "D", "-."),
}
METRICS = ("ap", "auroc", "macro_top_precision", "macro_top_recall", "score_seconds")


def optional_number(value, label, bounded=False):
    if value is None or value.strip() in ("", "None", "null"):
        return None
    number = float(value)
    if not np.isfinite(number) or number < 0 or (bounded and number > 1):
        raise ValueError(f"Invalid {label}: {value}")
    return number


def read_results(results):
    """Read one summary row per dataset and method; blanks mean undefined."""
    required = {"dataset", "method", "runs", "top_n"}
    required.update(f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "sd"))
    tables, cutoffs = {}, set()
    with (Path(results) / "summary.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("summary.csv is missing required columns")
        for original in reader:
            code, method = original["dataset"].strip(), original["method"].strip()
            if not code or method not in METHODS or method in tables.get(code, {}):
                raise ValueError(f"Unknown or duplicate dataset/method: {code}/{method}")
            row = dict(runs=int(original["runs"]), top_n=int(original["top_n"]))
            if row["runs"] < 1 or row["top_n"] < 1:
                raise ValueError("runs and top_n must be positive")
            cutoffs.add(row["top_n"])
            for metric in METRICS:
                mean = optional_number(original[f"{metric}_mean"], metric, metric != "score_seconds")
                sd = optional_number(original[f"{metric}_sd"], metric + " SD")
                if mean is None and sd is not None:
                    raise ValueError("A standard deviation needs a defined mean")
                row[f"{metric}_mean"], row[f"{metric}_sd"] = mean, sd
            tables.setdefault(code, {})[method] = row
    if not tables or len(cutoffs) != 1 or len({tuple(sorted(t)) for t in tables.values()}) != 1:
        raise ValueError("Results need matching methods and top_n across datasets")
    return tables


def draw_panel(ax, tables, metric, label, methods):
    positions = np.arange(len(methods))
    for index, (code, table) in enumerate(tables.items()):
        name, color, marker, style = DATASETS.get(code, (code, f"C{index % 10}", "o", "-"))
        means = [table[m][f"{metric}_mean"] for m in methods]
        errors = [table[m][f"{metric}_sd"] for m in methods]
        means = [np.nan if value is None else value for value in means]
        errors = [0 if value is None else value for value in errors]
        ax.errorbar(positions, means, yerr=errors, label=name, color=color,
                    marker=marker, linestyle=style, capsize=3, linewidth=1.5)
    ax.set_xticks(positions, [METHODS[m] for m in methods], rotation=25, ha="right")
    ax.set_ylabel(label)
    ax.grid(axis="y", color="#E5E5E5")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def plot_results(tables, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    first = next(iter(tables.values()))
    methods = [method for method in METHODS if method in first]
    top_n = next(iter(first.values()))["top_n"]
    labels = ("Average precision", "AUROC", f"Macro precision@min({top_n}, candidates)",
              f"Macro recall@{top_n}")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric, label in zip(axes.flat, METRICS[:4], labels):
        draw_panel(ax, tables, metric, label, methods)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=min(4, len(tables)), frameon=False,
               bbox_to_anchor=(0.5, 0.025))
    fig.suptitle("Held-out interaction recovery: mean and seed standard deviation")
    fig.text(0.5, 0.005, "Macro metrics include drugs with held-out positives only.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    for extension in ("png", "pdf"):
        fig.savefig(output / f"benchmark.{extension}", dpi=220)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    draw_panel(ax, tables, "score_seconds", "Scoring time (seconds)", methods)
    ax.set_title("Scoring time: mean and seed standard deviation")
    ax.legend(frameon=False, ncol=min(4, len(tables)))
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"runtime.{extension}", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "output")
    args = parser.parse_args()
    plot_results(read_results(args.output), args.output)
    print(f"Saved benchmark and runtime figures to {args.output}")

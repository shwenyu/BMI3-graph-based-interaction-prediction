"""Plot one test metric per figure for the four Yamanishi datasets."""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter


DATASETS = {
    "e": "Enzyme",
    "ic": "Ion channel",
    "gpcr": "GPCR",
    "nr": "Nuclear receptor",
}
METHODS = {
    "random": ("Random", "#777777"),
    "target-degree": ("Target degree", "#E69F00"),
    "drug-only": ("Drug only", "#0072B2"),
    "protein-only": ("Protein only", "#009E73"),
    "two-sided": ("Two-sided", "#CC79A7"),
}


def read_results(results, seed):
    tables = {}
    for code in DATASETS:
        path = results / f"{code}_seed{seed}" / "test.csv"
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        table = {row["method"]: row for row in rows}
        if len(table) != len(rows) or not set(METHODS).issubset(table):
            raise ValueError(f"Expected one row per benchmark method in {path}")
        tables[code] = table
    return tables


def plot_metric(tables, metric, label, output, top_n):
    fig, ax = plt.subplots(figsize=(8.3, 4.4), layout="constrained")
    positions = range(len(DATASETS))
    width = 0.15
    largest = 0

    for index, (method, (name, color)) in enumerate(METHODS.items()):
        offset = width * (index - (len(METHODS) - 1) / 2)
        x = [position + offset for position in positions]
        values = [float(tables[code][method][metric]) for code in DATASETS]
        if any(not 0 <= value <= 1 for value in values):
            raise ValueError(f"{method}: {metric} must contain scores from 0 to 1")
        ax.bar(x, values, width=width, label=name, color=color)
        largest = max(largest, max(values))

    ax.set_xticks(list(positions), list(DATASETS.values()))
    ax.set_ylabel(label)
    ax.set_ylim(0, min(1, max(0.05, largest * 1.12)))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E4E4E4")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=5, frameon=False)

    stem = metric
    if metric.startswith("macro_top_"):
        stem = f"{metric}_{top_n}"
    fig.savefig(output / f"{stem}.png", dpi=300)
    fig.savefig(output / f"{stem}.pdf")
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=root / "output")
    parser.add_argument("--output", type=Path, help="Directory for PNG and PDF figures")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tables = read_results(args.results, args.seed)
    top_counts = {int(row["top_n"]) for table in tables.values() for row in table.values()}
    if len(top_counts) != 1:
        raise ValueError("All methods and datasets must use the same top-N cutoff")
    top_n = top_counts.pop()
    metrics = {
        "ap": "Average precision (AP)",
        "auroc": "AUROC",
        "macro_top_precision": f"Mean precision in top {top_n} targets",
        "macro_top_recall": f"Mean recall in top {top_n} targets",
    }
    output = args.output or args.results / f"figures_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    for metric, label in metrics.items():
        plot_metric(tables, metric, label, output, top_n)
    print(f"Saved {len(metrics)} figures as PNG and PDF in {output}")


if __name__ == "__main__":
    main()

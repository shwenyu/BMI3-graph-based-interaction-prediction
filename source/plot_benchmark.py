"""Plot four test metrics in a single figure for the Yamanishi datasets."""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter


DATASETS = {
    "e": ("Enzyme", "#0072B2", "o", "-"),
    "ic": ("Ion channel", "#E69F00", "^", "--"),
    "gpcr": ("GPCR", "#CC79A7", "s", ":"),
    "nr": ("Nuclear receptor", "#606060", "D", "-."),
}
METHODS = {
    "random": "Random",
    "target-degree": "Target degree",
    "drug-only": "Drug only",
    "protein-only": "Protein only",
    "two-sided": "Two-sided",
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


def plot_metric(ax, tables, metric, label):
    positions = list(range(len(METHODS)))
    largest = 0

    for code, (name, color, marker, line_style) in DATASETS.items():
        values = [float(tables[code][method][metric]) for method in METHODS]
        if any(not 0 <= value <= 1 for value in values):
            raise ValueError(f"{code}: {metric} must contain scores from 0 to 1")
        ax.plot(positions, values, label=name, color=color, marker=marker,
                linestyle=line_style, linewidth=2, markersize=5)
        largest = max(largest, max(values))

    ax.set_xticks(positions, list(METHODS.values()))
    ax.set_xlim(-0.25, len(METHODS) - 0.75)
    ax.set_ylim(0, min(1, max(0.05, largest * 1.12)))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_axisbelow(True)
    ax.grid(color="#EAEAEA", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("#454545")
        spine.set_linewidth(0.8)

    # Draw a grey metric strip above each panel.
    strip_height = 0.11
    ax.add_patch(Rectangle((0, 1), 1, strip_height, transform=ax.transAxes,
                           facecolor="#E3E3E3", edgecolor="#454545",
                           linewidth=0.8, clip_on=False))
    ax.text(0.5, 1 + strip_height / 2, label, transform=ax.transAxes,
            ha="center", va="center", fontsize=11, fontweight="bold")


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
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.14, top=0.95,
                        wspace=0.18, hspace=0.43)
    for ax, (metric, label) in zip(axes.flat, metrics.items()):
        plot_metric(ax, tables, metric, label)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.03),
               ncol=4, frameon=False, handlelength=2.8)
    fig.savefig(output / "benchmark.png", dpi=300)
    fig.savefig(output / "benchmark.pdf")
    plt.close(fig)
    print(f"Saved the 2x2 benchmark figure as PNG and PDF in {output}")


if __name__ == "__main__":
    main()

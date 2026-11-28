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


def plot_study(output):
    """Keep single-seed, repeated-split and component comparisons separate."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    output = Path(output)
    def read(name):
        with (output / f"{name}.csv").open(newline="") as handle:
            return list(csv.DictReader(handle))
    def number(row, metric):
        value = optional_number(row[metric], metric, metric != "score_seconds")
        return np.nan if value is None else value
    def finish(fig, name):
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{name}.{extension}", dpi=220)
        plt.close(fig)
    def style(ax, metric):
        ax.grid(axis="y", color="#E5E5E5")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        if metric != "score_seconds":
            ax.yaxis.set_major_formatter(PercentFormatter(1))

    # The colleague's view: five shared methods and one fixed split.
    rows = read("seed42")
    top_n = int(rows[0].get("top_n", 10))
    methods = ("random", "target-degree", "drug-only", "protein-only", "two-sided")
    labels = ("Average precision", "AUROC", f"Top-{top_n} macro precision", f"Top-{top_n} macro recall", "Scoring time (ms, log scale)")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[0, 2])
    for ax, metric, label in zip(panels, METRICS, labels):
        for code in DATASETS:
            table = {r["method"]: r for r in rows if r["dataset"] == code}
            if not table:
                continue
            name, color, marker, line = DATASETS[code]
            values = [number(table[m], metric) for m in methods]
            if metric == "score_seconds":
                values = [v * 1000 if v > 0 else np.nan for v in values]
            ax.plot(range(5), values, label=name, color=color, marker=marker, linestyle=line)
        ax.set_xticks(range(5), [METHODS[m] for m in methods], rotation=20, ha="right")
        ax.set_title(label)
        style(ax, metric)
        if metric == "score_seconds":
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0, top=1 if metric == "auroc" else None)
    axes[1, 2].axis("off")
    axes[1, 2].legend(*panels[0].get_legend_handles_labels(), loc="center", frameon=False)
    fig.suptitle("DTI benchmark: seed 42, five shared methods")
    fig.tight_layout(rect=(0, 0, 1, .95))
    finish(fig, "seed42")

    # Repeated NR splits: the same seed pairs the fixed and enhanced methods.
    rows = read("stability")
    if rows:
        values = {(int(r["seed"]), r["method"]): number(r, "ap") for r in rows}
        seeds = sorted({seed for seed, _ in values})
        methods = ("drug-only", "protein-only", "two-sided", "two-sided-topk")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for method, color in zip(methods, ("#0072B2", "#CC79A7", "#606060", "#E69F00")):
            axes[0].plot(seeds, [values[s, method] for s in seeds], marker="o", markersize=4,
                         label=METHODS[method], color=color)
        delta = [values[s, "two-sided-topk"] - values[s, "two-sided"] for s in seeds]
        axes[1].bar(seeds, delta, color=["#E69F00" if v >= 0 else "#606060" for v in delta])
        axes[1].axhline(0, color="black", linewidth=.8)
        axes[0].set_ylabel("AP")
        axes[0].set_ylim(bottom=0)
        axes[1].set_ylabel("AP difference: Top-k minus fixed fusion")
        for ax in axes:
            ax.set_xlabel("Split seed")
            ax.set_xticks(seeds[::max(1, len(seeds) // 10)])
            style(ax, "ap")
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle("Nuclear receptor: repeated edge splits, paired by seed")
        fig.tight_layout(rect=(0, .05, 1, .93))
        fig.text(.5, .01, "Repeated splits reuse the same dataset; they are not independent datasets.", ha="center", fontsize=9)
        finish(fig, "stability")

    rows = read("ablation")
    stages = ("baseline", "active", "topk", "power", "availability")
    labels = ("Fixed fusion", "Active neighbours", "+ Top-k", "+ Power", "+ Availability")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric, title in zip(axes.flat, METRICS[:4], ("Average precision", "AUROC", f"Top-{top_n} macro precision", f"Top-{top_n} macro recall")):
        for code in DATASETS:
            if not any(r["dataset"] == code for r in rows):
                continue
            groups = [[number(r, metric) for r in rows if r["dataset"] == code and r["stage"] == stage] for stage in stages]
            means = [np.mean(g) for g in groups]
            errors = [np.std(g, ddof=1) if len(g) > 1 else 0 for g in groups]
            name, color, marker, line = DATASETS[code]
            ax.errorbar(range(5), means, yerr=errors, label=name, color=color,
                        marker=marker, linestyle=line, capsize=3)
        ax.set_xticks(range(5), labels, rotation=20, ha="right")
        ax.set_title(title)
        style(ax, metric)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Component study: validation-selected settings, mean and seed SD")
    fig.tight_layout(rect=(0, .07, 1, .95))
    finish(fig, "ablation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "output")
    args = parser.parse_args()
    plot_results(read_results(args.output), args.output)
    print(f"Saved benchmark and runtime figures to {args.output}")

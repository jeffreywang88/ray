"""
Visualization for routing algorithm study results.

Generates:
- Latency CDFs comparing algorithms
- Fairness metrics bar charts
- Heatmaps across configuration space
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis.aggregate import (
    RESULTS_DIR,
    create_summary_dataframe,
    load_raw_results,
    load_manifest,
)


# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# Output directory
FIGURES_DIR = RESULTS_DIR.parent / "figures"


def plot_latency_cdf(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    title: str = "Latency CDF by Algorithm",
) -> plt.Figure:
    """
    Plot latency CDF comparing algorithms.

    Args:
        df: DataFrame with latency columns per algorithm.
        output_path: Path to save figure (optional).
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    algorithms = df["algorithm"].unique()
    colors = {"pow2": "#2ecc71", "random": "#e74c3c", "round_robin": "#3498db"}

    for alg in algorithms:
        alg_df = df[df["algorithm"] == alg]
        latencies = alg_df["latency_p99"].sort_values()

        # Compute CDF
        cdf = np.arange(1, len(latencies) + 1) / len(latencies)

        ax.plot(
            latencies,
            cdf,
            label=alg.replace("_", " ").title(),
            color=colors.get(alg, None),
            linewidth=2,
        )

    ax.set_xlabel("p99 Latency (ms)", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    return fig


def plot_latency_comparison(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    percentile: str = "p99",
) -> plt.Figure:
    """
    Plot latency comparison across algorithms as grouped bar chart.

    Args:
        df: Summary DataFrame.
        output_path: Path to save figure.
        percentile: Which percentile to plot (p50, p90, p95, p99).

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    latency_col = f"latency_{percentile}"

    # Group by algorithm and scale
    grouped = df.groupby(["scale", "algorithm"])[latency_col].mean().unstack()

    grouped.plot(kind="bar", ax=ax, width=0.8)

    ax.set_xlabel("Scale", fontsize=12)
    ax.set_ylabel(f"{percentile.upper()} Latency (ms)", fontsize=12)
    ax.set_title(f"{percentile.upper()} Latency by Scale and Algorithm", fontsize=14)
    ax.legend(title="Algorithm")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    return fig


def plot_fairness_comparison(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot Jain's Fairness Index comparison across algorithms.

    Args:
        df: Summary DataFrame.
        output_path: Path to save figure.

    Returns:
        matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Child replica fairness
    ax1 = axes[0]
    grouped_child = df.groupby("algorithm")["child_jains"].agg(["mean", "std"])
    x = range(len(grouped_child))
    ax1.bar(x, grouped_child["mean"], yerr=grouped_child["std"], capsize=5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([a.replace("_", " ").title() for a in grouped_child.index])
    ax1.set_ylabel("Jain's Fairness Index", fontsize=12)
    ax1.set_title("Child Replica Fairness", fontsize=14)
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="Perfect Fairness")
    ax1.legend()

    # Parent replica fairness
    ax2 = axes[1]
    grouped_parent = df.groupby("algorithm")["parent_jains"].agg(["mean", "std"])
    ax2.bar(x, grouped_parent["mean"], yerr=grouped_parent["std"], capsize=5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([a.replace("_", " ").title() for a in grouped_parent.index])
    ax2.set_ylabel("Jain's Fairness Index", fontsize=12)
    ax2.set_title("Parent Replica Fairness", fontsize=14)
    ax2.set_ylim(0, 1.05)
    ax2.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="Perfect Fairness")
    ax2.legend()

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    return fig


def plot_heatmap(
    df: pd.DataFrame,
    metric: str,
    row_var: str,
    col_var: str,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    cmap: str = "RdYlGn_r",
    fmt: str = ".2f",
) -> plt.Figure:
    """
    Plot heatmap of a metric across two configuration variables.

    Args:
        df: Summary DataFrame.
        metric: Column name for the metric to plot.
        row_var: Variable for rows.
        col_var: Variable for columns.
        output_path: Path to save figure.
        title: Plot title.
        cmap: Colormap name.
        fmt: Format string for annotations.

    Returns:
        matplotlib Figure.
    """
    # Pivot data
    pivot = df.pivot_table(values=metric, index=row_var, columns=col_var, aggfunc="mean")

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        ax=ax,
        cbar_kws={"label": metric},
    )

    ax.set_title(title or f"{metric} by {row_var} and {col_var}", fontsize=14)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    return fig


def plot_algorithm_heatmap(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot heatmap of p99 latency across topology and locality for each algorithm.

    Args:
        df: Summary DataFrame (should be filtered to Large scale).
        output_path: Path to save figure.

    Returns:
        matplotlib Figure.
    """
    algorithms = df["algorithm"].unique()
    n_algs = len(algorithms)

    fig, axes = plt.subplots(1, n_algs, figsize=(5 * n_algs, 4))
    if n_algs == 1:
        axes = [axes]

    for ax, alg in zip(axes, algorithms):
        alg_df = df[df["algorithm"] == alg]

        # Create combined index
        alg_df = alg_df.copy()
        alg_df["config"] = alg_df["topology"] + "\n" + alg_df["locality"].map(
            {True: "local", False: "no-local"}
        )

        pivot = alg_df.pivot_table(
            values="latency_p99",
            index="ratio",
            columns="config",
            aggfunc="mean",
        )

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn_r",
            ax=ax,
        )

        ax.set_title(alg.replace("_", " ").title(), fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("Ratio" if ax == axes[0] else "")

    fig.suptitle("p99 Latency (ms) by Configuration", fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    return fig


def plot_throughput_vs_latency(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot throughput vs p99 latency scatter plot.

    Args:
        df: Summary DataFrame.
        output_path: Path to save figure.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    algorithms = df["algorithm"].unique()
    colors = {"pow2": "#2ecc71", "random": "#e74c3c", "round_robin": "#3498db"}
    markers = {"pow2": "o", "random": "s", "round_robin": "^"}

    for alg in algorithms:
        alg_df = df[df["algorithm"] == alg]
        ax.scatter(
            alg_df["goodput"],
            alg_df["latency_p99"],
            label=alg.replace("_", " ").title(),
            color=colors.get(alg),
            marker=markers.get(alg, "o"),
            s=50,
            alpha=0.7,
        )

    ax.set_xlabel("Goodput (successful req/s)", fontsize=12)
    ax.set_ylabel("p99 Latency (ms)", fontsize=12)
    ax.set_title("Throughput vs Latency Trade-off", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    return fig


def plot_error_rate(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot error rate by load level and algorithm.

    Args:
        df: Summary DataFrame.
        output_path: Path to save figure.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by load level and algorithm
    grouped = df.groupby(["load_level", "algorithm"])["error_rate"].mean().unstack()

    grouped.plot(kind="bar", ax=ax, width=0.8)

    ax.set_xlabel("Load Level", fontsize=12)
    ax.set_ylabel("Error Rate", fontsize=12)
    ax.set_title("Error Rate by Load Level and Algorithm", fontsize=14)
    ax.legend(title="Algorithm")
    ax.set_xticklabels([f"{int(x * 100)}%" for x in grouped.index], rotation=0)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    return fig


def plot_routing_delay_comparison(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot routing delay comparison across algorithms.

    Routing delay measures time from Parent send to Child receive,
    capturing routing decision time + network RTT + queue wait.

    Args:
        df: Summary DataFrame.
        output_path: Path to save figure.

    Returns:
        matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"pow2": "#2ecc71", "random": "#e74c3c", "round_robin": "#3498db"}

    # Plot 1: Routing delay by algorithm (bar chart)
    ax1 = axes[0]
    grouped = df.groupby("algorithm")["routing_delay_mean"].agg(["mean", "std"])
    x = range(len(grouped))
    ax1.bar(x, grouped["mean"], yerr=grouped["std"], capsize=5,
            color=[colors.get(alg, "#888888") for alg in grouped.index])
    ax1.set_xticks(x)
    ax1.set_xticklabels([a.replace("_", " ").title() for a in grouped.index])
    ax1.set_ylabel("Routing Delay (ms)", fontsize=12)
    ax1.set_title("Mean Routing Delay by Algorithm", fontsize=14)

    # Plot 2: Routing delay percentiles by algorithm
    ax2 = axes[1]
    algorithms = df["algorithm"].unique()
    x = np.arange(len(algorithms))
    width = 0.35

    p50_vals = [df[df["algorithm"] == alg]["routing_delay_p50"].mean() for alg in algorithms]
    p99_vals = [df[df["algorithm"] == alg]["routing_delay_p99"].mean() for alg in algorithms]

    ax2.bar(x - width/2, p50_vals, width, label="p50", alpha=0.8)
    ax2.bar(x + width/2, p99_vals, width, label="p99", alpha=0.8)

    ax2.set_xticks(x)
    ax2.set_xticklabels([a.replace("_", " ").title() for a in algorithms])
    ax2.set_ylabel("Routing Delay (ms)", fontsize=12)
    ax2.set_title("Routing Delay Percentiles", fontsize=14)
    ax2.legend()

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    return fig


def plot_latency_breakdown(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot latency breakdown showing simulated work, routing delay, and overhead.

    This helps visualize where time is spent:
    - Simulated latency: actual work time in Child
    - Routing delay: Parent→Child transmission (routing + network + queue)
    - Other overhead: remaining time (HTTP, serialization, etc.)

    Args:
        df: Summary DataFrame.
        output_path: Path to save figure.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "simulated": "#2ecc71",  # Green - actual work
        "routing": "#f39c12",    # Orange - routing delay
        "overhead": "#e74c3c",   # Red - other overhead
    }

    algorithms = df["algorithm"].unique()
    x = range(len(algorithms))
    width = 0.6

    simulated = []
    routing = []
    overhead = []

    for alg in algorithms:
        alg_df = df[df["algorithm"] == alg]
        sim_mean = alg_df["simulated_latency_mean"].mean()
        route_mean = alg_df["routing_delay_mean"].mean()
        total_mean = alg_df["latency_mean"].mean()
        other = max(0, total_mean - sim_mean - route_mean)

        simulated.append(sim_mean)
        routing.append(route_mean)
        overhead.append(other)

    # Stacked bar chart
    ax.bar(x, simulated, width, label="Simulated Work", color=colors["simulated"])
    ax.bar(x, routing, width, bottom=simulated, label="Routing Delay", color=colors["routing"])
    ax.bar(x, overhead, width, bottom=[s + r for s, r in zip(simulated, routing)],
           label="Other Overhead", color=colors["overhead"])

    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("_", " ").title() for a in algorithms])
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Latency Breakdown by Algorithm", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    return fig


def generate_all_figures(
    df: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
) -> List[Path]:
    """
    Generate all standard figures from summary DataFrame.

    Args:
        df: Summary DataFrame.
        output_dir: Directory to save figures.

    Returns:
        List of paths to generated figures.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    # 1. Latency comparison by scale
    path = output_dir / "latency_by_scale.png"
    plot_latency_comparison(df, path)
    generated.append(path)

    # 2. Fairness comparison
    path = output_dir / "fairness_comparison.png"
    plot_fairness_comparison(df, path)
    generated.append(path)

    # 3. Throughput vs latency
    path = output_dir / "throughput_vs_latency.png"
    plot_throughput_vs_latency(df, path)
    generated.append(path)

    # 4. Error rate by load level
    path = output_dir / "error_rate_by_load.png"
    plot_error_rate(df, path)
    generated.append(path)

    # 5. Routing delay comparison
    path = output_dir / "routing_delay_comparison.png"
    plot_routing_delay_comparison(df, path)
    generated.append(path)

    # 6. Latency breakdown (simulated work vs routing vs overhead)
    path = output_dir / "latency_breakdown.png"
    plot_latency_breakdown(df, path)
    generated.append(path)

    # 8. Algorithm heatmap (Large scale only)
    large_df = df[df["scale"] == "large"]
    if len(large_df) > 0:
        path = output_dir / "algorithm_heatmap_large.png"
        plot_algorithm_heatmap(large_df, path)
        generated.append(path)

    # 9. Heatmaps for key metrics
    heatmap_metrics = [
        ("latency_p99", "RdYlGn_r"),
        ("child_jains", "RdYlGn"),
        ("error_rate", "RdYlGn_r"),
        ("routing_delay_mean", "RdYlGn_r"),
        ("simulated_latency_mean", "RdYlGn_r"),
    ]

    for metric, cmap in heatmap_metrics:
        path = output_dir / f"heatmap_{metric}_algo_scale.png"
        plot_heatmap(
            df,
            metric=metric,
            row_var="algorithm",
            col_var="scale",
            output_path=path,
            cmap=cmap,
        )
        generated.append(path)

    print(f"\nGenerated {len(generated)} figures in {output_dir}")
    return generated


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate visualizations")
    parser.add_argument(
        "--input",
        type=Path,
        default=RESULTS_DIR / "all_metrics.csv",
        help="Input CSV file with aggregated metrics",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURES_DIR,
        help=f"Output directory for figures. Default: {FIGURES_DIR}",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        print("Run analysis/aggregate.py first to generate metrics.")
        exit(1)

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    generate_all_figures(df, args.output_dir)


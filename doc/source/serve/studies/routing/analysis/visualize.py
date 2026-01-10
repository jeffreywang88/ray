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


# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# Default output directory
FIGURES_DIR = Path("/tmp/routing_results/figures")


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
    ax.tick_params(axis="x", rotation=45)

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
    # Format algorithm names for display
    plot_df = df.copy()
    plot_df["algorithm"] = plot_df["algorithm"].str.replace("_", " ").str.title()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Child replica fairness
    ax1 = axes[0]
    sns.barplot(data=plot_df, x="algorithm", y="child_jains", ax=ax1, errorbar="sd", 
                hue="algorithm", palette="husl", legend=False)
    ax1.set_xlabel("")
    ax1.set_ylabel("Jain's Fairness Index", fontsize=12)
    ax1.set_title("Child Replica Fairness", fontsize=14)
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=1.0, color="green", linestyle="--", alpha=0.5)

    # Parent replica fairness
    ax2 = axes[1]
    sns.barplot(data=plot_df, x="algorithm", y="parent_jains", ax=ax2, errorbar="sd",
                hue="algorithm", palette="husl", legend=False)
    ax2.set_xlabel("")
    ax2.set_ylabel("Jain's Fairness Index", fontsize=12)
    ax2.set_title("Parent Replica Fairness", fontsize=14)
    ax2.set_ylim(0, 1.05)
    ax2.axhline(y=1.0, color="green", linestyle="--", alpha=0.5)

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
    # Format load level ticks as percentages
    ax.set_xticks(range(len(grouped.index)))
    ax.set_xticklabels([f"{int(x * 100)}%" for x in grouped.index], rotation=0)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    return fig


def plot_parent_to_child_delay_comparison(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot parent-to-child delay comparison across algorithms.

    Parent→Child delay measures time from Parent send to Child receive,
    capturing routing decision time + network RTT + queue wait.

    Args:
        df: Summary DataFrame.
        output_path: Path to save figure.

    Returns:
        matplotlib Figure.
    """
    if "parent_child_delay_mean" not in df.columns:
        print("Skipping parent-to-child delay plot: no data")
        return None

    # Format algorithm names for display
    plot_df = df.copy()
    plot_df["algorithm"] = plot_df["algorithm"].str.replace("_", " ").str.title()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Mean delay by algorithm
    ax1 = axes[0]
    sns.barplot(data=plot_df, x="algorithm", y="parent_child_delay_mean", ax=ax1,
                errorbar="sd", hue="algorithm", palette="husl", legend=False)
    ax1.set_xlabel("")
    ax1.set_ylabel("Parent→Child Delay (ms)", fontsize=12)
    ax1.set_title("Mean Parent→Child Delay", fontsize=14)

    # Plot 2: Percentiles by algorithm
    ax2 = axes[1]
    pct_cols = ["parent_child_delay_p50", "parent_child_delay_p99"]
    pct_cols = [c for c in pct_cols if c in df.columns]
    if pct_cols:
        pct_df = plot_df.melt(id_vars=["algorithm"], value_vars=pct_cols,
                              var_name="percentile", value_name="delay_ms")
        pct_df["percentile"] = pct_df["percentile"].str.extract(r"_p(\d+)$")[0].apply(lambda x: f"p{x}")
        sns.barplot(data=pct_df, x="algorithm", y="delay_ms", hue="percentile", ax=ax2)
        ax2.legend(title="")
    ax2.set_xlabel("")
    ax2.set_ylabel("Parent→Child Delay (ms)", fontsize=12)
    ax2.set_title("Parent→Child Delay Percentiles", fontsize=14)

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
    Plot latency breakdown showing all latency components.

    Shows where time is spent in the full request path, grouped by load level
    with adjacent bars for each algorithm.

    Args:
        df: Summary DataFrame.
        output_path: Path to save figure.

    Returns:
        matplotlib Figure.
    """
    # Define components and their display properties
    components = [
        ("client_parent_delay_mean", "① Client→Parent", "#3498db"),
        ("parent_child_delay_mean", "② Parent→Child", "#9b59b6"),
        ("simulated_latency_mean", "③ Simulated Work", "#2ecc71"),
        ("child_parent_delay_mean", "④ Child→Parent", "#f39c12"),
        ("parent_client_delay_mean", "⑤ Parent→Client", "#e67e22"),
    ]

    # Filter to columns that exist
    components = [(col, label, color) for col, label, color in components if col in df.columns]

    # Aggregate data by load_level and algorithm
    group_cols = ["load_level", "algorithm"]
    value_cols = [col for col, _, _ in components] + ["latency_mean"]
    agg_df = df.groupby(group_cols)[value_cols].mean().reset_index()

    # Calculate overhead (unexplained latency)
    known_sum = agg_df[[col for col, _, _ in components]].sum(axis=1)
    agg_df["overhead"] = (agg_df["latency_mean"] - known_sum).clip(lower=0)
    components.append(("overhead", "⑥ Other Overhead", "#e74c3c"))

    # Melt to long format for easier plotting
    id_vars = ["load_level", "algorithm"]
    value_vars = [col for col, _, _ in components]
    melted = agg_df.melt(id_vars=id_vars, value_vars=value_vars,
                         var_name="component", value_name="latency_ms")

    # Create component order and color mapping
    comp_order = [col for col, _, _ in components]
    comp_labels = {col: label for col, label, _ in components}
    comp_colors = {col: color for col, _, color in components}

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot stacked bars using pandas pivot
    load_levels = sorted(agg_df["load_level"].unique())
    algorithms = sorted(agg_df["algorithm"].unique())
    n_alg = len(algorithms)
    bar_width = 0.25
    group_spacing = 0.3

    # X positions for each bar
    x_base = np.arange(len(load_levels)) * (n_alg * bar_width + group_spacing)

    for alg_idx, alg in enumerate(algorithms):
        alg_data = agg_df[agg_df["algorithm"] == alg].set_index("load_level")
        x_pos = x_base + alg_idx * bar_width
        bottom = np.zeros(len(load_levels))

        for col, label, color in components:
            values = [alg_data.loc[ll, col] if ll in alg_data.index else 0 for ll in load_levels]
            ax.bar(x_pos, values, bar_width, bottom=bottom, color=color,
                   label=label if alg_idx == 0 else "", edgecolor="white", linewidth=0.5)
            bottom += values

    # Create x-tick labels combining load level and algorithm
    x_tick_positions = []
    x_tick_labels = []
    for ll_idx, ll in enumerate(load_levels):
        for alg_idx, alg in enumerate(algorithms):
            x_tick_positions.append(x_base[ll_idx] + alg_idx * bar_width)
            alg_name = alg.replace("_", " ").title()
            x_tick_labels.append(f"{int(ll * 100)}% - {alg_name}")

    # X-axis labels (load level + algorithm)
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels, fontsize=9, rotation=45, ha="right")
    ax.set_xlabel("Load Level / Algorithm", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("End-to-End Latency Breakdown", fontsize=14)

    # Legend
    ax.legend(loc="upper left", fontsize=9, ncol=2)

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    return fig


def plot_utilization_comparison(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot child replica utilization comparison across algorithms.

    Utilization = total_work_time / (duration × max_ongoing_requests)
    Low utilization + high routing delay indicates router bottleneck.
    Grouped by load level with adjacent bars for each algorithm.

    Args:
        df: Summary DataFrame.
        output_path: Path to save figure.

    Returns:
        matplotlib Figure.
    """
    if "child_util_mean" not in df.columns:
        print("Skipping utilization plot: no utilization data")
        return None

    # Get unique load levels and algorithms
    load_levels = sorted(df["load_level"].unique())
    algorithms = sorted(df["algorithm"].unique())
    n_alg = len(algorithms)
    n_load = len(load_levels)

    # Aggregate data
    agg_df = df.groupby(["load_level", "algorithm"]).agg({
        "child_util_mean": "mean",
        "child_util_min": "mean" if "child_util_min" in df.columns else "first",
    }).reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Bar positioning
    bar_width = 0.25
    group_spacing = 0.3
    x_base = np.arange(n_load) * (n_alg * bar_width + group_spacing)

    colors = sns.color_palette("husl", n_alg)

    for alg_idx, alg in enumerate(algorithms):
        alg_data = agg_df[agg_df["algorithm"] == alg].set_index("load_level")
        x_pos = x_base + alg_idx * bar_width
        
        values = [alg_data.loc[ll, "child_util_mean"] if ll in alg_data.index else 0 
                  for ll in load_levels]
        
        ax.bar(x_pos, values, bar_width, color=colors[alg_idx], 
               label=alg.replace("_", " ").title(), edgecolor="white", linewidth=0.5)

        # Add algorithm label below bars
        for i, x in enumerate(x_pos):
            ax.text(x, -0.03, alg[0].upper(), ha="center", va="top", fontsize=9, color="#555")

    # X-axis labels (load levels)
    ax.set_xticks(x_base + (n_alg - 1) * bar_width / 2)
    ax.set_xticklabels([f"{int(ll * 100)}%" for ll in load_levels], fontsize=11)
    ax.set_xlabel("Load Level", fontsize=12)
    ax.set_ylabel("Child Utilization", fontsize=12)
    ax.set_title("Child Replica Utilization by Load Level", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="100%")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    return fig


def plot_client_to_parent_delay(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot client-to-parent delay comparison across algorithms.

    This measures the time from when the client sends a request to when
    the parent replica receives it. High values indicate router bottleneck
    (e.g., queue length probing blocking the request path).

    Args:
        df: Summary DataFrame.
        output_path: Path to save figure.

    Returns:
        matplotlib Figure.
    """
    if "client_parent_delay_mean" not in df.columns:
        print("Skipping client-to-parent delay plot: no data")
        return None

    # Format algorithm names for display
    plot_df = df.copy()
    plot_df["algorithm"] = plot_df["algorithm"].str.replace("_", " ").str.title()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Mean delay by algorithm
    ax1 = axes[0]
    sns.barplot(data=plot_df, x="algorithm", y="client_parent_delay_mean", ax=ax1,
                errorbar="sd", hue="algorithm", palette="husl", legend=False)
    ax1.set_xlabel("")
    ax1.set_ylabel("Client→Parent Delay (ms)", fontsize=12)
    ax1.set_title("Mean Client→Parent Delay", fontsize=14)

    # Plot 2: Percentiles by algorithm
    ax2 = axes[1]
    pct_cols = ["client_parent_delay_p50", "client_parent_delay_p90", "client_parent_delay_p99"]
    pct_cols = [c for c in pct_cols if c in df.columns]
    if pct_cols:
        pct_df = plot_df.melt(id_vars=["algorithm"], value_vars=pct_cols,
                              var_name="percentile", value_name="delay_ms")
        pct_df["percentile"] = pct_df["percentile"].str.extract(r"_p(\d+)$")[0].apply(lambda x: f"p{x}")
        sns.barplot(data=pct_df, x="algorithm", y="delay_ms", hue="percentile", ax=ax2)
        ax2.legend(title="")
    ax2.set_xlabel("")
    ax2.set_ylabel("Client→Parent Delay (ms)", fontsize=12)
    ax2.set_title("Client→Parent Delay Percentiles", fontsize=14)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    return fig


def plot_replica_coverage(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot replica coverage (unique replica percentage) by algorithm.

    This shows what percentage of expected replicas actually received requests.
    Low coverage indicates routing is not reaching all replicas.

    Args:
        df: Summary DataFrame.
        output_path: Path to save figure.

    Returns:
        matplotlib Figure.
    """
    if "child_unique_pct" not in df.columns:
        print("Skipping replica coverage plot: no data")
        return None

    # Format algorithm names for display
    plot_df = df.copy()
    plot_df["algorithm"] = plot_df["algorithm"].str.replace("_", " ").str.title()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Melt child and parent coverage into long format
    coverage_cols = ["child_unique_pct", "parent_unique_pct"]
    coverage_cols = [c for c in coverage_cols if c in df.columns]
    
    cov_df = plot_df.melt(id_vars=["algorithm"], value_vars=coverage_cols,
                          var_name="replica_type", value_name="coverage_pct")
    cov_df["replica_type"] = cov_df["replica_type"].map({
        "child_unique_pct": "Child", "parent_unique_pct": "Parent"
    })

    sns.barplot(data=cov_df, x="algorithm", y="coverage_pct", hue="replica_type",
                ax=ax, errorbar="sd")
    ax.set_xlabel("")
    ax.set_ylabel("Replica Coverage (%)", fontsize=12)
    ax.set_title("Percentage of Replicas Receiving Requests", fontsize=14)
    ax.set_ylim(0, 105)
    ax.axhline(y=100, color="green", linestyle="--", alpha=0.5, label="100%")
    ax.legend(title="")
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

    # 5. Parent→Child delay comparison
    path = output_dir / "parent_child_delay_comparison.png"
    plot_parent_to_child_delay_comparison(df, path)
    generated.append(path)

    # 6. Latency breakdown (all components)
    path = output_dir / "latency_breakdown.png"
    plot_latency_breakdown(df, path)
    generated.append(path)

    # 7. Client-to-parent delay comparison
    if "client_parent_delay_mean" in df.columns:
        path = output_dir / "client_to_parent_delay.png"
        fig = plot_client_to_parent_delay(df, path)
        if fig:
            generated.append(path)

    # 8. Utilization comparison
    if "child_util_mean" in df.columns:
        path = output_dir / "utilization_comparison.png"
        fig = plot_utilization_comparison(df, path)
        if fig:
            generated.append(path)

    # 9. Replica coverage (unique replica percentage)
    if "child_unique_pct" in df.columns:
        path = output_dir / "replica_coverage.png"
        fig = plot_replica_coverage(df, path)
        if fig:
            generated.append(path)

    # 10. Algorithm heatmap (Large scale only)
    large_df = df[df["scale"] == "large"]
    if len(large_df) > 0:
        path = output_dir / "algorithm_heatmap_large.png"
        plot_algorithm_heatmap(large_df, path)
        generated.append(path)

    # 11. Heatmaps for key metrics
    heatmap_metrics = [
        ("latency_p99", "RdYlGn_r"),
        ("child_jains", "RdYlGn"),
        ("error_rate", "RdYlGn_r"),
        ("parent_child_delay_mean", "RdYlGn_r"),
        ("simulated_latency_mean", "RdYlGn_r"),
    ]

    # Add utilization heatmap if available
    if "child_util_mean" in df.columns:
        heatmap_metrics.append(("child_util_mean", "RdYlGn"))

    for metric, cmap in heatmap_metrics:
        if metric in df.columns:
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
        required=True,
        help="Input CSV file with aggregated metrics (e.g., summary.csv from aggregate.py)",
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


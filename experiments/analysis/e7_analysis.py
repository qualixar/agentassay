"""E7 analysis: Token-Efficient Testing Evaluation.

Reads E7 experiment results, computes summary statistics for each
approach (fixed_n, sprt_only, sprt_fingerprint, sprt_fp_budget,
full_system), and generates comparison tables and figures suitable
for inclusion in the AgentAssay paper.

Usage::

    python -m experiments.analysis.e7_analysis
    python -m experiments.analysis.e7_analysis --results experiments/results/e7/
    python -m experiments.analysis.e7_analysis --figures experiments/figures/e7/

Outputs:
    - Console summary table (Rich)
    - LaTeX table fragment for paper Section 8
    - Bar chart: cost per approach
    - Box plot: trial count distribution per approach
    - Scatter: cost vs. power trade-off
    - Line: cost savings by model capability tier
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "e7"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "experiments" / "figures" / "e7"

APPROACH_ORDER = [
    "fixed_n",
    "sprt_only",
    "sprt_fingerprint",
    "sprt_fp_budget",
    "full_system",
]

APPROACH_LABELS = {
    "fixed_n": "Fixed-N",
    "sprt_only": "SPRT Only",
    "sprt_fingerprint": "SPRT + FP",
    "sprt_fp_budget": "SPRT + FP + Budget",
    "full_system": "Full System",
}

# Model capability tiers (for the tier-savings figure)
MODEL_TIERS: dict[str, str] = {
    "gpt-5.2-chat": "Frontier",
    "claude-sonnet-4-6": "Frontier",
    "claude-haiku-4-5": "Efficient",
    "DeepSeek-R1-0528": "Reasoning",
    "Llama-3.3-70B-Instruct": "Open-Weight",
    "Mistral-Large-3": "Commercial",
    "Llama-4-Maverick-17B-128E-Instruct-FP8": "Open-Weight",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_e7_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load all E7 per-model result files from the results directory.

    Returns a flat list of individual approach results across all
    models, scenarios, and repetitions.
    """
    all_results: list[dict[str, Any]] = []

    if not results_dir.exists():
        logger.error("E7 results directory not found: %s", results_dir)
        return []

    for path in sorted(results_dir.glob("*.json")):
        if path.name == "summary.json":
            continue

        try:
            with open(path) as f:
                data = json.load(f)

            results = data.get("results", [])
            all_results.extend(results)

        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load %s: %s", path.name, exc)

    logger.info(
        "Loaded %d approach results from %s",
        len(all_results),
        results_dir,
    )
    return all_results


# ---------------------------------------------------------------------------
# Statistical aggregation
# ---------------------------------------------------------------------------

def compute_approach_stats(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute summary statistics per approach.

    For each approach, computes mean +/- std for:
        - trials_used
        - tokens_used
        - cost_usd (candidate only, excluding baseline)
        - power (regression detection rate)

    Returns a dict keyed by approach name.
    """
    by_approach: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        approach = r.get("approach")
        if approach and r.get("verdict") != "ERROR":
            by_approach[approach].append(r)

    stats: dict[str, dict[str, Any]] = {}

    for approach in APPROACH_ORDER:
        entries = by_approach.get(approach, [])
        if not entries:
            stats[approach] = {
                "n": 0,
                "trials_mean": 0.0,
                "trials_std": 0.0,
                "tokens_mean": 0.0,
                "tokens_std": 0.0,
                "cost_mean": 0.0,
                "cost_std": 0.0,
                "power": 0.0,
                "power_std": 0.0,
                "inconclusive_rate": 0.0,
            }
            continue

        trials = np.array([e.get("trials_used", 0) for e in entries], dtype=float)
        tokens = np.array([e.get("tokens_used", 0) for e in entries], dtype=float)
        costs = np.array([e.get("cost_usd", 0.0) for e in entries], dtype=float)
        detected = np.array(
            [1.0 if e.get("verdict") == "FAIL" else 0.0 for e in entries],
            dtype=float,
        )
        inconclusive = np.array(
            [1.0 if e.get("verdict") == "INCONCLUSIVE" else 0.0 for e in entries],
            dtype=float,
        )

        stats[approach] = {
            "n": len(entries),
            "trials_mean": float(np.mean(trials)),
            "trials_std": float(np.std(trials, ddof=1)) if len(trials) > 1 else 0.0,
            "trials_median": float(np.median(trials)),
            "tokens_mean": float(np.mean(tokens)),
            "tokens_std": float(np.std(tokens, ddof=1)) if len(tokens) > 1 else 0.0,
            "cost_mean": float(np.mean(costs)),
            "cost_std": float(np.std(costs, ddof=1)) if len(costs) > 1 else 0.0,
            "power": float(np.mean(detected)),
            "power_std": float(np.std(detected, ddof=1)) if len(detected) > 1 else 0.0,
            "inconclusive_rate": float(np.mean(inconclusive)),
        }

    return stats


def compute_model_stats(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compute per-model, per-approach statistics.

    Returns a nested dict: model -> approach -> stats.
    """
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        model = r.get("model")
        if model and r.get("verdict") != "ERROR":
            by_model[model].append(r)

    model_stats: dict[str, dict[str, dict[str, Any]]] = {}
    for model, entries in by_model.items():
        model_stats[model] = compute_approach_stats(entries)

    return model_stats


def compute_tier_savings(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Compute cost savings per model capability tier.

    For each tier, computes the average cost savings of each approach
    relative to fixed_n. Returns: tier -> approach -> savings_pct.
    """
    # Group by tier
    by_tier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        model = r.get("model", "")
        tier = MODEL_TIERS.get(model, "Unknown")
        if r.get("verdict") != "ERROR":
            by_tier[tier].append(r)

    tier_savings: dict[str, dict[str, float]] = {}

    for tier, entries in by_tier.items():
        tier_stats = compute_approach_stats(entries)
        baseline_cost = tier_stats.get("fixed_n", {}).get("cost_mean", 1.0)
        if baseline_cost <= 0:
            baseline_cost = 1.0  # avoid division by zero

        savings: dict[str, float] = {}
        for approach in APPROACH_ORDER:
            approach_cost = tier_stats.get(approach, {}).get("cost_mean", 0.0)
            savings[approach] = (1.0 - approach_cost / baseline_cost) * 100.0

        tier_savings[tier] = savings

    return tier_savings


# ---------------------------------------------------------------------------
# Console output (Rich)
# ---------------------------------------------------------------------------

def print_summary_table(stats: dict[str, dict[str, Any]]) -> None:
    """Print a Rich-formatted summary table to the console."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(
            title="E7: Token-Efficient Testing — Approach Comparison",
            show_lines=True,
        )

        table.add_column("Approach", style="bold")
        table.add_column("N", justify="right")
        table.add_column("Trials (mean +/- std)", justify="right")
        table.add_column("Tokens (mean)", justify="right")
        table.add_column("Cost USD (mean +/- std)", justify="right")
        table.add_column("Power", justify="right")
        table.add_column("Inconcl.", justify="right")

        for approach in APPROACH_ORDER:
            s = stats.get(approach, {})
            if s.get("n", 0) == 0:
                table.add_row(APPROACH_LABELS.get(approach, approach), "0", "-", "-", "-", "-", "-")
                continue

            table.add_row(
                APPROACH_LABELS.get(approach, approach),
                str(s["n"]),
                f"{s['trials_mean']:.1f} +/- {s['trials_std']:.1f}",
                f"{s['tokens_mean']:.0f}",
                f"${s['cost_mean']:.4f} +/- ${s['cost_std']:.4f}",
                f"{s['power']:.1%}",
                f"{s['inconclusive_rate']:.1%}",
            )

        console.print(table)

    except ImportError:
        # Fallback: plain text
        print("\nE7: Token-Efficient Testing — Approach Comparison")
        print("-" * 90)
        header = f"{'Approach':<20} {'N':>5} {'Trials':>12} {'Cost USD':>14} {'Power':>8} {'Inconcl':>8}"
        print(header)
        print("-" * 90)

        for approach in APPROACH_ORDER:
            s = stats.get(approach, {})
            if s.get("n", 0) == 0:
                print(f"{APPROACH_LABELS.get(approach, approach):<20} {'0':>5} {'-':>12} {'-':>14} {'-':>8} {'-':>8}")
                continue

            print(
                f"{APPROACH_LABELS.get(approach, approach):<20} "
                f"{s['n']:>5} "
                f"{s['trials_mean']:>5.1f}+/-{s['trials_std']:<5.1f} "
                f"${s['cost_mean']:>6.4f}+/-{s['cost_std']:<6.4f} "
                f"{s['power']:>7.1%} "
                f"{s['inconclusive_rate']:>7.1%}"
            )
        print("-" * 90)


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

def generate_latex_table(
    stats: dict[str, dict[str, Any]],
    output_path: Path | None = None,
) -> str:
    """Generate a LaTeX table fragment for paper Section 8.

    The table matches the paper format: one row per approach, columns
    for trials, cost, power, and savings relative to fixed-N.
    """
    baseline_cost = stats.get("fixed_n", {}).get("cost_mean", 1.0)
    if baseline_cost <= 0:
        baseline_cost = 1.0

    lines: list[str] = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{E7: Cost efficiency of testing approaches. Mean $\pm$ std over 50 repetitions $\times$ 7 models $\times$ 4 scenarios.}",
        r"\label{tab:e7-efficiency}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Approach & Trials & Cost (USD) & Power & Savings \\",
        r"\midrule",
    ]

    for approach in APPROACH_ORDER:
        s = stats.get(approach, {})
        label = APPROACH_LABELS.get(approach, approach)

        if s.get("n", 0) == 0:
            lines.append(f"{label} & -- & -- & -- & -- \\\\")
            continue

        approach_cost = s["cost_mean"]
        savings = (1.0 - approach_cost / baseline_cost) * 100.0

        lines.append(
            f"{label} & "
            f"${s['trials_mean']:.1f} \\pm {s['trials_std']:.1f}$ & "
            f"\\${s['cost_mean']:.4f} \\pm {s['cost_std']:.4f}$ & "
            f"${s['power']:.1%}$ & "
            f"${savings:+.1f}\\%$ \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex)
        logger.info("LaTeX table written: %s", output_path)

    return latex


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def generate_figures(
    results: list[dict[str, Any]],
    stats: dict[str, dict[str, Any]],
    tier_savings: dict[str, dict[str, float]],
    figures_dir: Path,
) -> list[Path]:
    """Generate all E7 figures. Returns list of paths to generated files.

    Figures:
        1. Bar chart: mean cost per approach with error bars
        2. Box plot: trial count distribution per approach
        3. Scatter: cost vs. power for each approach
        4. Line: cost savings by model capability tier
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "matplotlib not available — skipping figure generation. "
            "Install with: pip install matplotlib"
        )
        return []

    # Shared style
    plt.rcParams.update({
        "font.size": 11,
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })

    # ---- Figure 1: Bar chart — cost per approach ----
    try:
        fig, ax = plt.subplots()
        approaches = APPROACH_ORDER
        labels = [APPROACH_LABELS[a] for a in approaches]
        means = [stats.get(a, {}).get("cost_mean", 0.0) for a in approaches]
        stds = [stats.get(a, {}).get("cost_std", 0.0) for a in approaches]

        colors = ["#4e79a7", "#59a14f", "#f28e2b", "#e15759", "#76b7b2"]
        bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, edgecolor="black", linewidth=0.5)

        ax.set_ylabel("Cost (USD)")
        ax.set_title("E7: Mean Cost per Approach")
        ax.tick_params(axis="x", rotation=20)

        # Add value labels
        for bar_item, mean_val in zip(bars, means):
            ax.text(
                bar_item.get_x() + bar_item.get_width() / 2,
                bar_item.get_height() + max(stds) * 0.1,
                f"${mean_val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        path = figures_dir / "e7_cost_per_approach.png"
        fig.savefig(path)
        plt.close(fig)
        generated.append(path)
        logger.info("Generated: %s", path.name)
    except Exception as exc:
        logger.error("Failed to generate cost bar chart: %s", exc)

    # ---- Figure 2: Box plot — trial count distribution ----
    try:
        fig, ax = plt.subplots()

        trial_data: list[list[float]] = []
        labels_box: list[str] = []

        for approach in APPROACH_ORDER:
            entries = [
                r.get("trials_used", 0)
                for r in results
                if r.get("approach") == approach and r.get("verdict") != "ERROR"
            ]
            if entries:
                trial_data.append(entries)
                labels_box.append(APPROACH_LABELS[approach])

        if trial_data:
            bp = ax.boxplot(
                trial_data,
                labels=labels_box,
                patch_artist=True,
                showmeans=True,
                meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 5},
            )

            colors = ["#4e79a7", "#59a14f", "#f28e2b", "#e15759", "#76b7b2"]
            for patch, color in zip(bp["boxes"], colors[: len(bp["boxes"])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_ylabel("Trials Used")
            ax.set_title("E7: Trial Count Distribution per Approach")
            ax.tick_params(axis="x", rotation=20)

        path = figures_dir / "e7_trial_boxplot.png"
        fig.savefig(path)
        plt.close(fig)
        generated.append(path)
        logger.info("Generated: %s", path.name)
    except Exception as exc:
        logger.error("Failed to generate trial box plot: %s", exc)

    # ---- Figure 3: Scatter — cost vs. power ----
    try:
        fig, ax = plt.subplots()

        colors_scatter = {
            "fixed_n": "#4e79a7",
            "sprt_only": "#59a14f",
            "sprt_fingerprint": "#f28e2b",
            "sprt_fp_budget": "#e15759",
            "full_system": "#76b7b2",
        }

        for approach in APPROACH_ORDER:
            s = stats.get(approach, {})
            if s.get("n", 0) == 0:
                continue

            ax.scatter(
                s["cost_mean"],
                s["power"],
                s=120,
                c=colors_scatter.get(approach, "gray"),
                label=APPROACH_LABELS[approach],
                edgecolors="black",
                linewidth=0.5,
                zorder=3,
            )

            # Error bars
            ax.errorbar(
                s["cost_mean"],
                s["power"],
                xerr=s["cost_std"],
                yerr=s["power_std"],
                fmt="none",
                ecolor="gray",
                alpha=0.5,
                capsize=3,
            )

        ax.set_xlabel("Mean Cost (USD)")
        ax.set_ylabel("Regression Detection Power")
        ax.set_title("E7: Cost vs. Power Trade-off")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_ylim(-0.05, 1.15)
        ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.3, label="80% power")

        path = figures_dir / "e7_cost_vs_power.png"
        fig.savefig(path)
        plt.close(fig)
        generated.append(path)
        logger.info("Generated: %s", path.name)
    except Exception as exc:
        logger.error("Failed to generate cost vs power scatter: %s", exc)

    # ---- Figure 4: Line — cost savings by tier ----
    try:
        fig, ax = plt.subplots()

        tier_order = ["Frontier", "Commercial", "Reasoning", "Efficient", "Open-Weight"]
        available_tiers = [t for t in tier_order if t in tier_savings]

        if available_tiers:
            x = np.arange(len(available_tiers))
            width = 0.15

            for i, approach in enumerate(APPROACH_ORDER[1:]):  # Skip fixed_n (0% savings)
                savings_vals = [
                    tier_savings.get(tier, {}).get(approach, 0.0)
                    for tier in available_tiers
                ]
                offset = (i - 1.5) * width
                ax.bar(
                    x + offset,
                    savings_vals,
                    width=width,
                    label=APPROACH_LABELS[approach],
                    color=["#59a14f", "#f28e2b", "#e15759", "#76b7b2"][i],
                    edgecolor="black",
                    linewidth=0.3,
                )

            ax.set_xlabel("Model Capability Tier")
            ax.set_ylabel("Cost Savings vs Fixed-N (%)")
            ax.set_title("E7: Cost Savings by Model Tier")
            ax.set_xticks(x)
            ax.set_xticklabels(available_tiers, rotation=15)
            ax.legend(fontsize=8, loc="upper left")
            ax.axhline(y=0, color="black", linewidth=0.5)

        path = figures_dir / "e7_tier_savings.png"
        fig.savefig(path)
        plt.close(fig)
        generated.append(path)
        logger.info("Generated: %s", path.name)
    except Exception as exc:
        logger.error("Failed to generate tier savings chart: %s", exc)

    return generated


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_summary_json(
    stats: dict[str, dict[str, Any]],
    model_stats: dict[str, dict[str, dict[str, Any]]],
    tier_savings: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    """Export the full analysis as a structured JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "experiment": "e7_efficiency",
        "aggregate_stats": stats,
        "per_model_stats": model_stats,
        "tier_savings": tier_savings,
        "approach_order": APPROACH_ORDER,
        "approach_labels": APPROACH_LABELS,
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    logger.info("JSON summary written: %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    latex_output: Path | None = None,
) -> None:
    """Run the full E7 analysis pipeline.

    1. Load results from disk.
    2. Compute aggregate and per-model statistics.
    3. Print summary table to console.
    4. Generate LaTeX table fragment.
    5. Generate all figures.
    6. Export JSON summary.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("E7 Analysis: Token-Efficient Testing Evaluation")
    logger.info("Results dir: %s", results_dir)
    logger.info("Figures dir: %s", figures_dir)
    logger.info("=" * 60)

    # Load
    results = load_e7_results(results_dir)
    if not results:
        logger.error("No E7 results found. Run the experiment first.")
        sys.exit(1)

    # Compute
    stats = compute_approach_stats(results)
    model_stats = compute_model_stats(results)
    tier_savings = compute_tier_savings(results)

    # Console
    print_summary_table(stats)

    # LaTeX
    if latex_output is None:
        latex_output = figures_dir / "e7_table.tex"
    latex = generate_latex_table(stats, latex_output)
    print("\n--- LaTeX Table ---")
    print(latex)

    # Figures
    generated = generate_figures(results, stats, tier_savings, figures_dir)
    if generated:
        logger.info("Generated %d figures in %s", len(generated), figures_dir)

    # JSON export
    json_path = results_dir / "e7_analysis_summary.json"
    export_summary_json(stats, model_stats, tier_savings, json_path)

    logger.info("=" * 60)
    logger.info("E7 analysis complete.")
    logger.info("=" * 60)


def cli() -> None:
    """CLI entry point for E7 analysis."""
    parser = argparse.ArgumentParser(
        prog="e7_analysis",
        description="Analyze E7 (Token-Efficient Testing) experiment results.",
    )
    parser.add_argument(
        "--results",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Path to E7 results directory",
    )
    parser.add_argument(
        "--figures",
        type=str,
        default=str(DEFAULT_FIGURES_DIR),
        help="Path to figures output directory",
    )
    parser.add_argument(
        "--latex",
        type=str,
        default=None,
        help="Path to LaTeX table output file",
    )

    args = parser.parse_args()

    main(
        results_dir=Path(args.results),
        figures_dir=Path(args.figures),
        latex_output=Path(args.latex) if args.latex else None,
    )


if __name__ == "__main__":
    cli()

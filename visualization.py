"""
Visualizations for the Voucher ROI Product Science portfolio.

Two figures, both referenced in README.md:
1. Tier x lever heatmap — incremental profit_per_user for each named lever
   comparison (voucher vs none / min-spend / copay / count), by tier.
2. Budget efficiency frontier — total incremental profit achievable as the
   subsidy budget scales from $0 to well beyond the actual NT$1,000,000 cap,
   showing where the current budget sits on the curve and where returns
   start flattening out.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from budget_allocator import build_tier_condition_table, greedy_fractional_allocation, BUDGET_USD
from experiment_analysis import run_named_comparisons, NAMED_COMPARISONS, TIERS_ASC


def plot_tier_lever_heatmap(comparisons: pd.DataFrame, save_path: str) -> None:
    """Plot incremental profit_per_user by tier x named lever comparison.

    Args:
        comparisons: Output of `experiment_analysis.run_named_comparisons`.
        save_path: File path to save the figure to.
    """
    profit_rows = comparisons[comparisons["metric"] == "profit_per_user"]
    pivot = profit_rows.pivot(index="comparison", columns="tier", values="pct_lift")
    pivot = pivot.reindex(columns=TIERS_ASC)
    sig_pivot = profit_rows.pivot(index="comparison", columns="tier", values="significant")
    sig_pivot = sig_pivot.reindex(columns=TIERS_ASC)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    vmax = np.nanmax(np.abs(pivot.to_numpy()))
    im = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Usage tier")
    ax.set_title("% lift in profit_per_user by lever comparison and tier\n(* = FDR-significant at q<0.05)")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.isna(value):
                continue
            star = "*" if sig_pivot.iloc[i, j] else ""
            ax.text(j, i, f"{value:.1f}%{star}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="% lift", shrink=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_budget_efficiency_frontier(dgp, calibration: pd.DataFrame, save_path: str, max_budget: float) -> None:
    """Plot total incremental profit achievable as the budget scales up.

    Args:
        dgp: A fitted VoucherDGP instance (passed in rather than constructed
            here, so callers control the calibration path — needed when this
            is invoked from a notebook in a subdirectory).
        calibration: Calibration targets DataFrame.
        save_path: File path to save the figure to.
        max_budget: Upper end of the budget range to sweep, in USD.
    """
    econ_table = build_tier_condition_table(dgp, calibration)

    budgets = np.linspace(0, max_budget, 60)
    profits = []
    for b in budgets:
        _, summary = greedy_fractional_allocation(econ_table, budget=b)
        profits.append(summary["total_incremental_profit"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(budgets, profits, color="#6366F1", linewidth=2)
    ax.axvline(BUDGET_USD, color="#F59E0B", linestyle="--", label=f"Actual budget (${BUDGET_USD:,.0f})")
    ax.set_xlabel("Subsidy budget (USD)")
    ax.set_ylabel("Total incremental profit (USD)")
    ax.set_title("Budget efficiency frontier: diminishing returns as spend scales")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    import os

    os.makedirs("outputs/figures", exist_ok=True)

    df = pd.read_csv("data/processed/experiment_log.csv")
    comparisons = run_named_comparisons(df)
    plot_tier_lever_heatmap(comparisons, "outputs/figures/tier_lever_heatmap.png")
    print("Saved outputs/figures/tier_lever_heatmap.png")

    calibration = pd.read_csv("data/raw_benchmarks/case_summary_tables.csv")
    from data_generation import VoucherDGP

    dgp = VoucherDGP()
    plot_budget_efficiency_frontier(
        dgp, calibration, "outputs/figures/budget_efficiency_frontier.png", max_budget=BUDGET_USD * 5
    )
    print("Saved outputs/figures/budget_efficiency_frontier.png")


if __name__ == "__main__":
    main()

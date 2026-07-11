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


def plot_forest_comparison(
    comparisons_small: pd.DataFrame,
    comparisons_big: pd.DataFrame,
    save_path: str,
    label_small: str = "N=200k",
    label_big: str = "N=750k",
) -> None:
    """Forest plot of order-rate lifts with 95% CIs, small-N vs. powered-N.

    One row per comparison x tier, two panels side by side: the same 24
    estimates at the original sample size (CIs mostly spanning zero) and at
    the power-analysis-prescribed size (large true effects separating from
    zero). This is the visual form of Finding #1's before/after evidence.

    Args:
        comparisons_small: `run_named_comparisons` output at the original N.
        comparisons_big: Same, at the adequately-powered N.
        save_path: File path to save the figure to.
        label_small: Panel title for the original-N side.
        label_big: Panel title for the powered-N side.
    """
    def _order_rows(comp: pd.DataFrame) -> pd.DataFrame:
        rows = comp[comp["metric"] == "order_rate"].copy()
        rows["label"] = rows["comparison"] + " | " + rows["tier"]
        return rows.set_index("label")

    small = _order_rows(comparisons_small)
    big = _order_rows(comparisons_big).reindex(small.index)
    y = np.arange(len(small))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(11, 8), sharey=True)
    for ax, data, title in [(axes[0], small, label_small), (axes[1], big, label_big)]:
        colors = np.where(data["significant"], "#DC2626", "#6366F1")
        ax.hlines(y, data["ci_low"], data["ci_high"], color=colors, linewidth=1.6)
        ax.scatter(data["abs_diff"], y, color=colors, s=18, zorder=3)
        ax.axvline(0, color="#9CA3AF", linestyle="--", linewidth=1)
        n_sig = int(data["significant"].sum())
        ax.set_title(f"{title} — {n_sig}/24 significant (FDR)")
        ax.set_xlabel("Absolute order-rate difference (95% CI)")
        ax.grid(axis="x", alpha=0.25)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(small.index, fontsize=7)
    fig.suptitle("Same experiment, two sample sizes: the power diagnosis, visualized", y=0.995)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_calibration_scatter(dgp, save_path: str, tolerance_pct: float = 15.0) -> None:
    """Scatter of simulated vs. target aggregates with a ±tolerance band.

    Every (tier, condition, metric) cell from the aggregation-recovery check
    is one point; the 45° line is perfect reproduction and the shaded band
    is the ±15% tolerance used in validation.py. Points outside the band are
    annotated — making '41/42 order cells recovered' and 'residuals
    concentrate in the 90-99% tier' visible at a glance.

    Args:
        dgp: A fitted VoucherDGP instance.
        save_path: File path to save the figure to.
        tolerance_pct: Half-width of the tolerance band, in percent.
    """
    from validation import aggregation_recovery_check

    recovery = aggregation_recovery_check(dgp)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    panels = [
        ("order_per_user", "target_order", "order_within_tolerance", "Order per user"),
        ("profit_per_user", "target_profit", "profit_within_tolerance", "Profit per user"),
    ]
    for ax, (sim_col, target_col, tol_col, title) in zip(axes, panels):
        target = recovery[target_col].to_numpy()
        simulated = recovery[sim_col].to_numpy()
        within = recovery[tol_col].to_numpy()

        lo, hi = 0, max(target.max(), simulated.max()) * 1.1
        line = np.linspace(lo, hi, 50)
        ax.plot(line, line, color="#9CA3AF", linewidth=1, label="Perfect reproduction")
        ax.fill_between(
            line, line * (1 - tolerance_pct / 100), line * (1 + tolerance_pct / 100),
            color="#6366F1", alpha=0.12, label=f"±{tolerance_pct:.0f}% tolerance",
        )
        ax.scatter(target[within], simulated[within], color="#6366F1", s=24, zorder=3)
        ax.scatter(
            target[~within], simulated[~within], color="#DC2626", s=36, zorder=4, marker="D"
        )
        for _, row in recovery[~recovery[tol_col]].iterrows():
            ax.annotate(
                f"{row['tier']} | {row['condition']}",
                (row[target_col], row[sim_col]),
                textcoords="offset points", xytext=(6, -4), fontsize=7, color="#DC2626",
            )
        n_ok, n_all = int(within.sum()), len(within)
        ax.set_title(f"{title} — {n_ok}/{n_all} cells within ±{tolerance_pct:.0f}%")
        ax.set_xlabel("Target (transcribed source table)")
        ax.set_ylabel("Simulated (large-N aggregate)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle("Calibration check: does the simulation reproduce the known aggregates?", y=1.0)
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

    big_log = dgp.simulate_users(n_users=750_000, seed=7)
    big_comparisons = run_named_comparisons(big_log)
    plot_forest_comparison(
        comparisons, big_comparisons, "outputs/figures/forest_200k_vs_750k.png"
    )
    print("Saved outputs/figures/forest_200k_vs_750k.png")

    plot_calibration_scatter(dgp, "outputs/figures/calibration_scatter.png")
    print("Saved outputs/figures/calibration_scatter.png")


if __name__ == "__main__":
    main()

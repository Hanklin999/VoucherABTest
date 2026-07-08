# %% [markdown]
# # 04 — Budget Optimization
#
# Allocates a fixed subsidy budget across tier x voucher-config options via
# a greedy fractional knapsack (optimal for the LP relaxation), benchmarks
# against the PDF's 3 named strategies, and stress-tests the recommendation
# against the 3 unvalidated business assumptions. See `budget_allocator.py`
# for full docstrings and the framing note on why this is a hard-budget-cap
# problem, not a "spend until marginal ROI hits zero" problem.

# %%
import sys

sys.path.insert(0, "..")

import pandas as pd

from data_generation import VoucherDGP
from budget_allocator import (
    build_tier_condition_table,
    greedy_fractional_allocation,
    evaluate_fixed_strategy,
    sensitivity_analysis,
    BUDGET_TWD,
    BUDGET_USD,
    TOTAL_ADDRESSABLE_USERS,
)
from visualization import plot_tier_lever_heatmap, plot_budget_efficiency_frontier
from experiment_analysis import run_named_comparisons

calibration = pd.read_csv("../data/raw_benchmarks/case_summary_tables.csv")
experiment_log = pd.read_csv("../data/processed/experiment_log.csv")
dgp = VoucherDGP(calibration_path="../data/raw_benchmarks/case_summary_tables.csv")

print(f"Budget: NT${BUDGET_TWD:,.0f} = ${BUDGET_USD:,.2f} USD")
print(f"Addressable population: {TOTAL_ADDRESSABLE_USERS:,} users")

# %% [markdown]
# ## Economics table: expected profit, cost, and ROI ratio per (tier, condition)

# %%
econ_table = build_tier_condition_table(dgp, calibration)
econ_table[
    ["tier", "condition", "profit_per_user", "cost_per_user", "incremental_profit_per_user", "roi_ratio"]
].sort_values(["tier", "roi_ratio"], ascending=[True, False])

# %% [markdown]
# ## Our allocation: greedy fractional knapsack

# %%
alloc_df, summary = greedy_fractional_allocation(econ_table)
alloc_df

# %%
print(f"Total cost: ${summary['total_cost']:,.2f} ({summary['budget_utilization_pct']:.1f}% of budget)")
print(f"Total incremental profit: ${summary['total_incremental_profit']:,.2f}")
print(f"Blended ROI: {summary['blended_roi_pct']:.1f}%")

# %% [markdown]
# ## Benchmark: PDF's 3 named strategies at the same budget

# %%
scenario_1 = evaluate_fixed_strategy(econ_table, ["90-99"], "s0_m249_c3", BUDGET_USD)
scenario_2 = evaluate_fixed_strategy(econ_table, ["30-69"], "s9_m249_c3", BUDGET_USD)
scenario_3_high = evaluate_fixed_strategy(econ_table, ["90-99"], "s0_m249_c3", BUDGET_USD * 0.6)
scenario_3_mod = evaluate_fixed_strategy(econ_table, ["30-69"], "s9_m249_c3", BUDGET_USD * 0.4)
scenario_3 = {
    "total_cost": scenario_3_high["total_cost"] + scenario_3_mod["total_cost"],
    "total_incremental_profit": scenario_3_high["total_incremental_profit"]
    + scenario_3_mod["total_incremental_profit"],
    "users_reached": scenario_3_high["users_reached"] + scenario_3_mod["users_reached"],
}
scenario_3["blended_roi_pct"] = (
    round(scenario_3["total_incremental_profit"] / scenario_3["total_cost"] * 100, 1)
    if scenario_3["total_cost"] > 0
    else 0.0
)

benchmark_table = pd.DataFrame(
    [
        {"strategy": "PDF Scenario 1 (order-volume max)", **scenario_1},
        {"strategy": "PDF Scenario 2 (profit-per-user max)", **scenario_2},
        {"strategy": "PDF Scenario 3 (60/40 blended)", **scenario_3},
        {"strategy": "This project's ROI-ranked allocation", **summary},
    ]
)
benchmark_table

# %% [markdown]
# ## Sensitivity analysis: does the recommendation hold under different assumptions?

# %%
sensitivity = sensitivity_analysis(calibration, calibration_path="../data/raw_benchmarks/case_summary_tables.csv")
sensitivity.to_csv("../outputs/sensitivity_analysis.csv", index=False)
sensitivity

# %%
n_unique = sensitivity["top_allocated_segment"].nunique()
print(
    f"Top-allocated segment is identical across all {len(sensitivity)} assumption "
    f"combinations tested ({n_unique} unique choice)"
    if n_unique == 1
    else f"Top-allocated segment varies across {n_unique} different choices — recommendation is assumption-sensitive."
)

# %% [markdown]
# ## Visualizations

# %%
comparisons = run_named_comparisons(experiment_log)
plot_tier_lever_heatmap(comparisons, "../outputs/figures/tier_lever_heatmap.png")
plot_budget_efficiency_frontier(
    dgp, calibration, "../outputs/figures/budget_efficiency_frontier.png", max_budget=BUDGET_USD * 5
)
print("Figures saved to outputs/figures/")

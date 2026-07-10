# %% [markdown]
# # 02 — Experiment Analysis
#
# FDR-corrected significance tests mirroring the source case study's 4 Key Insights, an
# interaction-regression cross-check, and a power analysis explaining why
# nothing survives correction at the current sample size. See
# `experiment_analysis.py` for full docstrings.

# %%
import sys

sys.path.insert(0, "..")

import pandas as pd

from experiment_analysis import (
    run_named_comparisons,
    fit_interaction_models,
    required_sample_size,
    NAMED_COMPARISONS,
    TIERS_ASC,
)

experiment_log = pd.read_csv("../data/processed/experiment_log.csv")

# %% [markdown]
# ## Named comparisons (mirrors the source case study's 4 Key Insights), FDR-corrected

# %%
comparisons = run_named_comparisons(experiment_log)
comparisons.to_csv("../outputs/named_comparisons.csv", index=False)
comparisons[["comparison", "tier", "metric", "pct_lift", "p_value", "q_value", "significant"]]

# %% [markdown]
# **Headline finding: 0 of 48 comparisons are significant after FDR
# correction**, despite most trending in the direction the source case study reports.
# This is the most important result in this notebook, not a footnote —
# see the power analysis below for why.

# %% [markdown]
# ## Interaction regression (tier x condition), cross-check

# %%
order_summary, profit_summary = fit_interaction_models(experiment_log)
print("Order (logit) — top 10 terms by |coef|:")
order_summary.reindex(order_summary["coef"].abs().sort_values(ascending=False).index).head(10)

# %%
print("Profit (OLS) — top 10 terms by |coef|:")
profit_summary.reindex(profit_summary["coef"].abs().sort_values(ascending=False).index).head(10)

# %% [markdown]
# ## Sample size needed: worked example (90-99% tier, voucher_vs_none)

# %%
n_comparisons = len(NAMED_COMPARISONS) * len(TIERS_ASC)
example = comparisons[
    (comparisons["comparison"] == "voucher_vs_none")
    & (comparisons["tier"] == "90-99")
    & (comparisons["metric"] == "order_rate")
].iloc[0]
n_needed = required_sample_size(
    baseline_rate=example["mean_b"], pct_lift=example["pct_lift"], n_comparisons=n_comparisons
)
current_n = experiment_log[
    (experiment_log["tier"] == "90-99") & (experiment_log["condition"] == "no_voucher")
].shape[0]

print(f"Observed lift: {example['pct_lift']:.1f}% on a {example['mean_b']:.3f} baseline order rate")
print(f"Required N per arm (80% power, Bonferroni-corrected for {n_comparisons} tests): {n_needed:,}")
print(f"Current simulated N in this cell: {current_n:,}")
print("UNDERPOWERED" if current_n < n_needed else "Adequately powered")

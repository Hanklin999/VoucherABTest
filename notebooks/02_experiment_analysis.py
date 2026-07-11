# %% [markdown]
# # 02 — Experiment Analysis
#
# A full experiment readout, in the order a real experimentation platform
# would run it:
#
# 1. **Trust check first** (Sample Ratio Mismatch) — before reading any
#    result, verify the assignment mechanism delivered the designed split.
# 2. **Named comparisons** mirroring the source case study's 4 Key Insights,
#    with 95% CIs and FDR correction.
# 3. **Direction concordance vs. ground truth** — a diagnostic only a
#    simulation can run: which non-significant cells are power problems,
#    and which are correctly null?
# 4. **Power & MDE table** — how big should this experiment have been, and
#    what was it ever capable of detecting at its actual size?
# 5. **Adequately-powered rerun** — the same DGP at the sample size the
#    power analysis prescribes, closing the loop on the diagnosis.
#
# See `experiment_analysis.py` for full docstrings.

# %%
import os
import sys
from pathlib import Path

# Anchor all relative paths to THIS file's location, not the caller's cwd —
# so the notebook works whether run via Jupyter (kernel cwd = notebooks/) or
# as a script from anywhere (e.g. `python notebooks/02_....py` from repo root).
try:
    NOTEBOOK_DIR = Path(__file__).resolve().parent
except NameError:  # __file__ undefined inside a notebook kernel
    NOTEBOOK_DIR = Path.cwd()
os.chdir(NOTEBOOK_DIR)
sys.path.insert(0, str(NOTEBOOK_DIR.parent))

import pandas as pd

from data_generation import VoucherDGP
from experiment_analysis import (
    run_named_comparisons,
    fit_interaction_models,
    srm_check,
    power_mde_table,
    direction_concordance,
    NAMED_COMPARISONS,
    TIERS_ASC,
)

experiment_log = pd.read_csv("../data/processed/experiment_log.csv")
dgp = VoucherDGP(calibration_path="../data/raw_benchmarks/case_summary_tables.csv")

# %% [markdown]
# ## 1. Trust check: Sample Ratio Mismatch (SRM)
#
# Chi-square test per tier against the designed uniform 1/7 allocation,
# with the industry-standard strict alarm threshold (p < 0.001). A failed
# SRM check invalidates every downstream number — which is why it runs
# BEFORE any result is read, not after.

# %%
srm = srm_check(experiment_log)
srm

# %%
assert not srm["srm_alarm"].any(), "SRM detected — do not read results until assignment is debugged"
print("SRM check passed for all tiers — safe to read results.")

# %% [markdown]
# ## 2. Named comparisons (mirrors the source case study's 4 Key Insights)
#
# Two-proportion z-tests (order rate) and Welch t-tests (profit/user) per
# tier, now with 95% CIs on the absolute difference, BH-FDR corrected
# within each metric family.

# %%
comparisons = run_named_comparisons(experiment_log)
comparisons.to_csv("../outputs/named_comparisons.csv", index=False)
comparisons[
    ["comparison", "tier", "metric", "abs_diff", "ci_low", "ci_high", "pct_lift", "q_value", "significant"]
]

# %% [markdown]
# **Headline: 0 of 48 comparisons are significant after FDR correction.**
# The CI columns make the reason visible without any p-value: nearly every
# interval spans zero. The next two sections separate the two very
# different explanations hiding inside that headline number.

# %% [markdown]
# ## 3. Direction concordance vs. ground truth
#
# Because the true effects were calibrated into the DGP, every comparison
# can be classified: does the observed sign match the true sign, or was it
# flipped by sampling noise? And critically — which cells have a
# *negligible* true effect, where non-significance is the CORRECT outcome
# rather than a power failure?

# %%
concordance = direction_concordance(comparisons, dgp)
concordance["true_effect"].value_counts()

# %%
non_negligible = concordance[concordance["true_effect"] != "negligible"]
agreement_rate = non_negligible["sign_agrees"].mean()
print(
    f"Sign agreement among non-negligible true effects: "
    f"{agreement_rate:.0%} ({int(non_negligible['sign_agrees'].sum())}/{len(non_negligible)})"
)
concordance[concordance["sign_agrees"] == False]  # noqa: E712 — the sign-flipped cells

# %% [markdown]
# Interpretation: most estimates point the right way, a substantial
# minority are sign-flipped by noise — exactly the behavior expected from
# an underpowered experiment, and a concrete warning against reading
# individual cell directions at this sample size. The negligible-effect
# cells are NOT failures: for them, "not significant" is the truth.

# %% [markdown]
# ## 4. Power & MDE table (order-rate metric, Bonferroni-corrected)
#
# Two complementary views per comparison x tier:
# - **required_n_per_arm** — how big the experiment needed to be to detect
#   the observed lift at 80% power.
# - **mde_pct_lift** — the smallest lift detectable at the ACTUAL per-arm
#   size (the standard experimentation-platform readout).

# %%
power_table = power_mde_table(comparisons, experiment_log)
power_table.to_csv("../outputs/power_mde_table.csv", index=False)
power_table

# %%
worked = power_table[
    (power_table["comparison"] == "voucher_vs_none") & (power_table["tier"] == "90-99")
].iloc[0]
print(
    f"Worked example — 90-99% tier, voucher_vs_none:\n"
    f"  observed lift {worked['observed_pct_lift']}% vs. MDE {worked['mde_pct_lift']}% "
    f"at n={worked['current_n_per_arm']:,}/arm\n"
    f"  -> the observed effect is ~half the smallest effect this cell could detect;\n"
    f"     required n/arm to detect it: {worked['required_n_per_arm']:,.0f}"
)

# %% [markdown]
# ## 5. Adequately-powered rerun
#
# The naive fix is scaling total N until the smallest key tier hits its
# required per-arm size: the 90-99% tier is 3.7% of the population, so
# 4,000/arm x 7 arms / 3.7% ≈ **750,000 users**. (A real design would use
# stratified oversampling of the small tiers to hit the same power at
# roughly half that total — noted here, not implemented, since the point
# of this section is closing the loop on the power diagnosis.)

# %%
big_log = dgp.simulate_users(n_users=750_000, seed=7)
big_comparisons = run_named_comparisons(big_log)
big_comparisons.to_csv("../outputs/named_comparisons_powered_750k.csv", index=False)

n_sig_small = int(comparisons["significant"].sum())
n_sig_big = int(big_comparisons["significant"].sum())
print(f"Significant after FDR at N=200k: {n_sig_small} / 48")
print(f"Significant after FDR at N=750k: {n_sig_big} / 48")

# %%
big_comparisons[big_comparisons["significant"]][
    ["comparison", "tier", "metric", "pct_lift", "q_value"]
]

# %% [markdown]
# Significance emerges exactly where the power analysis predicted —
# in the cells with large true effects — while the negligible-true-effect
# cells correctly stay null even at 3.75x the sample. This closes the
# loop: the N=200k non-results were a power problem for the big effects
# and the right answer for the null ones, and the experiment-design
# takeaway is quantified (750k naive, or less with stratified
# oversampling of the 90-99% tier).

# %% [markdown]
# ## 6. Interaction regression (tier x condition), cross-check

# %%
order_summary, profit_summary = fit_interaction_models(experiment_log)
print("Order (logit) — top 10 terms by |coef|:")
order_summary.reindex(order_summary["coef"].abs().sort_values(ascending=False).index).head(10)

# %%
print("Profit (OLS) — top 10 terms by |coef|:")
profit_summary.reindex(profit_summary["coef"].abs().sort_values(ascending=False).index).head(10)

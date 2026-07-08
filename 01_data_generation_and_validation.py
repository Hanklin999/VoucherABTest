# %% [markdown]
# # 01 — Data Generation & Validation
#
# Builds the calibrated individual-level DGP and runs the three-layer
# validation suite (aggregation recovery, sampling-noise sanity check,
# placebo check). See `data_generation.py` and `validation.py` for full
# docstrings, including the two bugs the validation process caught and
# fixed (cross-tier interpolation smearing, and the no_voucher/$0-copay
# anchor conflation).

# %%
import sys

sys.path.insert(0, "..")

import pandas as pd

from data_generation import VoucherDGP
from validation import (
    aggregation_recovery_check,
    sampling_noise_check,
    placebo_check,
    TOLERANCE_PCT,
)

# %% [markdown]
# ## Generate the individual-level experiment log

# %%
dgp = VoucherDGP(calibration_path="../data/raw_benchmarks/case_summary_tables.csv")
experiment_log = dgp.simulate_users()
experiment_log.to_csv("../data/processed/experiment_log.csv", index=False)
print(f"Simulated {len(experiment_log):,} users")
experiment_log.head()

# %% [markdown]
# ## Check 1 — Aggregation recovery vs. the 7 originally-tested conditions
#
# Simulates at large N (low noise) and compares tier x condition aggregates
# against the transcribed PDF numbers. Cells outside +/-15% are flagged.

# %%
recovery = aggregation_recovery_check(dgp)
recovery[["tier", "condition", "target_order", "order_per_user", "order_pct_error"]]

# %%
n_fail_order = (~recovery["order_within_tolerance"]).sum()
n_fail_profit = (~recovery["profit_within_tolerance"]).sum()
print(f"Order cells outside +/-{TOLERANCE_PCT}%: {n_fail_order} / {len(recovery)}")
print(f"Profit cells outside +/-{TOLERANCE_PCT}%: {n_fail_profit} / {len(recovery)}")

# %% [markdown]
# Residual misses concentrate in the 90-99% tier under stacked
# copay+count extrapolation — see DGP_ASSUMPTIONS.md for why this is a
# plausible real copay x count interaction, not a bug.

# %% [markdown]
# ## Check 2 — Sampling-noise sanity check
#
# A DGP that reproduces the target to 4 decimal places on every repeat at
# realistic N would itself be a red flag (real data is noisy).

# %%
noise = sampling_noise_check(dgp, n_users=20_000)
print(noise[["order_rate"]].describe())
print(f"Empirical SD: {noise['empirical_sd'].iloc[0]:.5f}")
print(f"Theoretical binomial SE: {noise['theoretical_binomial_se'].iloc[0]:.5f}")

# %% [markdown]
# ## Check 3 — Placebo check
#
# Shuffling condition labels should collapse the apparent treatment effect.

# %%
placebo = placebo_check(dgp)
placebo

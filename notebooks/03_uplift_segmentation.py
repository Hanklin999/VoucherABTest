# %% [markdown]
# # 03 — Uplift / CATE Segmentation
#
# Compares a RandomForest T-learner against a correctly-specified
# grouped-means baseline, to check whether `usage_rate` adds real
# heterogeneity beyond the tier label. See `uplift_model.py` for full
# docstrings and the reasoning behind this comparison.

# %%
import os
import sys
from pathlib import Path

# Anchor all relative paths to THIS file's location, not the caller's cwd —
# so the notebook works whether run via Jupyter (kernel cwd = notebooks/) or
# as a script from anywhere (e.g. `python notebooks/00_....py` from repo root).
try:
    NOTEBOOK_DIR = Path(__file__).resolve().parent
except NameError:  # __file__ undefined inside a notebook kernel
    NOTEBOOK_DIR = Path.cwd()
os.chdir(NOTEBOOK_DIR)
sys.path.insert(0, str(NOTEBOOK_DIR.parent))

import pandas as pd
from sklearn.model_selection import train_test_split

from uplift_model import (
    prepare_features,
    fit_t_learner,
    predict_cate,
    fit_tier_grouped_baseline,
    check_within_vs_between_tier_heterogeneity,
    build_targeting_priority_table,
)

experiment_log = pd.read_csv("../data/processed/experiment_log.csv")
X, treatment = prepare_features(experiment_log)

X_train, X_test, t_train, t_test, y_train, y_test, df_train, df_test = train_test_split(
    X, treatment, experiment_log["profit"].to_numpy(), experiment_log, test_size=0.3, random_state=42
)
df_test = df_test.reset_index(drop=True)

# %% [markdown]
# ## Model A: RandomForest T-learner (tier one-hot + usage_rate)

# %%
model_treated, model_control = fit_t_learner(X_train, t_train, y_train)
cate_rf = predict_cate(model_treated, model_control, X_test)
variance_rf = check_within_vs_between_tier_heterogeneity(df_test, cate_rf)
variance_rf

# %% [markdown]
# ## Model B: grouped-means baseline (tier only, correctly specified)
#
# By construction, this has exactly 0% within-tier variance — it's the
# reference point for judging whether Model A found real signal or noise.

# %%
cate_baseline = fit_tier_grouped_baseline(df_test, t_test, y_test)
variance_baseline = check_within_vs_between_tier_heterogeneity(df_test, cate_baseline)
variance_baseline

# %% [markdown]
# ## Conclusion
#
# Model A still attributes a large share of CATE variance to "within-tier"
# even after regularization — that's model noise, not discovered
# heterogeneity, given the DGP has no real within-tier signal. **Use the
# grouped-means baseline for targeting decisions** until richer
# individual-level covariates are available.

# %% [markdown]
# ## Targeting priority table (feeds 04_budget_optimization)

# %%
priority = build_targeting_priority_table(df_test, cate_baseline)
priority.to_csv("../outputs/targeting_priority.csv", index=False)
priority

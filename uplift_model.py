"""
Uplift / CATE modeling for the Voucher ROI Product Science portfolio.

Deliberately lightweight (T-learner, not Causal Forest DML) — this is a
randomized factorial experiment, so a simple two-model approach is
sufficient and more interpretable for a business audience than a heavier
causal-forest stack.

=============================================================================
WHAT THIS NOTEBOOK IS ACTUALLY TESTING
=============================================================================
The simulated DGP (src/data_generation.py) assigns every user's true
treatment effect based ONLY on their discrete `tier` — `usage_rate` is
generated as a within-tier random draw and does NOT feed the outcome model.
So the honest expectation going in is:

  - The T-learner's CATE estimates should differ strongly ACROSS tiers.
  - Within a tier, CATE estimates should NOT differ meaningfully by
    `usage_rate` — any apparent pattern there is the model fitting noise,
    not a real signal, because there IS no real signal at that level of
    granularity in this DGP.

This notebook explicitly checks for that, rather than just reporting
CATE-by-user numbers and hoping they look interesting. Finding "no real
within-tier heterogeneity" is a valid, reportable result — it means the
existing tier segmentation is already capturing what matters, which is
itself decision-relevant (no need to over-engineer a finer targeting rule).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

TIERS_ASC = ["0-29", "30-69", "70-79", "80-89", "90-99", "100"]


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Build the T-learner feature matrix and binary treatment indicator.

    Args:
        df: Individual-level experiment log (must have `tier`, `usage_rate`, `condition`).

    Returns:
        Tuple of (feature DataFrame with one-hot tier + usage_rate, treatment array).
        Treatment is 1 if ANY voucher was issued, 0 for `no_voucher` — matching
        the source case study's Key Insight #1 framing ("voucher issued" vs "no voucher").
    """
    features = pd.get_dummies(df["tier"], prefix="tier", drop_first=False)
    features["usage_rate"] = df["usage_rate"].to_numpy()
    treatment = (df["condition"] != "no_voucher").astype(int).to_numpy()
    return features, treatment


def fit_t_learner(
    X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray, seed: int = 42
) -> tuple[RandomForestRegressor, RandomForestRegressor]:
    """Fit the two T-learner arms (treated-outcome model, control-outcome model).

    Regularization note: `min_samples_leaf` is set aggressively large
    (2,000) and `max_depth` shallow (2). A first pass at this model with
    the scikit-learn defaults for `min_samples_leaf` attributed ~79% of
    total CATE variance to "within-tier" noise — implausible given the
    DGP has no real within-tier signal (see module docstring) — which
    meant the forest was splitting on `usage_rate` noise in a ~5-6k-row,
    mostly-zero profit outcome. These settings force each leaf to be large
    enough to average out that noise; see `check_within_vs_between_tier_heterogeneity`
    for the diagnostic that caught this and confirms the fix.

    Args:
        X: Feature matrix.
        treatment: Binary treatment indicator, same length as X.
        y: Outcome to model (e.g. profit or ordered).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (model fit on treated units, model fit on control units).
    """
    model_treated = RandomForestRegressor(
        n_estimators=300, max_depth=2, min_samples_leaf=2000, random_state=seed
    )
    model_control = RandomForestRegressor(
        n_estimators=300, max_depth=2, min_samples_leaf=2000, random_state=seed
    )
    model_treated.fit(X[treatment == 1], y[treatment == 1])
    model_control.fit(X[treatment == 0], y[treatment == 0])
    return model_treated, model_control


def predict_cate(
    model_treated: RandomForestRegressor, model_control: RandomForestRegressor, X: pd.DataFrame
) -> np.ndarray:
    """Predict individual-level CATE = E[Y|treated,X] - E[Y|control,X].

    Args:
        model_treated: Outcome model fit on treated units.
        model_control: Outcome model fit on control units.
        X: Feature matrix to score.

    Returns:
        Array of predicted CATE values, same length as X.
    """
    return model_treated.predict(X) - model_control.predict(X)


def check_within_vs_between_tier_heterogeneity(df: pd.DataFrame, cate: np.ndarray) -> pd.DataFrame:
    """Decompose CATE variance into between-tier and within-tier components.

    Args:
        df: Individual-level experiment log (must have `tier`).
        cate: Predicted CATE per user, same length as df.

    Returns:
        DataFrame with total, between-tier, and within-tier variance, plus
        the share of variance explained by tier — the diagnostic for
        whether usage_rate is adding real signal beyond the tier label.
    """
    work = df[["tier"]].copy()
    work["cate"] = cate
    tier_means = work.groupby("tier", observed=True)["cate"].transform("mean")

    total_var = work["cate"].var()
    between_var = tier_means.var()
    within_var = (work["cate"] - tier_means).var()

    return pd.DataFrame(
        {
            "component": ["total", "between_tier", "within_tier"],
            "variance": [total_var, between_var, within_var],
            "share_of_total": [1.0, between_var / total_var, within_var / total_var],
        }
    ).round(6)


def build_targeting_priority_table(df: pd.DataFrame, cate_profit: np.ndarray) -> pd.DataFrame:
    """Rank tiers by predicted profit uplift, weighted by population share.

    This is the direct input to the budget-optimization step (04): tells you
    which tiers give the most incremental profit per user reached, and how
    big each tier's addressable population is.

    Args:
        df: Individual-level experiment log (must have `tier`).
        cate_profit: Predicted profit CATE per user, same length as df.

    Returns:
        DataFrame sorted by mean CATE descending, with population share and
        a simple priority score (mean_cate x population_share).
    """
    work = df[["tier"]].copy()
    work["cate_profit"] = cate_profit
    summary = (
        work.groupby("tier", observed=True)
        .agg(mean_cate_profit=("cate_profit", "mean"), n_users=("cate_profit", "count"))
        .reindex(TIERS_ASC)
        .reset_index()
    )
    summary["population_share"] = summary["n_users"] / summary["n_users"].sum()
    summary["priority_score"] = summary["mean_cate_profit"] * summary["population_share"]
    return summary.sort_values("priority_score", ascending=False).round(6)


def fit_tier_grouped_baseline(df: pd.DataFrame, treatment: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute the simplest possible T-learner: difference-in-means BY TIER ONLY.

    No `usage_rate`, no flexible model — just treated-mean minus control-mean
    within each tier, broadcast back to every user in that tier. Since the
    DGP's true effect is purely a function of tier, this baseline is, by
    construction, the correctly-specified model: it has exactly 0% within-tier
    variance because it CANNOT vary within a tier. Used as the reference point
    for judging whether the more flexible RandomForest T-learner is finding
    real signal or fitting noise.

    Args:
        df: Individual-level experiment log (must have `tier`).
        treatment: Binary treatment indicator.
        y: Outcome array (e.g. profit).

    Returns:
        Array of CATE values per user (identical for all users sharing a tier).
    """
    work = df[["tier"]].copy()
    work["treatment"] = treatment
    work["y"] = y
    tier_effect = work.groupby("tier", observed=True).apply(
        lambda g: g.loc[g["treatment"] == 1, "y"].mean() - g.loc[g["treatment"] == 0, "y"].mean(),
        include_groups=False,
    )
    return work["tier"].map(tier_effect).to_numpy()



def main() -> None:
    df = pd.read_csv("data/processed/experiment_log.csv")
    X, treatment = prepare_features(df)

    X_train, X_test, t_train, t_test, y_train, y_test, df_train, df_test = train_test_split(
        X, treatment, df["profit"].to_numpy(), df, test_size=0.3, random_state=42
    )
    df_test = df_test.reset_index(drop=True)

    print("=" * 70)
    print("MODEL A: RANDOM FOREST T-LEARNER (tier one-hot + usage_rate)")
    print("=" * 70)
    model_treated, model_control = fit_t_learner(X_train, t_train, y_train)
    cate_rf = predict_cate(model_treated, model_control, X_test)
    variance_rf = check_within_vs_between_tier_heterogeneity(df_test, cate_rf)
    print(variance_rf)
    within_share_rf = variance_rf.loc[variance_rf["component"] == "within_tier", "share_of_total"].iloc[0]

    print("\n" + "=" * 70)
    print("MODEL B: GROUPED-MEANS BASELINE (tier only, correctly specified)")
    print("=" * 70)
    cate_baseline = fit_tier_grouped_baseline(df_test, t_test, y_test)
    variance_baseline = check_within_vs_between_tier_heterogeneity(df_test, cate_baseline)
    print(variance_baseline)

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(
        f"RF T-learner still attributes {within_share_rf:.1%} of CATE variance to 'within-tier' "
        "even after regularization (max_depth=2, min_samples_leaf=2000), vs. 0% for the "
        "correctly-specified grouped-means baseline (by construction). Given the DGP has no real "
        "within-tier signal, this gap is model noise, not discovered heterogeneity. "
        "RECOMMENDATION: use the grouped-means baseline for targeting decisions in this dataset — "
        "a flexible ML uplift model isn't earning its complexity here. It would be worth revisiting "
        "if richer individual-level covariates (e.g. device, channel, browsing history) become "
        "available, since those could carry real within-tier signal that a grouped mean can't capture."
    )

    print("\n" + "=" * 70)
    print("TARGETING PRIORITY TABLE (grouped-means baseline, feeds 04_budget_optimization)")
    print("=" * 70)
    priority = build_targeting_priority_table(df_test, cate_baseline)
    print(priority.to_string(index=False))
    priority.to_csv("outputs/targeting_priority.csv", index=False)


if __name__ == "__main__":
    main()

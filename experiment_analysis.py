"""
Experiment analysis for the Voucher ROI Product Science portfolio.

Two complementary approaches, both standard Product Data Scientist tooling
(deliberately NOT causal-inference-heavy — this experiment is a clean
randomized factorial design, so no DiD/PSM is needed):

1. NAMED PAIRWISE COMPARISONS (mirrors the source case study's 4 "Key Insights" exactly):
   two-proportion z-test for order rate, Welch's t-test for profit per user,
   run separately within each usage tier, then FDR-corrected across all
   tests in the same metric family (24 tests each for order/profit) so we
   don't chase noise across 6 tiers x 4 comparisons.

2. INTERACTION REGRESSION: `outcome ~ C(tier) * C(condition)`, run as OLS
   for profit and logistic regression for order. This recovers ALL
   tier x condition cells at once with proper standard errors, instead of
   one comparison at a time — useful for building the tier x lever heatmap
   in the README, and for sanity-checking the pairwise tests against a
   second method.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

TIERS_ASC = ["0-29", "30-69", "70-79", "80-89", "90-99", "100"]

# Named comparisons mirroring the source case study's 4 Key Insights.
# Each tuple: (comparison_name, condition_a, condition_b, description)
NAMED_COMPARISONS: list[tuple[str, str, str, str]] = [
    ("voucher_vs_none", "s0_m249_c3", "no_voucher", "Any voucher vs. no voucher issued"),
    ("minspend_299_vs_249", "s0_m299_c3", "s0_m249_c3", "Raise min-spend $249 -> $299 (same copay/count)"),
    ("copay_9_vs_0", "s9_m249_c3", "s0_m249_c3", "Raise buyer copay $0 -> $9 (same min-spend/count)"),
    ("count_3_vs_1", "s19_m199_c3", "s19_m199_c1", "Issue 3 vouchers instead of 1 (same copay/min-spend)"),
]


def _proportion_test(df: pd.DataFrame, tier: str, cond_a: str, cond_b: str) -> tuple[float, float, float, float]:
    """Two-proportion z-test for order rate between two conditions within a tier.

    Args:
        df: Individual-level experiment log (must have `tier`, `condition`, `ordered`).
        tier: Tier label to filter on.
        cond_a: First condition key.
        cond_b: Second condition key (the comparison baseline).

    Returns:
        Tuple of (rate_a, rate_b, z_stat, p_value).
    """
    a = df[(df["tier"] == tier) & (df["condition"] == cond_a)]["ordered"]
    b = df[(df["tier"] == tier) & (df["condition"] == cond_b)]["ordered"]
    counts = np.array([a.sum(), b.sum()])
    nobs = np.array([len(a), len(b)])
    z_stat, p_value = proportions_ztest(counts, nobs)
    return a.mean(), b.mean(), z_stat, p_value


def _profit_ttest(df: pd.DataFrame, tier: str, cond_a: str, cond_b: str) -> tuple[float, float, float, float]:
    """Welch's t-test for profit per user between two conditions within a tier.

    Args:
        df: Individual-level experiment log (must have `tier`, `condition`, `profit`).
        tier: Tier label to filter on.
        cond_a: First condition key.
        cond_b: Second condition key (the comparison baseline).

    Returns:
        Tuple of (mean_a, mean_b, t_stat, p_value).
    """
    a = df[(df["tier"] == tier) & (df["condition"] == cond_a)]["profit"]
    b = df[(df["tier"] == tier) & (df["condition"] == cond_b)]["profit"]
    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
    return a.mean(), b.mean(), t_stat, p_value


def run_named_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """Run all named comparisons x tiers x metrics, with FDR correction.

    Args:
        df: Individual-level experiment log.

    Returns:
        Tidy DataFrame with one row per (comparison, tier, metric), including
        raw p-value, BH-FDR-corrected q-value, and a significance flag at
        q < 0.05. Correction is applied SEPARATELY within each metric family
        (order, profit), matching the standard practice of not pooling
        p-values across unrelated outcome types.
    """
    rows = []
    for name, cond_a, cond_b, description in NAMED_COMPARISONS:
        for tier in TIERS_ASC:
            rate_a, rate_b, z_stat, p_order = _proportion_test(df, tier, cond_a, cond_b)
            rows.append(
                {
                    "comparison": name,
                    "description": description,
                    "tier": tier,
                    "metric": "order_rate",
                    "mean_a": rate_a,
                    "mean_b": rate_b,
                    "pct_lift": (rate_a - rate_b) / rate_b * 100 if rate_b else np.nan,
                    "stat": z_stat,
                    "p_value": p_order,
                }
            )
            profit_a, profit_b, t_stat, p_profit = _profit_ttest(df, tier, cond_a, cond_b)
            rows.append(
                {
                    "comparison": name,
                    "description": description,
                    "tier": tier,
                    "metric": "profit_per_user",
                    "mean_a": profit_a,
                    "mean_b": profit_b,
                    "pct_lift": (profit_a - profit_b) / profit_b * 100 if profit_b else np.nan,
                    "stat": t_stat,
                    "p_value": p_profit,
                }
            )

    result = pd.DataFrame(rows)
    for metric in result["metric"].unique():
        mask = result["metric"] == metric
        _, q_values, _, _ = multipletests(result.loc[mask, "p_value"], method="fdr_bh")
        result.loc[mask, "q_value"] = q_values
    result["significant"] = result["q_value"] < 0.05
    return result.round(5)


def fit_interaction_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit tier x condition interaction models for order (logit) and profit (OLS).

    Recovers every tier x condition cell simultaneously with proper standard
    errors, as a cross-check against the one-at-a-time pairwise tests above.

    Args:
        df: Individual-level experiment log.

    Returns:
        Tuple of (order_model_summary, profit_model_summary) DataFrames, each
        with coefficient, standard error, and p-value per tier x condition term.
    """
    model_df = df.copy()
    model_df["tier"] = pd.Categorical(model_df["tier"], categories=TIERS_ASC, ordered=True)
    reference_condition = "no_voucher"
    model_df["condition"] = pd.Categorical(
        model_df["condition"],
        categories=[reference_condition]
        + [c for c in model_df["condition"].unique() if c != reference_condition],
    )

    order_model = smf.logit("ordered ~ C(tier) * C(condition)", data=model_df).fit(disp=0)
    profit_model = smf.ols("profit ~ C(tier) * C(condition)", data=model_df).fit()

    def _tidy(fit_result) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "term": fit_result.params.index,
                "coef": fit_result.params.values,
                "std_err": fit_result.bse.values,
                "p_value": fit_result.pvalues.values,
            }
        ).round(5)

    return _tidy(order_model), _tidy(profit_model)


def required_sample_size(
    baseline_rate: float, pct_lift: float, n_comparisons: int, power: float = 0.8
) -> int:
    """Compute required per-arm sample size to detect a lift at target power.

    Uses a Bonferroni-adjusted alpha (0.05 / n_comparisons) to reflect the
    same multiple-comparison correction applied in `run_named_comparisons`,
    since the honest question is "how much traffic to detect THIS effect
    AFTER correcting for testing 24 things," not the uncorrected number.

    Args:
        baseline_rate: Baseline conversion rate (e.g. the control arm's order rate).
        pct_lift: Relative lift to detect, as a percent (e.g. 15.0 for +15%).
        n_comparisons: Number of comparisons the correction is spread over.
        power: Desired statistical power (default 0.80).

    Returns:
        Required sample size per arm, rounded up to the nearest 100.
    """
    from statsmodels.stats.power import NormalIndPower
    from statsmodels.stats.proportion import proportion_effectsize

    treatment_rate = baseline_rate * (1 + pct_lift / 100)
    effect_size = proportion_effectsize(treatment_rate, baseline_rate)
    alpha_corrected = 0.05 / n_comparisons

    analysis = NormalIndPower()
    n_required = analysis.solve_power(
        effect_size=effect_size, alpha=alpha_corrected, power=power, ratio=1.0
    )
    return int(np.ceil(n_required / 100.0) * 100)



def main() -> None:
    df = pd.read_csv("data/processed/experiment_log.csv")

    print("=" * 70)
    print("NAMED COMPARISONS (mirrors source case study's 4 Key Insights), FDR-corrected")
    print("=" * 70)
    comparisons = run_named_comparisons(df)
    print(
        comparisons[
            ["comparison", "tier", "metric", "pct_lift", "p_value", "q_value", "significant"]
        ].to_string(index=False)
    )
    comparisons.to_csv("outputs/named_comparisons.csv", index=False)

    print("\n" + "=" * 70)
    print("INTERACTION REGRESSION (tier x condition, cross-check)")
    print("=" * 70)
    order_summary, profit_summary = fit_interaction_models(df)
    print("\nOrder (logit) — top 10 terms by |coef|:")
    print(order_summary.reindex(order_summary["coef"].abs().sort_values(ascending=False).index).head(10))
    print("\nProfit (OLS) — top 10 terms by |coef|:")
    print(profit_summary.reindex(profit_summary["coef"].abs().sort_values(ascending=False).index).head(10))

    print("\n" + "=" * 70)
    print("SAMPLE SIZE NEEDED (worked example: 90-99 tier, voucher_vs_none)")
    print("=" * 70)
    n_comparisons = len(NAMED_COMPARISONS) * len(TIERS_ASC)  # 24, matches order-metric family
    example = comparisons[
        (comparisons["comparison"] == "voucher_vs_none")
        & (comparisons["tier"] == "90-99")
        & (comparisons["metric"] == "order_rate")
    ].iloc[0]
    n_needed = required_sample_size(
        baseline_rate=example["mean_b"], pct_lift=example["pct_lift"], n_comparisons=n_comparisons
    )
    current_n = df[(df["tier"] == "90-99") & (df["condition"] == "no_voucher")].shape[0]
    print(f"Observed lift: {example['pct_lift']:.1f}% on a {example['mean_b']:.3f} baseline order rate")
    print(
        f"Required N per arm (80% power, Bonferroni-corrected alpha for {n_comparisons} tests): {n_needed:,}"
    )
    print(
        f"Current simulated N in this cell: {current_n:,} -> "
        f"{'UNDERPOWERED' if current_n < n_needed else 'adequately powered'}"
    )


if __name__ == "__main__":
    main()

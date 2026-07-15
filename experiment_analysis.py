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


def _proportion_test(
    df: pd.DataFrame, tier: str, cond_a: str, cond_b: str
) -> tuple[float, float, float, float, float, float]:
    """Two-proportion z-test + Newcombe CI for order rate between two conditions.

    Args:
        df: Individual-level experiment log (must have `tier`, `condition`, `ordered`).
        tier: Tier label to filter on.
        cond_a: First condition key.
        cond_b: Second condition key (the comparison baseline).

    Returns:
        Tuple of (rate_a, rate_b, z_stat, p_value, ci_low, ci_high), where the
        CI is a 95% Newcombe score interval on the absolute rate difference
        (rate_a - rate_b).
    """
    from statsmodels.stats.proportion import confint_proportions_2indep

    a = df[(df["tier"] == tier) & (df["condition"] == cond_a)]["ordered"]
    b = df[(df["tier"] == tier) & (df["condition"] == cond_b)]["ordered"]
    counts = np.array([a.sum(), b.sum()])
    nobs = np.array([len(a), len(b)])
    z_stat, p_value = proportions_ztest(counts, nobs)
    ci_low, ci_high = confint_proportions_2indep(
        a.sum(), len(a), b.sum(), len(b), compare="diff", method="newcomb"
    )
    return a.mean(), b.mean(), z_stat, p_value, ci_low, ci_high


def _profit_ttest(
    df: pd.DataFrame, tier: str, cond_a: str, cond_b: str
) -> tuple[float, float, float, float, float, float]:
    """Welch's t-test + 95% CI for profit per user between two conditions.

    Args:
        df: Individual-level experiment log (must have `tier`, `condition`, `profit`).
        tier: Tier label to filter on.
        cond_a: First condition key.
        cond_b: Second condition key (the comparison baseline).

    Returns:
        Tuple of (mean_a, mean_b, t_stat, p_value, ci_low, ci_high), where the
        CI uses the Welch-Satterthwaite degrees of freedom.
    """
    a = df[(df["tier"] == tier) & (df["condition"] == cond_a)]["profit"]
    b = df[(df["tier"] == tier) & (df["condition"] == cond_b)]["profit"]
    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)

    var_a, var_b = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(var_a + var_b)
    dof = (var_a + var_b) ** 2 / (var_a**2 / (len(a) - 1) + var_b**2 / (len(b) - 1))
    t_crit = stats.t.ppf(0.975, dof)
    diff = a.mean() - b.mean()
    return a.mean(), b.mean(), t_stat, p_value, diff - t_crit * se, diff + t_crit * se


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
            rate_a, rate_b, z_stat, p_order, ci_lo_o, ci_hi_o = _proportion_test(
                df, tier, cond_a, cond_b
            )
            rows.append(
                {
                    "comparison": name,
                    "description": description,
                    "tier": tier,
                    "metric": "order_rate",
                    "mean_a": rate_a,
                    "mean_b": rate_b,
                    "abs_diff": rate_a - rate_b,
                    "ci_low": ci_lo_o,
                    "ci_high": ci_hi_o,
                    "pct_lift": (rate_a - rate_b) / rate_b * 100 if rate_b else np.nan,
                    "stat": z_stat,
                    "p_value": p_order,
                }
            )
            profit_a, profit_b, t_stat, p_profit, ci_lo_p, ci_hi_p = _profit_ttest(
                df, tier, cond_a, cond_b
            )
            rows.append(
                {
                    "comparison": name,
                    "description": description,
                    "tier": tier,
                    "metric": "profit_per_user",
                    "mean_a": profit_a,
                    "mean_b": profit_b,
                    "abs_diff": profit_a - profit_b,
                    "ci_low": ci_lo_p,
                    "ci_high": ci_hi_p,
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


# Condition-name -> (copay, min_spend, count); mirrors data_generation's default set
CONDITION_PARAMS: dict[str, tuple[float, float, int]] = {
    "no_voucher": (0, 0, 0),
    "s0_m199_c1": (0, 199, 1),
    "s0_m249_c3": (0, 249, 3),
    "s0_m299_c3": (0, 299, 3),
    "s9_m249_c3": (9, 249, 3),
    "s19_m199_c1": (19, 199, 1),
    "s19_m199_c3": (19, 199, 3),
}
COMPARISON_MAP = {name: (a, b) for name, a, b, _ in NAMED_COMPARISONS}


def primary_pooled_estimate(df: pd.DataFrame, outcome: str = "ordered") -> dict:
    """Population-weighted primary readout: any voucher vs. control, one number.

    The headline estimate a decision memo leads with: the effect of issuing
    ANY voucher (all 6 treatment arms pooled) vs. control on the primary
    metric, estimated within each tier and combined with tier-population
    weights — a stratified difference-in-means, which is unbiased here
    because randomization is within-tier by design. Pooling arms answers
    the program-level question ("does the voucher program move orders?");
    the per-cell named comparisons answer the design question ("which
    configuration, for whom?").

    Args:
        df: Individual-level experiment log.
        outcome: Binary outcome column to estimate on (default: `ordered`).

    Returns:
        Dict with the weighted absolute difference (percentage points),
        relative lift vs. the weighted control mean, 95% CI, z-statistic,
        and p-value.
    """
    est, var, ctrl_mean_weighted = 0.0, 0.0, 0.0
    total_n = len(df)
    for tier in TIERS_ASC:
        tier_df = df[df["tier"] == tier]
        weight = len(tier_df) / total_n
        treat = tier_df[tier_df["condition"] != "no_voucher"][outcome]
        ctrl = tier_df[tier_df["condition"] == "no_voucher"][outcome]
        diff = treat.mean() - ctrl.mean()
        cell_var = treat.var(ddof=1) / len(treat) + ctrl.var(ddof=1) / len(ctrl)
        est += weight * diff
        var += weight**2 * cell_var
        ctrl_mean_weighted += weight * ctrl.mean()

    se = np.sqrt(var)
    z = est / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return {
        "abs_diff_pp": est * 100,
        "relative_lift_pct": est / ctrl_mean_weighted * 100,
        "ci_low_pp": (est - 1.96 * se) * 100,
        "ci_high_pp": (est + 1.96 * se) * 100,
        "z": z,
        "p_value": p_value,
        "control_mean": ctrl_mean_weighted,
    }


def srm_check(df: pd.DataFrame, alarm_threshold: float = 0.001) -> pd.DataFrame:
    """Sample Ratio Mismatch check: did randomization deliver the designed split?

    Chi-square goodness-of-fit per tier against a uniform 1/7 allocation.
    This is the standard trust check experimentation platforms run BEFORE
    reading any results — a broken assignment mechanism invalidates every
    downstream number regardless of how significant it looks. The alarm
    threshold is deliberately strict (p < 0.001, industry convention) because
    at large N even tiny imbalances become "significant" at 0.05 without
    indicating a real assignment bug.

    Args:
        df: Individual-level experiment log (must have `tier`, `condition`).
        alarm_threshold: p-value below which to flag an SRM alarm.

    Returns:
        DataFrame with one row per tier: user count, chi-square statistic,
        p-value, and SRM alarm flag.
    """
    rows = []
    for tier in TIERS_ASC:
        counts = df[df["tier"] == tier]["condition"].value_counts().to_numpy()
        expected = np.full(len(counts), counts.sum() / len(counts))
        chi2, p_value = stats.chisquare(counts, expected)
        rows.append(
            {
                "tier": tier,
                "n_users": int(counts.sum()),
                "n_conditions": len(counts),
                "chi2": chi2,
                "p_value": p_value,
                "srm_alarm": p_value < alarm_threshold,
            }
        )
    return pd.DataFrame(rows).round(5)


def power_mde_table(
    comparisons: pd.DataFrame, df: pd.DataFrame, power: float = 0.8
) -> pd.DataFrame:
    """Per-comparison power requirements and minimum detectable effect (MDE).

    For every named comparison x tier (order-rate metric), reports two
    complementary numbers, both at a Bonferroni-corrected alpha:
    - required_n_per_arm: sample size needed to detect the OBSERVED lift at
      the target power (the "how big should this experiment have been" view).
    - mde_pct_lift: the smallest relative lift detectable at the CURRENT
      per-arm sample size (the "what could this experiment ever have told
      us" view — the standard readout format on experimentation platforms).

    A cell is a genuine power problem only when the observed/true effect is
    smaller than the MDE but real; cells whose true effect is ~0 are not
    "underpowered", they are correctly null (see `direction_concordance`).

    Args:
        comparisons: Output of `run_named_comparisons`.
        df: Individual-level experiment log (for current per-arm counts).
        power: Target statistical power.

    Returns:
        DataFrame with one row per comparison x tier: observed lift, current
        n/arm, required n/arm, MDE, and a status label.
    """
    from statsmodels.stats.power import NormalIndPower
    from statsmodels.stats.proportion import proportion_effectsize

    n_comparisons = len(NAMED_COMPARISONS) * len(TIERS_ASC)
    alpha = 0.05 / n_comparisons
    analysis = NormalIndPower()

    rows = []
    order_rows = comparisons[comparisons["metric"] == "order_rate"]
    for _, r in order_rows.iterrows():
        cond_a, cond_b = COMPARISON_MAP[r["comparison"]]
        n_a = ((df["tier"] == r["tier"]) & (df["condition"] == cond_a)).sum()
        n_b = ((df["tier"] == r["tier"]) & (df["condition"] == cond_b)).sum()
        n_current = int(min(n_a, n_b))

        # Required N to detect the observed lift
        p_base = r["mean_b"]
        p_treat = p_base * (1 + r["pct_lift"] / 100)
        effect = abs(proportion_effectsize(p_treat, p_base))
        if effect < 1e-6:
            required_n = np.nan
        else:
            required_n = float(
                analysis.solve_power(effect_size=effect, alpha=alpha, power=power, ratio=1.0)
            )
            required_n = np.ceil(required_n / 100.0) * 100

        # MDE at the current per-arm N (solve for effect size, invert to a lift)
        h_mde = float(
            analysis.solve_power(effect_size=None, nobs1=n_current, alpha=alpha, power=power)
        )
        p_mde = np.sin(np.clip(np.arcsin(np.sqrt(p_base)) + h_mde / 2, 0, np.pi / 2)) ** 2
        mde_pct_lift = (p_mde - p_base) / p_base * 100 if p_base > 0 else np.nan

        if np.isnan(required_n):
            status = "no observed effect"
        elif n_current >= required_n:
            status = "adequately powered"
        else:
            status = "underpowered"

        rows.append(
            {
                "comparison": r["comparison"],
                "tier": r["tier"],
                "observed_pct_lift": round(r["pct_lift"], 1),
                "current_n_per_arm": n_current,
                "required_n_per_arm": required_n,
                "mde_pct_lift": round(mde_pct_lift, 1),
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def direction_concordance(comparisons: pd.DataFrame, dgp) -> pd.DataFrame:
    """Check whether observed effect signs agree with the DGP's ground truth.

    This is a diagnostic only a simulation can run: because the true effects
    were calibrated in, we can classify every comparison as (a) sign-agrees,
    (b) sign-flipped by sampling noise, or (c) negligible true effect (where
    "agreement" is meaningless and non-significance is the CORRECT outcome,
    not a power failure). It turns the claim "estimates trend in the right
    direction" from an assertion into a measured rate, and separates the
    genuinely underpowered cells from the correctly-null ones.

    Args:
        comparisons: Output of `run_named_comparisons`.
        dgp: A fitted VoucherDGP instance (for ground-truth expected values).

    Returns:
        DataFrame with one row per comparison x tier x metric: true diff,
        observed diff, truth classification, and sign agreement.
    """
    negligible_eps = 5e-4

    def expected_value(condition: str, tier_idx: int, metric: str) -> float:
        copay, min_spend, count = CONDITION_PARAMS[condition]
        if condition == "no_voucher":
            order = dgp.base_order[tier_idx]
            ppo = dgp.base_profit_per_order[tier_idx]
        else:
            order, ppo = dgp.predict_segment(
                np.array([tier_idx]), np.array([float(copay)]),
                np.array([float(min_spend)]), np.array([count]),
            )
            order, ppo = float(order[0]), float(ppo[0])
        return order if metric == "order_rate" else order * ppo

    rows = []
    for _, r in comparisons.iterrows():
        cond_a, cond_b = COMPARISON_MAP[r["comparison"]]
        tier_idx = TIERS_ASC.index(r["tier"])
        true_diff = expected_value(cond_a, tier_idx, r["metric"]) - expected_value(
            cond_b, tier_idx, r["metric"]
        )
        observed_diff = r["mean_a"] - r["mean_b"]
        negligible = abs(true_diff) < negligible_eps
        rows.append(
            {
                "comparison": r["comparison"],
                "tier": r["tier"],
                "metric": r["metric"],
                "true_diff": round(true_diff, 5),
                "observed_diff": round(observed_diff, 5),
                "true_effect": "negligible" if negligible else ("positive" if true_diff > 0 else "negative"),
                "sign_agrees": bool(np.sign(true_diff) == np.sign(observed_diff)) if not negligible else None,
            }
        )
    return pd.DataFrame(rows)



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

"""
Individual-level data generating process for the Voucher ROI Product
Science portfolio, calibrated against data/raw_benchmarks/case_summary_tables.csv
(the tidy transcription of the original PDF case study).

=============================================================================
MODELING STRATEGY
=============================================================================
The PDF only reports SEGMENT-LEVEL aggregates (order/user, profit/user by
usage tier x test condition). There is no individual-level data. So the
approach here is:

1. BASELINE SURFACE (no_voucher condition): looked up EXACTLY per discrete
   tier (no cross-tier interpolation). An earlier version of this DGP
   interpolated continuously over a latent `usage_rate` in [0, 1], but the
   validation step caught that tier "100" is NOT the top of a smooth
   monotonic curve — it has a much LOWER order rate than tier "90-99"
   (likely because "100%-free-shipping" buyers are dominated by one-time
   purchasers who always used free shipping simply because they only
   ordered once). Smoothing across that discontinuity dragged the "90-99"
   tier's predicted order rate down by 20-38%. The fix: treat tier as a
   discrete categorical lookup key, matching the actual resolution of the
   source data. `usage_rate` is still simulated (uniform within each
   tier's range) and kept as a covariate for future segmentation-refinement
   work, but it does NOT feed the outcome model in this version.

2. TREATMENT-EFFECT LEVERS (copay / min-spend / voucher-count): each
   lever's effect is estimated from ONLY the specific pair of conditions in
   the PDF that isolates it, PER discrete tier, then assumed to scale
   LINEARLY outside the observed copay/min-spend range (this extrapolation
   axis is continuous-dollar, not cross-tier, so it doesn't hit the same
   discontinuity problem). Combinations outside the 7 originally-tested
   conditions are model predictions, not reproductions — validation.py
   labels them accordingly.

3. Effects are combined ADDITIVELY within each tier (no lever x lever
   interaction terms). This is a simplifying assumption forced by data
   sparsity: with only 7 distinct conditions we cannot identify 3-way
   interactions. Stated explicitly here and in DGP_ASSUMPTIONS.md.

Definitions:
- `copay_usd`: amount the BUYER pays out of pocket for shipping (this is
  the PDF's "buyer paid $X"). LOWER copay = MORE generous voucher. This is
  the inverse of "platform subsidy amount" — named this way to avoid sign
  confusion, since higher copay empirically REDUCES order rate but
  INCREASES profit-per-order (platform keeps more margin).
=============================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_USERS = 200_000

TIERS_ASC = ["0-29", "30-69", "70-79", "80-89", "90-99", "100"]
TIER_MIDPOINTS_ASC = np.array([0.145, 0.495, 0.745, 0.845, 0.945, 1.000])
TIER_RANGES = {
    "0-29": (0.00, 0.29),
    "30-69": (0.30, 0.69),
    "70-79": (0.70, 0.79),
    "80-89": (0.80, 0.89),
    "90-99": (0.90, 0.99),
    "100": (1.00, 1.00),
}

CALIBRATION_PATH = "data/raw_benchmarks/case_summary_tables.csv"


def _load_condition_vector(cal: pd.DataFrame, condition: str, metric: str) -> np.ndarray:
    """Pull a 6-tier vector (ascending usage order) for one condition/metric.

    Args:
        cal: The calibration-targets DataFrame.
        condition: Condition key (e.g. "no_voucher", "s0_m249_c3").
        metric: "order_per_user" or "profit_per_user".

    Returns:
        Length-6 array ordered to match TIERS_ASC (low usage -> high usage).
    """
    sub = cal[(cal["condition"] == condition) & (cal["metric"] == metric)]
    sub = sub.set_index("tier").reindex(TIERS_ASC)
    return sub["value"].to_numpy(dtype=float)


class VoucherDGP:
    """Calibrated data generating process for the voucher ROI experiment.

    Fits interpolators for the baseline surface and each treatment-effect
    lever from `case_summary_tables.csv`, then exposes `predict_segment`
    (segment-level expected values, for validation) and `simulate_users`
    (individual-level draws, for downstream test/uplift/optimization work).
    """

    def __init__(self, calibration_path: str = CALIBRATION_PATH) -> None:
        cal = pd.read_csv(calibration_path)
        self.calibration = cal

        # --- Baseline surface (no_voucher) — a QUALITATIVELY DIFFERENT regime
        # from "voucher issued with $0 copay". Used ONLY for has_voucher=False. ---
        self.base_order = _load_condition_vector(cal, "no_voucher", "order_per_user")
        base_profit_user = _load_condition_vector(cal, "no_voucher", "profit_per_user")
        self.base_profit_per_order = base_profit_user / self.base_order

        # --- Voucher-issued intercept: anchored at the DIRECTLY OBSERVED
        # (copay=0, min_spend=249, count=3) cell. All lever effects below are
        # expressed as offsets from THIS anchor, not from no_voucher — issuing
        # a voucher at all is its own level shift, separate from how generous
        # its terms are. ---
        order_0_249_3 = _load_condition_vector(cal, "s0_m249_c3", "order_per_user")
        self.voucher_intercept_order = order_0_249_3
        self.voucher_intercept_ppo = _load_condition_vector(cal, "s0_m249_c3", "profit_per_user") / order_0_249_3

        # --- Copay lever: averaged per-dollar slope from two known deltas ---
        # Delta at $9 (min_spend=249, count=3) and delta at $19 (min_spend=199, count=1).
        order_9_249_3 = _load_condition_vector(cal, "s9_m249_c3", "order_per_user")
        order_0_199_1 = _load_condition_vector(cal, "s0_m199_c1", "order_per_user")
        order_19_199_1 = _load_condition_vector(cal, "s19_m199_c1", "order_per_user")

        ppo_0_249_3 = self.voucher_intercept_ppo
        ppo_9_249_3 = _load_condition_vector(cal, "s9_m249_c3", "profit_per_user") / order_9_249_3
        ppo_0_199_1 = _load_condition_vector(cal, "s0_m199_c1", "profit_per_user") / order_0_199_1
        ppo_19_199_1 = _load_condition_vector(cal, "s19_m199_c1", "profit_per_user") / order_19_199_1

        slope_order_at_9 = (order_9_249_3 - order_0_249_3) / 9.0
        slope_order_at_19 = (order_19_199_1 - order_0_199_1) / 19.0
        self.slope_order_copay = (slope_order_at_9 + slope_order_at_19) / 2.0

        slope_ppo_at_9 = (ppo_9_249_3 - ppo_0_249_3) / 9.0
        slope_ppo_at_19 = (ppo_19_199_1 - ppo_0_199_1) / 19.0
        self.slope_ppo_copay = (slope_ppo_at_9 + slope_ppo_at_19) / 2.0

        # --- Min-spend lever: slope from the one known $249 -> $299 delta ---
        order_299 = _load_condition_vector(cal, "s0_m299_c3", "order_per_user")
        ppo_299 = _load_condition_vector(cal, "s0_m299_c3", "profit_per_user") / order_299
        self.slope_order_minspend = (order_299 - order_0_249_3) / 50.0
        self.slope_ppo_minspend = (ppo_299 - ppo_0_249_3) / 50.0

        # --- Voucher-count lever: fixed shift, expressed relative to count=3
        # (the anchor's own count level), i.e. k(count=3)=0, k(count=1)=-delta. ---
        order_19_199_3 = _load_condition_vector(cal, "s19_m199_c3", "order_per_user")
        ppo_19_199_3 = _load_condition_vector(cal, "s19_m199_c3", "profit_per_user") / order_19_199_3
        self.delta_order_count3_vs_1 = order_19_199_3 - order_19_199_1
        self.delta_ppo_count3_vs_1 = ppo_19_199_3 - ppo_19_199_1

        # Reference levels the slopes/deltas are anchored to
        self._ref_copay = 0.0
        self._ref_minspend = 249.0

    # ------------------------------------------------------------------
    def _lookup(self, values_asc: np.ndarray, tier_idx: np.ndarray) -> np.ndarray:
        """Exact per-tier lookup (no cross-tier interpolation)."""
        return values_asc[tier_idx]

    def predict_segment(
        self, tier_idx: np.ndarray, copay: np.ndarray, min_spend: np.ndarray, count: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict expected order_prob and profit_per_order for given inputs.

        Args:
            tier_idx: Integer tier index into TIERS_ASC (0=lowest usage tier).
            copay: Buyer out-of-pocket shipping copay in USD (0 if no voucher).
            min_spend: Minimum spend threshold in USD (0 used as "no voucher" sentinel).
            count: Number of vouchers issued (1 or 3; ignored if no voucher).

        Returns:
            Tuple of (order_prob, profit_per_order) arrays, same shape as inputs.
        """
        base_order = self._lookup(self.voucher_intercept_order, tier_idx)
        base_ppo = self._lookup(self.voucher_intercept_ppo, tier_idx)
        no_voucher_order = self._lookup(self.base_order, tier_idx)
        no_voucher_ppo = self._lookup(self.base_profit_per_order, tier_idx)

        slope_o_copay = self._lookup(self.slope_order_copay, tier_idx)
        slope_p_copay = self._lookup(self.slope_ppo_copay, tier_idx)
        slope_o_ms = self._lookup(self.slope_order_minspend, tier_idx)
        slope_p_ms = self._lookup(self.slope_ppo_minspend, tier_idx)
        delta_o_count = self._lookup(self.delta_order_count3_vs_1, tier_idx)
        delta_p_count = self._lookup(self.delta_ppo_count3_vs_1, tier_idx)

        has_voucher = min_spend > 0  # min_spend == 0 used as "no voucher" sentinel
        # count reference is 3 (the anchor cell's own count level); count=1 is
        # the offset -delta, not the other way around.
        count_term_order = np.where(count == 3, 0.0, -delta_o_count)
        count_term_ppo = np.where(count == 3, 0.0, -delta_p_count)

        order_prob = np.where(
            has_voucher,
            base_order
            + slope_o_copay * (copay - self._ref_copay)
            + slope_o_ms * (min_spend - self._ref_minspend)
            + count_term_order,
            no_voucher_order,
        )
        profit_per_order = np.where(
            has_voucher,
            base_ppo
            + slope_p_copay * (copay - self._ref_copay)
            + slope_p_ms * (min_spend - self._ref_minspend)
            + count_term_ppo,
            no_voucher_ppo,
        )
        return np.clip(order_prob, 0.001, 0.5), np.clip(profit_per_order, 0.001, 5.0)

    # ------------------------------------------------------------------
    def simulate_users(
        self,
        n_users: int = N_USERS,
        conditions: list[tuple[float, float, int]] | None = None,
        seed: int = RANDOM_SEED,
    ) -> pd.DataFrame:
        """Simulate individual-level users with a stratified factorial assignment.

        Every user is first assigned a tier (matching observed population
        shares) and a continuous usage_rate within that tier's range, then
        RANDOMLY assigned to one of the supplied test conditions, stratified
        WITHIN tier (mirrors the case study's within-segment test design).

        Args:
            n_users: Total number of users to simulate.
            conditions: List of (copay, min_spend, count) tuples to test.
                Use min_spend=0 to represent "no voucher issued".
                Defaults to the 7 originally-tested conditions.
            seed: Random seed for reproducibility.

        Returns:
            DataFrame with one row per user: tier, usage_rate, assigned
            condition, and simulated order (binary) + profit (float, 0 if
            no order).
        """
        rng = np.random.default_rng(seed)
        if conditions is None:
            conditions = [
                (0, 0, 0),      # no_voucher (sentinel: min_spend=0)
                (0, 199, 1),
                (0, 249, 3),
                (0, 299, 3),
                (9, 249, 3),
                (19, 199, 1),
                (19, 199, 3),
            ]

        cal = self.calibration
        pop_share = cal[cal["metric"] == "population_share_pct"].set_index("tier")
        pop_share = pop_share.reindex(TIERS_ASC)["value"].to_numpy(dtype=float)
        pop_share = pop_share / pop_share.sum()  # renormalize to sum to 1

        tier_idx = rng.choice(len(TIERS_ASC), size=n_users, p=pop_share)
        tier = np.array(TIERS_ASC)[tier_idx]

        usage_rate = np.empty(n_users)
        for i, t in enumerate(TIERS_ASC):
            mask = tier_idx == i
            lo, hi = TIER_RANGES[t]
            usage_rate[mask] = rng.uniform(lo, hi, size=mask.sum()) if hi > lo else lo

        cond_idx = rng.integers(0, len(conditions), size=n_users)
        copay = np.array([conditions[i][0] for i in cond_idx], dtype=float)
        min_spend = np.array([conditions[i][1] for i in cond_idx], dtype=float)
        count = np.array([conditions[i][2] for i in cond_idx], dtype=int)

        order_prob, profit_per_order = self.predict_segment(tier_idx, copay, min_spend, count)

        ordered = rng.binomial(1, order_prob)
        # Lognormal noise around the calibrated conditional mean so individual
        # profit isn't a deterministic function of tier + condition alone.
        profit = np.where(
            ordered == 1,
            profit_per_order * rng.lognormal(mean=0, sigma=0.35, size=n_users),
            0.0,
        )

        condition_label = np.array(
            [
                f"s{int(c)}_m{int(m)}_c{k}" if m > 0 else "no_voucher"
                for c, m, k in zip(copay, min_spend, count)
            ]
        )

        return pd.DataFrame(
            {
                "user_id": np.arange(n_users),
                "tier": tier,
                "usage_rate": usage_rate.round(4),
                "condition": condition_label,
                "copay_usd": copay,
                "min_spend_usd": min_spend,
                "voucher_count": count,
                "ordered": ordered,
                "profit": profit.round(4),
            }
        )


def main() -> None:
    """Generate the individual-level simulated experiment log and save it."""
    import os

    dgp = VoucherDGP()
    df = dgp.simulate_users()

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/experiment_log.csv", index=False)
    print(f"Simulated {len(df):,} users -> data/processed/experiment_log.csv")
    print(df.groupby(["tier", "condition"], observed=True)[["ordered", "profit"]].mean().head(15))


if __name__ == "__main__":
    main()

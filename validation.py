"""
Validation: reproduce Locker-project-style "does the DGP match reality"
rigor for the Voucher ROI project.

Three checks:
1. Aggregation recovery — simulate at large N, aggregate back to
   tier x condition, and compare against the transcribed source-table numbers for
   the 7 conditions that were ACTUALLY in the original case study. Flags
   any cell outside a tolerance band.
2. Sampling-noise sanity check — repeat the simulation at a REALISTIC
   sample size (not 200k) and show that cell-level estimates jitter by an
   amount consistent with binomial/lognormal sampling error, i.e. the DGP
   isn't suspiciously exact.
3. Placebo check — shuffle condition labels and confirm the "effect"
   disappears, as a safeguard against accidental data leakage in the DGP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_generation import VoucherDGP

TOLERANCE_PCT = 15.0  # cells within +/-15% of target are "recovered"


def aggregation_recovery_check(dgp: VoucherDGP, n_users: int = 500_000) -> pd.DataFrame:
    """Compare large-N simulated aggregates against the transcribed source-table targets.

    Args:
        dgp: A fitted VoucherDGP instance.
        n_users: Number of users to simulate for a low-noise comparison.

    Returns:
        DataFrame with target vs. simulated values and %-error per cell,
        for the 7 originally-tested conditions only (everything else is
        extrapolation and has no ground truth to check against).
    """
    sim = dgp.simulate_users(n_users=n_users)
    agg = (
        sim.groupby(["tier", "condition"], observed=True)
        .agg(order_per_user=("ordered", "mean"), profit_per_user=("profit", "mean"))
        .reset_index()
    )

    targets = dgp.calibration[
        dgp.calibration["metric"].isin(["order_per_user", "profit_per_user"])
    ].pivot_table(index=["tier", "condition"], columns="metric", values="value").reset_index()
    targets = targets.rename(
        columns={"order_per_user": "target_order", "profit_per_user": "target_profit"}
    )

    merged = agg.merge(targets, on=["tier", "condition"], how="inner")
    merged["order_pct_error"] = (
        (merged["order_per_user"] - merged["target_order"]) / merged["target_order"] * 100
    )
    merged["profit_pct_error"] = (
        (merged["profit_per_user"] - merged["target_profit"]) / merged["target_profit"] * 100
    )
    merged["order_within_tolerance"] = merged["order_pct_error"].abs() <= TOLERANCE_PCT
    merged["profit_within_tolerance"] = merged["profit_pct_error"].abs() <= TOLERANCE_PCT
    return merged.round(4)


def sampling_noise_check(dgp: VoucherDGP, n_users: int, n_repeats: int = 20) -> pd.DataFrame:
    """Show cell-level estimates jitter across repeated draws at realistic N.

    A DGP that reproduces the target to 4 decimal places on every repeat
    at a realistic sample size would itself be a red flag (real data is
    noisy). This check reports the standard deviation of the tier="100",
    condition="no_voucher" order rate across repeated simulations, and
    compares it to the theoretical binomial standard error.

    Args:
        dgp: A fitted VoucherDGP instance.
        n_users: Realistic total sample size to simulate per repeat.
        n_repeats: Number of independent repeats.

    Returns:
        DataFrame of per-repeat order rate for the tier="100"/no_voucher
        cell, plus the theoretical binomial SE for comparison.
    """
    rates = []
    for seed in range(n_repeats):
        sim = dgp.simulate_users(n_users=n_users, seed=seed)
        cell = sim[(sim["tier"] == "100") & (sim["condition"] == "no_voucher")]
        rates.append(cell["ordered"].mean() if len(cell) else np.nan)

    rates = np.array(rates)
    p_hat = np.nanmean(rates)
    n_cell = len(sim[(sim["tier"] == "100") & (sim["condition"] == "no_voucher")])
    theoretical_se = np.sqrt(p_hat * (1 - p_hat) / max(n_cell, 1))

    return pd.DataFrame(
        {
            "repeat": np.arange(n_repeats),
            "order_rate": rates,
            "empirical_sd": np.full(n_repeats, np.nanstd(rates)),
            "theoretical_binomial_se": np.full(n_repeats, theoretical_se),
        }
    )


def placebo_check(dgp: VoucherDGP, n_users: int = 200_000, seed: int = 99) -> pd.DataFrame:
    """Shuffle condition labels and confirm the treatment "effect" vanishes.

    Args:
        dgp: A fitted VoucherDGP instance.
        n_users: Number of users to simulate.
        seed: Random seed for the shuffle.

    Returns:
        DataFrame comparing real vs. placebo order-rate spread across
        conditions within the tier="90-99" segment (the segment with the
        largest real voucher-count effect per the case study).
    """
    sim = dgp.simulate_users(n_users=n_users)
    real = sim.groupby("condition", observed=True)["ordered"].mean()

    rng = np.random.default_rng(seed)
    shuffled = sim.copy()
    shuffled["condition"] = rng.permutation(shuffled["condition"].to_numpy())
    placebo = shuffled.groupby("condition", observed=True)["ordered"].mean()

    return pd.DataFrame(
        {
            "real_order_rate": real,
            "placebo_order_rate": placebo,
        }
    ).assign(
        real_range=lambda d: d["real_order_rate"].max() - d["real_order_rate"].min(),
        placebo_range=lambda d: d["placebo_order_rate"].max() - d["placebo_order_rate"].min(),
    )


def main() -> None:
    dgp = VoucherDGP()

    print("=" * 70)
    print("1. AGGREGATION RECOVERY CHECK (vs. 7 originally-tested conditions)")
    print("=" * 70)
    recovery = aggregation_recovery_check(dgp)
    print(recovery[["tier", "condition", "target_order", "order_per_user", "order_pct_error"]])
    n_fail_order = (~recovery["order_within_tolerance"]).sum()
    n_fail_profit = (~recovery["profit_within_tolerance"]).sum()
    print(f"\nOrder cells outside +/-{TOLERANCE_PCT}%: {n_fail_order} / {len(recovery)}")
    print(f"Profit cells outside +/-{TOLERANCE_PCT}%: {n_fail_profit} / {len(recovery)}")

    print("\n" + "=" * 70)
    print("2. SAMPLING NOISE CHECK (tier=100, no_voucher, realistic N)")
    print("=" * 70)
    noise = sampling_noise_check(dgp, n_users=20_000)
    print(noise[["order_rate"]].describe())
    print(f"Empirical SD: {noise['empirical_sd'].iloc[0]:.5f}")
    print(f"Theoretical binomial SE: {noise['theoretical_binomial_se'].iloc[0]:.5f}")

    print("\n" + "=" * 70)
    print("3. PLACEBO CHECK (shuffled condition labels)")
    print("=" * 70)
    placebo = placebo_check(dgp)
    print(placebo)


if __name__ == "__main__":
    main()

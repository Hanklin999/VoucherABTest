"""
Transcribes the aggregate tables from the original PDF case study
("[E-commerce Case] A/B Testing for Voucher ROI") into a tidy long-format
CSV. This file is the ONLY place raw PDF numbers are typed in — every other
script reads from the resulting CSV, so a transcription fix only has to
happen once.

Tier standardization note: the original deck uses two slightly different
column labelings across tables ("91-99%" in some, "90-100%" in others).
These are treated as the SAME tier here and standardized to "90-99". This
is a data-cleaning decision, not a re-measurement — documented in
DGP_ASSUMPTIONS.md.
"""

from __future__ import annotations

import pandas as pd

TIERS = ["100", "90-99", "80-89", "70-79", "30-69", "0-29"]
TIER_BOUNDS = {
    "100": (1.00, 1.00),
    "90-99": (0.90, 0.99),
    "80-89": (0.80, 0.89),
    "70-79": (0.70, 0.79),
    "30-69": (0.30, 0.69),
    "0-29": (0.00, 0.29),
}

# (subsidy_usd, min_spend_usd, voucher_count) -> None means "no voucher issued"
CONDITIONS: dict[str, tuple] = {
    "no_voucher": (None, None, None),
    "s0_m199_c1": (0, 199, 1),
    "s0_m249_c3": (0, 249, 3),
    "s0_m299_c3": (0, 299, 3),
    "s9_m249_c3": (9, 249, 3),
    "s19_m199_c1": (19, 199, 1),
    "s19_m199_c3": (19, 199, 3),
}

ORDER_PER_USER = {
    "no_voucher": [0.031, 0.097, 0.091, 0.089, 0.095, 0.082],
    "s0_m199_c1": [0.039, 0.118, 0.104, 0.102, 0.100, 0.084],
    "s0_m249_c3": [0.036, 0.113, 0.107, 0.102, 0.098, 0.082],
    "s0_m299_c3": [0.035, 0.113, 0.104, 0.099, 0.097, 0.083],
    "s9_m249_c3": [0.035, 0.103, 0.095, 0.103, 0.100, 0.082],
    "s19_m199_c1": [0.033, 0.097, 0.094, 0.095, 0.101, 0.083],
    "s19_m199_c3": [0.034, 0.107, 0.098, 0.096, 0.096, 0.084],
}

PROFIT_PER_USER = {
    "no_voucher": [0.028, 0.056, 0.075, 0.098, 0.147, 0.147],
    "s0_m199_c1": [0.031, 0.065, 0.080, 0.094, 0.138, 0.148],
    "s0_m249_c3": [0.026, 0.058, 0.076, 0.095, 0.131, 0.147],
    "s0_m299_c3": [0.027, 0.064, 0.086, 0.099, 0.135, 0.147],
    "s9_m249_c3": [0.030, 0.057, 0.079, 0.108, 0.140, 0.146],
    "s19_m199_c1": [0.028, 0.060, 0.078, 0.101, 0.150, 0.148],
    "s19_m199_c3": [0.027, 0.073, 0.087, 0.102, 0.139, 0.149],
}

# Population share (%) of buyers falling into each tier — taken from the
# "Free-Shipping orders as a % of buyer's total order" table, no-voucher row.
# Raw shares sum to ~93.8%, not 100%; the ~6.2% gap is unclassified/new
# buyers not broken out in the original table. Kept as-is (not
# renormalized) and documented in DGP_ASSUMPTIONS.md.
POPULATION_SHARE_PCT = [18.4, 3.5, 7.4, 8.1, 17.3, 39.1]


def build_calibration_table() -> pd.DataFrame:
    """Assemble the tidy long-format calibration-targets DataFrame.

    Returns:
        DataFrame with one row per (condition, tier, metric) combination,
        plus rows for the population share table.
    """
    rows = []
    for condition, (subsidy, min_spend, count) in CONDITIONS.items():
        for i, tier in enumerate(TIERS):
            lo, hi = TIER_BOUNDS[tier]
            rows.append(
                {
                    "condition": condition,
                    "subsidy_usd": subsidy,
                    "min_spend_usd": min_spend,
                    "voucher_count": count,
                    "tier": tier,
                    "tier_usage_min": lo,
                    "tier_usage_max": hi,
                    "metric": "order_per_user",
                    "value": ORDER_PER_USER[condition][i],
                }
            )
            rows.append(
                {
                    "condition": condition,
                    "subsidy_usd": subsidy,
                    "min_spend_usd": min_spend,
                    "voucher_count": count,
                    "tier": tier,
                    "tier_usage_min": lo,
                    "tier_usage_max": hi,
                    "metric": "profit_per_user",
                    "value": PROFIT_PER_USER[condition][i],
                }
            )

    for i, tier in enumerate(TIERS):
        lo, hi = TIER_BOUNDS[tier]
        rows.append(
            {
                "condition": "population",
                "subsidy_usd": None,
                "min_spend_usd": None,
                "voucher_count": None,
                "tier": tier,
                "tier_usage_min": lo,
                "tier_usage_max": hi,
                "metric": "population_share_pct",
                "value": POPULATION_SHARE_PCT[i],
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import os

    df = build_calibration_table()
    os.makedirs("data/raw_benchmarks", exist_ok=True)
    df.to_csv("data/raw_benchmarks/case_summary_tables.csv", index=False)
    print(f"Wrote {len(df)} rows to data/raw_benchmarks/case_summary_tables.csv")
    print(df.head(10))

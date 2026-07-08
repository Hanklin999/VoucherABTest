"""
Budget-constrained voucher allocation for the Voucher ROI Product Science
portfolio.

=============================================================================
FRAMING
=============================================================================
The PDF's `profit_per_user` is already NET of the shipping subsidy the
platform pays (raising copay from $0 to $9 RAISES profit in most tiers in
the original tables — that only makes sense if profit already has the
subsidy cost subtracted out). So this is NOT a "spend until marginal ROI
hits zero" problem — every tested voucher config could, in principle, be
net-profitable and you'd still hand out unlimited vouchers.

The real-world constraint the PDF's own "NT$1,000,000 voucher subsidy
budget" scenario describes is different: a finance-approved CAP on total
GROSS subsidy dollars paid out this quarter, independent of whether the
program is net-profitable — a common real constraint (cash flow, budget
cycles, executive approval limits). That's what's modeled here:

    maximize  sum(incremental_net_profit)
    subject to  sum(gross_subsidy_cost) <= BUDGET

`gross_subsidy_cost` is a MODELING ASSUMPTION not given by the PDF (the PDF
only reports net profit): cost_per_user = order_rate x voucher_count x
(shipping_base_cost - copay), i.e. how much the platform pays out per
redemption, floored at 0. `shipping_base_cost` ($30 USD) and
`TOTAL_ADDRESSABLE_USERS` (500,000) are reasonable-business-assumption
placeholders, documented as such — not measured. Budget is converted from
the PDF's NT$1,000,000 at an assumed ~31.5 TWD/USD, since profit figures in
the PDF are explicitly labeled USD.

Allocation method: greedy fractional knapsack, ranked by
incremental-profit-per-dollar-of-cost. This is the OPTIMAL solution to the
continuous LP relaxation of "how much of each tier's population should get
which voucher config" — well-known result for the fractional knapsack
problem — and it's also directly implementable (\"randomize X% of this
cohort into this voucher\").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIERS_ASC = ["0-29", "30-69", "70-79", "80-89", "90-99", "100"]

SHIPPING_BASE_COST_USD = 30.0
TOTAL_ADDRESSABLE_USERS = 500_000
TWD_PER_USD = 31.5
BUDGET_TWD = 1_000_000.0
BUDGET_USD = BUDGET_TWD / TWD_PER_USD


def build_tier_condition_table(
    dgp,
    calibration: pd.DataFrame,
    shipping_base_cost: float = SHIPPING_BASE_COST_USD,
    total_addressable_users: int = TOTAL_ADDRESSABLE_USERS,
) -> pd.DataFrame:
    """Build the per-(tier, condition) economics table used for allocation.

    Uses the DGP's EXPECTED-VALUE surface (`predict_segment`), not a single
    noisy Monte Carlo draw from `experiment_log.csv`. An earlier version of
    this function used `groupby(...).mean()` on the simulated experiment
    log, and at ~1,000-5,000 simulated users per (tier, condition) cell, the
    sampling noise was large enough to flip the SIGN of incremental profit
    for at least one cell (30-69 tier, s9_m249_c3 showed a spurious -0.018
    incremental loss that shouldn't be there given how that lever was
    calibrated). Budget-allocation decisions should be made against the
    model's expected value, not a single noisy realization — this is the
    same lesson as notebook 02's power analysis: don't trust a point
    estimate you haven't checked is adequately powered.

    Args:
        dgp: A fitted VoucherDGP instance (from data_generation.py).
        calibration: Calibration targets (for population shares).
        shipping_base_cost: Assumed shipping cost basis in USD (business
            assumption, see DGP_ASSUMPTIONS.md). Parameterized (rather than
            read from the module constant) so `sensitivity_analysis` can
            sweep it without mutating global state.
        total_addressable_users: Assumed total population size.

    Returns:
        DataFrame with one row per (tier, condition): expected profit_per_user,
        order_rate, copay/min_spend/count, modeled gross cost_per_user, and
        incremental_profit_per_user relative to that tier's no_voucher baseline.
    """
    from data_generation import TIERS_ASC as DGP_TIERS

    conditions = [
        ("no_voucher", 0, 0, 0),
        ("s0_m199_c1", 0, 199, 1),
        ("s0_m249_c3", 0, 249, 3),
        ("s0_m299_c3", 0, 299, 3),
        ("s9_m249_c3", 9, 249, 3),
        ("s19_m199_c1", 19, 199, 1),
        ("s19_m199_c3", 19, 199, 3),
    ]

    rows = []
    for tier_idx, tier in enumerate(DGP_TIERS):
        for name, copay, min_spend, count in conditions:
            if name == "no_voucher":
                order_rate = dgp.base_order[tier_idx]
                profit_per_order = dgp.base_profit_per_order[tier_idx]
            else:
                order_rate, profit_per_order = dgp.predict_segment(
                    np.array([tier_idx]), np.array([copay]), np.array([min_spend]), np.array([count])
                )
                order_rate, profit_per_order = order_rate[0], profit_per_order[0]
            rows.append(
                {
                    "tier": tier,
                    "condition": name,
                    "copay_usd": copay,
                    "min_spend_usd": min_spend,
                    "voucher_count": count,
                    "order_rate": order_rate,
                    "profit_per_user": order_rate * profit_per_order,
                }
            )

    stats = pd.DataFrame(rows)
    stats["cost_per_user"] = np.where(
        stats["condition"] == "no_voucher",
        0.0,
        stats["order_rate"]
        * stats["voucher_count"]
        * np.clip(shipping_base_cost - stats["copay_usd"], 0, None),
    )

    baseline = stats[stats["condition"] == "no_voucher"].set_index("tier")["profit_per_user"]
    stats["incremental_profit_per_user"] = stats["profit_per_user"] - stats["tier"].map(baseline)

    pop_share = calibration[calibration["metric"] == "population_share_pct"].set_index("tier")
    pop_share = pop_share.reindex(TIERS_ASC)["value"]
    pop_share = pop_share / pop_share.sum()
    stats["population_n"] = (
        stats["tier"].map(pop_share) * total_addressable_users
    ).round().astype(int)

    stats["roi_ratio"] = np.where(
        stats["cost_per_user"] > 0,
        stats["incremental_profit_per_user"] / stats["cost_per_user"],
        np.where(stats["incremental_profit_per_user"] > 0, np.inf, -np.inf),
    )
    return stats.round(6)


def greedy_fractional_allocation(
    econ_table: pd.DataFrame, budget: float = BUDGET_USD
) -> tuple[pd.DataFrame, dict]:
    """Allocate budget across (tier, condition) options via bang-per-buck greedy.

    Optimal for the LP relaxation: sort candidate options by ROI descending,
    fill each tier's remaining population at the best available option until
    either that tier's population or the remaining budget is exhausted.

    Args:
        econ_table: Output of `build_tier_condition_table`.
        budget: Total budget available, in USD.

    Returns:
        Tuple of (allocation DataFrame with one row per funded (tier, condition,
        n_allocated) decision, summary dict with totals).
    """
    candidates = econ_table[
        (econ_table["condition"] != "no_voucher") & (econ_table["incremental_profit_per_user"] > 0)
    ].sort_values("roi_ratio", ascending=False)

    remaining_capacity = econ_table.drop_duplicates("tier").set_index("tier")["population_n"].to_dict()
    remaining_budget = budget
    allocations = []

    for _, row in candidates.iterrows():
        if remaining_budget <= 0 or remaining_capacity[row["tier"]] <= 0:
            continue
        max_by_capacity = remaining_capacity[row["tier"]]
        max_by_budget = remaining_budget / row["cost_per_user"] if row["cost_per_user"] > 0 else max_by_capacity
        n_allocated = int(min(max_by_capacity, max_by_budget))
        if n_allocated <= 0:
            continue

        allocations.append(
            {
                "tier": row["tier"],
                "condition": row["condition"],
                "n_allocated": n_allocated,
                "cost": n_allocated * row["cost_per_user"],
                "incremental_profit": n_allocated * row["incremental_profit_per_user"],
                "roi_ratio": row["roi_ratio"],
            }
        )
        remaining_capacity[row["tier"]] -= n_allocated
        remaining_budget -= n_allocated * row["cost_per_user"]

    alloc_df = pd.DataFrame(allocations).round(2)
    summary = {
        "total_cost": alloc_df["cost"].sum() if len(alloc_df) else 0.0,
        "total_incremental_profit": alloc_df["incremental_profit"].sum() if len(alloc_df) else 0.0,
        "budget": budget,
        "budget_utilization_pct": (alloc_df["cost"].sum() / budget * 100) if len(alloc_df) else 0.0,
        "users_reached": alloc_df["n_allocated"].sum() if len(alloc_df) else 0,
    }
    summary["blended_roi_pct"] = (
        summary["total_incremental_profit"] / summary["total_cost"] * 100
        if summary["total_cost"] > 0
        else 0.0
    )
    return alloc_df, summary


def evaluate_fixed_strategy(
    econ_table: pd.DataFrame, target_tiers: list[str], condition: str, budget: float
) -> dict:
    """Evaluate one of the PDF's named strategies under the SAME budget cap.

    Splits the budget evenly (by population) across all listed target tiers
    at the given condition, capped at each tier's population and at the
    overall budget — for an apples-to-apples comparison against the greedy
    allocation above.

    Args:
        econ_table: Output of `build_tier_condition_table`.
        target_tiers: List of tier labels this strategy targets.
        condition: The single voucher condition applied to all target tiers.
        budget: Total budget available, in USD.

    Returns:
        Dict with total cost, incremental profit, users reached, and blended ROI.
    """
    rows = econ_table[
        (econ_table["tier"].isin(target_tiers)) & (econ_table["condition"] == condition)
    ]
    remaining_budget = budget
    total_cost, total_profit, users_reached = 0.0, 0.0, 0

    for _, row in rows.iterrows():
        if remaining_budget <= 0:
            break
        max_by_capacity = row["population_n"]
        max_by_budget = remaining_budget / row["cost_per_user"] if row["cost_per_user"] > 0 else max_by_capacity
        n = int(min(max_by_capacity, max_by_budget))
        total_cost += n * row["cost_per_user"]
        total_profit += n * row["incremental_profit_per_user"]
        users_reached += n
        remaining_budget -= n * row["cost_per_user"]

    return {
        "total_cost": round(total_cost, 2),
        "total_incremental_profit": round(total_profit, 2),
        "users_reached": users_reached,
        "budget_utilization_pct": round(total_cost / budget * 100, 1) if budget else 0.0,
        "blended_roi_pct": round(total_profit / total_cost * 100, 1) if total_cost > 0 else 0.0,
    }


def sensitivity_analysis(calibration: pd.DataFrame, calibration_path: str = "data/raw_benchmarks/case_summary_tables.csv") -> pd.DataFrame:
    """Sweep the three unvalidated business assumptions and re-run allocation.

    Answers: how much does the recommended allocation and its ROI change if
    `shipping_base_cost`, `TOTAL_ADDRESSABLE_USERS`, or the TWD/USD rate are
    wrong? These three were stated assumptions, not calibrated values (see
    DGP_ASSUMPTIONS.md) — this table is what makes that caveat concrete
    instead of just a warning in prose.

    Args:
        calibration: Calibration targets (for population shares).
        calibration_path: Path to the calibration CSV, passed through to
            `VoucherDGP` (needs overriding when called from a notebook in a
            subdirectory, since the default is relative to the project root).

    Returns:
        DataFrame with one row per assumption combination: top-allocated
        tier/condition, total incremental profit, and blended ROI.
    """
    from data_generation import VoucherDGP

    dgp = VoucherDGP(calibration_path=calibration_path)
    scenarios = []
    for base_cost in [20.0, 30.0, 40.0]:
        for total_pop in [250_000, 500_000, 1_000_000]:
            for fx in [28.0, 31.5, 35.0]:
                budget_usd = BUDGET_TWD / fx
                econ_table = build_tier_condition_table(
                    dgp, calibration, shipping_base_cost=base_cost, total_addressable_users=total_pop
                )
                alloc_df, summary = greedy_fractional_allocation(econ_table, budget_usd)
                top_choice = (
                    f"{alloc_df.iloc[0]['tier']}/{alloc_df.iloc[0]['condition']}"
                    if len(alloc_df)
                    else "none"
                )
                scenarios.append(
                    {
                        "shipping_base_cost": base_cost,
                        "total_addressable_users": total_pop,
                        "twd_per_usd": fx,
                        "budget_usd": round(budget_usd, 2),
                        "top_allocated_segment": top_choice,
                        "total_incremental_profit": round(summary["total_incremental_profit"], 2),
                        "blended_roi_pct": round(summary["blended_roi_pct"], 2),
                        "users_reached": summary["users_reached"],
                    }
                )

    return pd.DataFrame(scenarios)



def main() -> None:
    from data_generation import VoucherDGP

    calibration = pd.read_csv("data/raw_benchmarks/case_summary_tables.csv")
    dgp = VoucherDGP()

    econ_table = build_tier_condition_table(dgp, calibration)
    print("=" * 70)
    print(f"BUDGET: NT${BUDGET_TWD:,.0f} = ${BUDGET_USD:,.2f} USD (at {TWD_PER_USD} TWD/USD)")
    print(f"ADDRESSABLE POPULATION: {TOTAL_ADDRESSABLE_USERS:,} users")
    print("=" * 70)
    print(
        econ_table[
            ["tier", "condition", "profit_per_user", "cost_per_user", "incremental_profit_per_user", "roi_ratio"]
        ].sort_values(["tier", "roi_ratio"], ascending=[True, False]).to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("OUR ALLOCATION: greedy fractional knapsack (optimal LP relaxation)")
    print("=" * 70)
    alloc_df, summary = greedy_fractional_allocation(econ_table)
    print(alloc_df.to_string(index=False))
    print(f"\nTotal cost: ${summary['total_cost']:,.2f} ({summary['budget_utilization_pct']:.1f}% of budget)")
    print(f"Total incremental profit: ${summary['total_incremental_profit']:,.2f}")
    print(f"Users reached: {summary['users_reached']:,}")
    print(f"Blended ROI: {summary['blended_roi_pct']:.1f}%")

    print("\n" + "=" * 70)
    print("BENCHMARK: PDF's 3 named strategies, evaluated at the SAME budget")
    print("=" * 70)
    scenario_1 = evaluate_fixed_strategy(econ_table, ["90-99"], "s0_m249_c3", BUDGET_USD)
    scenario_2 = evaluate_fixed_strategy(econ_table, ["30-69"], "s9_m249_c3", BUDGET_USD)
    scenario_3_high = evaluate_fixed_strategy(econ_table, ["90-99"], "s0_m249_c3", BUDGET_USD * 0.6)
    scenario_3_mod = evaluate_fixed_strategy(econ_table, ["30-69"], "s9_m249_c3", BUDGET_USD * 0.4)
    scenario_3 = {
        "total_cost": scenario_3_high["total_cost"] + scenario_3_mod["total_cost"],
        "total_incremental_profit": scenario_3_high["total_incremental_profit"]
        + scenario_3_mod["total_incremental_profit"],
        "users_reached": scenario_3_high["users_reached"] + scenario_3_mod["users_reached"],
    }
    scenario_3["blended_roi_pct"] = round(
        scenario_3["total_incremental_profit"] / scenario_3["total_cost"] * 100, 1
    ) if scenario_3["total_cost"] > 0 else 0.0

    for name, result in [
        ("Scenario 1 (Maximize Order Volume: 90-99 tier only)", scenario_1),
        ("Scenario 2 (Maximize Profit/User: 30-69 tier only)", scenario_2),
        ("Scenario 3 (Balanced: 60% high / 40% moderate)", scenario_3),
        ("Our greedy allocation (all tiers, ROI-ranked)", summary),
    ]:
        print(
            f"{name}: profit=${result['total_incremental_profit']:,.0f}, "
            f"cost=${result['total_cost']:,.0f}, "
            f"ROI={result['blended_roi_pct']:.1f}%, "
            f"users={result['users_reached']:,}"
        )

    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS (27 combinations of the 3 unvalidated assumptions)")
    print("=" * 70)
    sensitivity = sensitivity_analysis(calibration)
    print(sensitivity.to_string(index=False))
    sensitivity.to_csv("outputs/sensitivity_analysis.csv", index=False)
    n_unique_segments = sensitivity["top_allocated_segment"].nunique()
    print(
        f"\nTop-allocated segment changes across {n_unique_segments} different "
        f"(tier, condition) pairs depending on assumptions -> allocation is "
        f"{'SENSITIVE' if n_unique_segments > 1 else 'ROBUST'} to these assumptions."
    )


if __name__ == "__main__":
    main()

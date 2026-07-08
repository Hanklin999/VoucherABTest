# DGP Assumptions — Voucher ROI Product Science Portfolio

Every number in the simulation traces back to one of three sources:
(A) directly transcribed from the PDF case study, (B) derived arithmetically
from transcribed numbers, or (C) a modeling assumption imposed to fill a gap
the PDF doesn't cover. This document lists which is which.

## (A) Directly transcribed (data/raw_benchmarks/case_summary_tables.csv)

- Population share by usage tier (6 values, from the "Free-Shipping orders
  as a % of buyer's total order" table). Raw shares sum to 93.8%, not 100%;
  the ~6.2% gap is unclassified/new buyers not broken out in the original
  table. **Not renormalized at the source** — renormalized only at
  simulation time (`simulate_users`), so the raw CSV stays a faithful
  transcription.
- order_per_user and profit_per_user for 7 tested conditions x 6 tiers
  (84 values total): no_voucher, and six voucher configurations spanning
  copay $0/$9/$19, min_spend $199/$249/$299, count #1/#3.

**Tier-label cleanup**: the original deck uses "91-99%" in one table and
"90-100%" in another for what is clearly the same segment. Standardized to
"90-99" here. This is a labeling harmonization, not a re-measurement.

## (B) Derived arithmetically (src/data_generation.py, `VoucherDGP.__init__`)

- `profit_per_order` (conditional on ordering) = `profit_per_user / order_per_user`,
  computed per tier per condition. The PDF only reports per-user (unconditional)
  averages; dividing by order rate recovers the conditional value needed to
  simulate individual transactions.
- **Copay slope** (effect per $1 of buyer copay): averaged from two
  *independent* observed deltas — ($0→$9 at min_spend=249, count=3) and
  ($0→$19 at min_spend=199, count=1). Averaging two slopes measured under
  different min_spend/count conditions assumes the copay effect doesn't
  interact with those other levers (additive-model assumption, see below).
- **Min-spend slope**: from the single observed $249→$299 delta (copay=0,
  count=3). Only one data point exists for this lever, so there's no way to
  check whether the slope is linear beyond this range — it's assumed linear
  by default, not validated.
- **Voucher-count effect** (#1 vs #3): from the single observed pair at
  copay=$19, min_spend=$199. Applied as a constant, tier-specific shift.

## (C) Modeling assumptions (imposed, not measured)

1. **Additive separability across levers.** `order_prob = intercept +
   copay_effect + minspend_effect + count_effect`, with no lever x lever
   interaction terms. This is forced by data sparsity — 7 conditions can't
   identify 3-way interactions. Consequence: predictions for combinations
   *outside* the 7 tested conditions (e.g. copay=$29) are genuine
   extrapolations, not reproductions, and should be labeled as such in any
   downstream report.
2. **Linear extrapolation in copay and min_spend.** Both levers are only
   observed at 2-3 points; the model assumes the per-dollar slope holds
   outside the tested range (e.g. copay=$29 is 29x the per-dollar slope, not
   independently validated at $29). Reasonable as a first-order approximation
   for a portfolio piece; a real experimentation team would need to actually
   *test* $29 before trusting this.
3. **Discrete per-tier lookup, not continuous interpolation.** An earlier
   version of this DGP interpolated smoothly across a continuous latent
   `usage_rate` in [0,1]. Validation caught that tier "100" (buyers whose
   free-shipping usage is literally 100% of orders) has a **much lower**
   order rate than tier "90-99" — almost certainly because the "100%" tier
   is dominated by one-time buyers who never had a chance to NOT use free
   shipping. Smoothing across that discontinuity dragged tier "90-99"'s
   predicted order rate down by 20-38%. Fixed by switching to exact discrete
   tier lookup; `usage_rate` is still simulated (uniform within each tier's
   numeric range) but only as a descriptive covariate, not an outcome-model
   input. **This bug-and-fix is intentionally kept visible in the code
   history/comments** — catching this kind of monotonicity violation via
   validation, rather than assuming smoothness, is the actual skill being
   demonstrated here.
4. **Anchor point for the additive model.** The intercept is anchored at the
   directly-observed (copay=$0, min_spend=$249, count=3) cell — NOT at
   `no_voucher`. An earlier version conflated these two, treating "no voucher
   issued" as equivalent to "voucher issued with $0 copay," which is wrong:
   issuing a voucher at all carries its own level shift (e.g., tier "90-99"
   jumps from 0.097 order/user with no voucher to 0.118 with a $0-copay
   voucher) separate from how generous its terms are. Conflating the two
   caused 15-30% errors on non-anchor cells; anchoring correctly reduced
   that to 1/42 order cells and 6/42 profit cells outside a ±15% tolerance
   band (see notebooks/01 validation output).
5. **Individual-level noise**: order outcome is Bernoulli(predicted
   probability); profit given an order is `profit_per_order x Lognormal(0,
   0.35)`. The lognormal spread (σ=0.35) is a reasonable-business-assumption
   placeholder, not derived from the PDF (which has no individual-level
   variance information) — chosen to be wide enough that repeated small-N
   simulations show realistic sampling jitter (validation.py check #2).

## Known limitation (stated, not hidden)

The residual ~15-22% errors that remain (concentrated in tier "90-99" under
stacked copay+count extrapolation) most likely indicate a genuine
**copay x count interaction** that an additive model can't capture — the
case study's own narrative says the count effect is "concentrated" in this
exact tier, which is itself a form of interaction (tier x count), and it's
plausible copay x count interacts too. A follow-up with a saturated
tier-by-condition model (no additivity assumption) would fit these 7 points
exactly, at the cost of being unable to predict any untested combination —
a classic bias-variance tradeoff, and one worth stating explicitly in the
final README rather than quietly picking whichever number looks better.

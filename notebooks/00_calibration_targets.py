# %% [markdown]
# # 00 — Calibration Targets
#
# Transcribes the aggregate tables from the original PDF case study into a
# tidy long-format CSV. This is the single source of truth every later
# notebook validates against — see `build_calibration_targets.py` for the
# full transcription and the tier-label harmonization note.

# %%
import sys

sys.path.insert(0, "..")

from build_calibration_targets import build_calibration_table

# %% [markdown]
# ## Build and preview the calibration table

# %%
calibration = build_calibration_table()
calibration.to_csv("../data/raw_benchmarks/case_summary_tables.csv", index=False)
print(f"{len(calibration)} rows written to data/raw_benchmarks/case_summary_tables.csv")
calibration.head(10)

# %% [markdown]
# ## Sanity check: population shares sum to ~93.8% (not 100%)
#
# The ~6.2% gap is unclassified/new buyers not broken out in the original
# table — not renormalized here, only at simulation time. See
# `DGP_ASSUMPTIONS.md` for the full explanation.

# %%
pop = calibration[calibration["metric"] == "population_share_pct"]
print(pop[["tier", "value"]])
print(f"\nSum: {pop['value'].sum():.1f}%")

"""Descriptive variability across three repeated split seeds for manuscript tables."""
from pathlib import Path
import pandas as pd

from common.config import DEFAULT_SEEDS
from common.paths import RUN_RESULTS_ROOT


def summarize(frame, keys, value):
    counts = frame.groupby(keys, dropna=False).seed.nunique()
    if not counts.eq(len(DEFAULT_SEEDS)).all():
        raise ValueError("Each reported condition must contain all three seeds")
    return frame.groupby(keys, as_index=False, dropna=False)[value].agg(
        mean="mean", sd="std", minimum="min", maximum="max", n="count"
    )


def main():
    inputs = RUN_RESULTS_ROOT / "paper_inputs"
    out = RUN_RESULTS_ROOT / "paper_inputs" / "seed_dispersion"
    out.mkdir(parents=True, exist_ok=True)
    perf = pd.read_csv(inputs / "performance_by_run.csv")
    perf = perf[perf.method.eq("Baseline")]
    if perf.duplicated(["dataset", "model", "seed"]).any():
        raise ValueError("Duplicate baseline performance rows")
    for metric in ["roc_auc", "balanced_accuracy"]:
        summarize(perf, ["dataset", "model"], metric).to_csv(out / f"table2_{metric}.csv", index=False)
    paired = pd.read_csv(inputs / "mitigation_paired.csv")
    for metric in ["baseline_aod", "aod", "delta_ba"]:
        summarize(paired, ["dataset", "model", "method"], metric).to_csv(out / f"table3_{metric}.csv", index=False)
    audit = pd.read_csv(inputs / "audit_by_condition.csv")
    audit = audit[audit.primary & audit.axis.eq("Intersection")]
    sex_eo = audit[audit.method.eq("EO") & audit.target.eq("Sex/gender")]
    summarize(sex_eo, ["dataset", "model", "method", "target"], "aod").to_csv(out / "table3_eo_sex_aod.csv", index=False)
    audit = audit[(audit.method.eq("Baseline") & audit.target.eq("none")) | audit.method.eq("EO")]
    # Each seed contributes the mean of the three models, matching Table 6.
    by_seed = audit.groupby(["dataset", "method", "target", "seed"], as_index=False).agg(
        aod=("aod", "mean"), n_models=("model", "nunique")
    )
    if not by_seed.n_models.eq(3).all():
        raise ValueError("Table 6 condition has fewer than three models")
    summarize(by_seed, ["dataset", "method", "target"], "aod").to_csv(out / "table6_aod.csv", index=False)
    print(f"Wrote seed dispersion tables to {out}")


if __name__ == "__main__":
    main()

"""Summarize paired Yale ED baseline and mitigation results across audit axes."""
from pathlib import Path
import argparse

import pandas as pd

from common.paths import run_results_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=run_results_dir("yale_ed"))
    args = parser.parse_args()
    gaps = pd.read_csv(args.run_dir / "fairness_gaps_by_split.csv")
    performance = pd.read_csv(args.run_dir / "overall_by_split.csv")
    seeds = {20261027, 20261028, 20261029}
    models = {"logistic_regression", "random_forest", "xgboost"}
    targets = {"gender", "race", "ethnicity", "race_x_gender", "age_band", "race_x_gender_x_age_band"}
    expected = {(s, m, "baseline", "none") for s in seeds for m in models}
    expected |= {(s, m, method, t) for s in seeds for m in models
                 for method in ("reweighing", "threshold_optimizer") for t in targets}
    keys = ["seed", "model", "method", "target_axis"]
    observed = set(map(tuple, performance[keys].itertuples(index=False, name=None)))
    if observed != expected or performance.duplicated(keys).any():
        raise ValueError(f"Run incomplete or inconsistent: missing {len(expected - observed)}, unexpected {len(observed - expected)} performance rows")
    audit_counts = gaps.groupby(keys).audit_axis.nunique()
    if len(audit_counts) != len(expected) or not (audit_counts == 11).all():
        raise ValueError("Run incomplete: each evaluation must include all 11 audit axes")
    gaps["aod"] = (gaps.tpr_gap + gaps.fpr_gap) / 2
    base = gaps[gaps.method.eq("baseline")][["seed", "model", "audit_axis", "aod"]]
    base = base.rename(columns={"aod": "baseline_aod"})
    mitigated = gaps[gaps.method.isin(["reweighing", "threshold_optimizer"])].copy()
    paired = mitigated.merge(base, on=["seed", "model", "audit_axis"], validate="many_to_one")
    paired["delta_aod"] = paired.aod - paired.baseline_aod
    perf = performance[["seed", "model", "method", "target_axis", "balanced_accuracy_expected"]]
    paired = paired.merge(perf, on=["seed", "model", "method", "target_axis"], validate="many_to_one")
    baseline_perf = performance[performance.method.eq("baseline")][["seed", "model", "balanced_accuracy_expected"]]
    baseline_perf = baseline_perf.rename(columns={"balanced_accuracy_expected": "baseline_balanced_accuracy"})
    paired = paired.merge(baseline_perf, on=["seed", "model"], validate="many_to_one")
    paired["delta_balanced_accuracy"] = paired.balanced_accuracy_expected - paired.baseline_balanced_accuracy
    paired.to_csv(args.run_dir / "paired_effects_by_split.csv", index=False)
    summary = paired.groupby(["method", "target_axis", "audit_axis"]).agg(
        evaluations=("aod", "count"), models=("model", "nunique"), seeds=("seed", "nunique"),
        baseline_aod_mean=("baseline_aod", "mean"), aod_mean=("aod", "mean"),
        aod_min=("aod", "min"), aod_max=("aod", "max"), delta_aod_mean=("delta_aod", "mean"),
        balanced_accuracy_mean=("balanced_accuracy_expected", "mean"),
        delta_balanced_accuracy_mean=("delta_balanced_accuracy", "mean"),
        minimum_supported_groups=("groups_in_gap_n", "min"),
    ).reset_index()
    summary.to_csv(args.run_dir / "paired_effects_summary.csv", index=False)
    groups = pd.read_csv(args.run_dir / "group_metrics_by_split.csv")
    group_keys = ["seed", "model", "audit_axis", "group"]
    group_base = groups[groups.method.eq("baseline")][group_keys + ["tpr_expected", "fpr_expected", "support_status"]]
    group_base = group_base.rename(columns={"tpr_expected": "baseline_tpr", "fpr_expected": "baseline_fpr", "support_status": "baseline_support_status"})
    group_effects = groups[groups.method.isin(["reweighing", "threshold_optimizer"])].merge(group_base, on=group_keys, validate="many_to_one")
    group_effects["delta_tpr"] = group_effects.tpr_expected - group_effects.baseline_tpr
    group_effects["delta_fpr"] = group_effects.fpr_expected - group_effects.baseline_fpr
    group_effects.to_csv(args.run_dir / "paired_group_effects_by_split.csv", index=False)
    print(f"Wrote {len(paired)} paired evaluations and {len(summary)} summary rows")


if __name__ == "__main__":
    main()

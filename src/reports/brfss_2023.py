"""Rebuild BRFSS aggregate result tables from checkpoint-level group outputs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


from common.paths import PROJECT_ROOT, run_results_dir


ROOT = PROJECT_ROOT
OUT = run_results_dir("brfss_2023")
AUDIT_AXES = ["race_ethnicity", "sex", "age_band", "race_x_sex", "race_x_age", "sex_x_age", "race_x_sex_x_age"]
TARGET_AXES = ["sex", "race_ethnicity", "age_band", "race_x_sex", "race_x_sex_x_age"]
SUPPORT_CUTOFFS = [20, 30, 50, 100, 200]


def grouped_gap_rows(group_metrics: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    keys = ["seed", "model", "method", "target_axis", "audit_axis"]
    rows = []
    for key, frame in group_metrics.groupby(keys, dropna=False, sort=False):
        base = dict(zip(keys, key))
        supported = frame[(frame.positive_n >= cutoff) & (frame.negative_n >= cutoff)]
        for scope, selected in [
            ("primary_substantive", supported[supported.substantive_group.astype(bool)]),
            ("including_unknown_refused", supported),
        ]:
            entry = {**base, "gap_scope": scope, "n_groups": int(selected.group.nunique()),
                     "support_threshold_each_class": cutoff}
            for suffix in ["w", "unw"]:
                for metric in ["tpr", "fpr", "selection_rate", "ppv"]:
                    values = selected[f"{metric}_{suffix}"].dropna().to_numpy()
                    entry[f"{metric}_gap_{suffix}"] = float(values.max() - values.min()) if len(values) >= 2 else np.nan
                entry[f"average_odds_gap_{suffix}"] = (
                    (entry[f"tpr_gap_{suffix}"] + entry[f"fpr_gap_{suffix}"]) / 2
                    if np.isfinite(entry[f"tpr_gap_{suffix}"]) and np.isfinite(entry[f"fpr_gap_{suffix}"])
                    else np.nan
                )
            rows.append(entry)
    return pd.DataFrame(rows)


def write_mean_range(frame, keys, value_columns, path):
    summary = frame.groupby(keys, dropna=False)[value_columns].agg(["mean", "min", "max"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(path, index=False)


def main():
    performance = pd.read_csv(OUT / "performance_by_split.csv")
    groups = pd.read_csv(OUT / "group_metrics_by_split.csv")
    gaps = grouped_gap_rows(groups, cutoff=100)
    gaps.to_csv(OUT / "fairness_gaps_by_split.csv", index=False)
    pcols = [c for c in performance.columns if c.endswith("_w") or c.endswith("_unw")]
    write_mean_range(performance, ["model", "method", "target_axis"], pcols, OUT / "performance_summary.csv")
    gcols = [c for c in gaps.columns if c.endswith("_w") or c.endswith("_unw")]
    write_mean_range(gaps, ["model", "method", "target_axis", "audit_axis", "gap_scope"], gcols,
                     OUT / "fairness_gap_summary.csv")

    sensitivity_rows = []
    for cutoff in SUPPORT_CUTOFFS:
        current = grouped_gap_rows(groups, cutoff=cutoff)
        sensitivity_rows.append(current)
    sensitivity = pd.concat(sensitivity_rows, ignore_index=True)
    sensitivity.to_csv(OUT / "support_sensitivity_by_split.csv", index=False)
    s = sensitivity[sensitivity.gap_scope == "primary_substantive"]
    summary = s.groupby(["support_threshold_each_class", "model", "method", "target_axis", "audit_axis"], dropna=False).agg(
        mean_average_odds_gap=("average_odds_gap_unw", "mean"),
        min_average_odds_gap=("average_odds_gap_unw", "min"),
        max_average_odds_gap=("average_odds_gap_unw", "max"),
        mean_tpr_gap=("tpr_gap_unw", "mean"), mean_fpr_gap=("fpr_gap_unw", "mean"),
        mean_supported_groups=("n_groups", "mean"),
    ).reset_index()
    summary.to_csv(OUT / "support_sensitivity_summary.csv", index=False)

    base = gaps[(gaps.method == "baseline") & (gaps.gap_scope == "primary_substantive")]
    base_target = base.set_index(["seed", "model", "audit_axis"])
    base_perf = performance[performance.method == "baseline"].set_index(["seed", "model"])
    target_rows = []
    cross_rows = []
    for method in ["reweighing", "equalized_odds_postprocessing"]:
        for target in TARGET_AXES:
            target_gaps = gaps[(gaps.method == method) & (gaps.target_axis == target) &
                               (gaps.audit_axis == target) & (gaps.gap_scope == "primary_substantive")]
            paired_gap = target_gaps.merge(
                base_target[["average_odds_gap_unw"]], left_on=["seed", "model", "audit_axis"],
                right_index=True, suffixes=("_mitigated", "_baseline"),
            )
            effect = paired_gap.average_odds_gap_unw_mitigated - paired_gap.average_odds_gap_unw_baseline
            current_perf = performance[(performance.method == method) & (performance.target_axis == target)].merge(
                base_perf[["balanced_accuracy_unw", "roc_auc_unw"]], left_on=["seed", "model"],
                right_index=True, suffixes=("", "_baseline"),
            )
            delta_ba = current_perf.balanced_accuracy_unw - current_perf.balanced_accuracy_unw_baseline
            delta_auc = current_perf.roc_auc_unw - current_perf.roc_auc_unw_baseline
            target_rows.append({
                "method": method, "target_axis": target,
                "baseline_target_AOD": paired_gap.average_odds_gap_unw_baseline.mean(),
                "mitigated_target_AOD": paired_gap.average_odds_gap_unw_mitigated.mean(),
                "delta_target_AOD": effect.mean(), "delta_target_AOD_min": effect.min(),
                "delta_target_AOD_max": effect.max(),
                "baseline_balanced_accuracy": current_perf.balanced_accuracy_unw_baseline.mean(),
                "mitigated_balanced_accuracy": current_perf.balanced_accuracy_unw.mean(),
                "delta_balanced_accuracy": delta_ba.mean(), "delta_auc": delta_auc.mean(),
            })
            all_gaps = gaps[(gaps.method == method) & (gaps.target_axis == target) &
                            (gaps.gap_scope == "primary_substantive")]
            for axis in AUDIT_AXES:
                current = all_gaps[all_gaps.audit_axis == axis].merge(
                    base_target[["average_odds_gap_unw"]], left_on=["seed", "model", "audit_axis"],
                    right_index=True, suffixes=("_mitigated", "_baseline"),
                )
                delta = current.average_odds_gap_unw_mitigated - current.average_odds_gap_unw_baseline
                cross_rows.append({"method": method, "target_axis": target, "audit_axis": axis,
                                   "baseline_AOD": current.average_odds_gap_unw_baseline.mean(),
                                   "mitigated_AOD": current.average_odds_gap_unw_mitigated.mean(),
                                   "delta_AOD": delta.mean(), "delta_AOD_min": delta.min(),
                                   "delta_AOD_max": delta.max()})
    pd.DataFrame(target_rows).to_csv(OUT / "targeted_mitigation_summary.csv", index=False)
    pd.DataFrame(cross_rows).to_csv(OUT / "cross_axis_mitigation_summary.csv", index=False)


if __name__ == "__main__":
    main()

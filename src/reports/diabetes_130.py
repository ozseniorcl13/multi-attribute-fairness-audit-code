"""Create descriptive aggregate summaries for harmonized Diabetes 130 runs."""
from pathlib import Path
import numpy as np
import pandas as pd

from common.paths import PROJECT_ROOT, run_results_dir

ROOT = PROJECT_ROOT
OUT = run_results_dir("diabetes_130")
TARGET_AXES = ["gender", "race", "age", "race_x_gender", "race_x_gender_x_age"]
AUDIT_AXES = ["race", "gender", "age", "race_x_gender", "race_x_age", "gender_x_age", "race_x_gender_x_age"]


def build_gaps(groups, cutoff):
    keys = ["seed", "model", "method", "target_axis", "audit_axis"]
    rows = []
    for key, frame in groups.groupby(keys, dropna=False, sort=False):
        base = dict(zip(keys, key))
        eligible = frame[(frame.positive_n >= cutoff) & (frame.negative_n >= cutoff)]
        for scope, selected in [
            ("primary_substantive", eligible[eligible.substantive_group.astype(bool)]),
            ("including_unknown", eligible),
        ]:
            r = {**base, "gap_scope": scope, "support_each_class": cutoff,
                 "n_groups": int(selected.group.nunique())}
            for metric in ["tpr", "fpr", "selection_rate", "ppv"]:
                values = selected[metric].dropna().to_numpy()
                r[f"{metric}_gap"] = float(values.max() - values.min()) if len(values) >= 2 else np.nan
            r["average_odds_gap"] = ((r["tpr_gap"] + r["fpr_gap"]) / 2
                                     if np.isfinite(r["tpr_gap"]) and np.isfinite(r["fpr_gap"]) else np.nan)
            rows.append(r)
    return pd.DataFrame(rows)


def mean_range(df, keys, cols, path):
    z = df.groupby(keys, dropna=False)[cols].agg(["mean", "min", "max"])
    z.columns = [f"{c}_{s}" for c, s in z.columns]
    z.reset_index().to_csv(path, index=False)


def main():
    perf = pd.read_csv(OUT / "performance_by_split.csv")
    group = pd.read_csv(OUT / "group_metrics_by_split.csv")
    gap = build_gaps(group, 100)
    gap.to_csv(OUT / "fairness_gaps_by_split.csv", index=False)
    metric_cols = [c for c in perf if c in ["accuracy", "balanced_accuracy", "tpr", "fpr", "specificity", "ppv",
                  "selection_rate", "f1", "mcc", "roc_auc", "average_precision"]]
    mean_range(perf, ["model", "method", "target_axis"], metric_cols, OUT / "performance_summary.csv")
    gap_cols = ["n_groups", "tpr_gap", "fpr_gap", "average_odds_gap", "selection_rate_gap", "ppv_gap"]
    mean_range(gap, ["model", "method", "target_axis", "audit_axis", "gap_scope"], gap_cols,
               OUT / "fairness_gap_summary.csv")

    sens = pd.concat([build_gaps(group, x).assign(support_sensitivity_cutoff=x) for x in [20, 50, 100, 200]],
                     ignore_index=True)
    sens.to_csv(OUT / "support_sensitivity_by_split.csv", index=False)
    sens_summary = sens[sens.gap_scope == "primary_substantive"].groupby(
        ["support_sensitivity_cutoff", "method", "target_axis", "audit_axis"], dropna=False
    ).agg(mean_AOD=("average_odds_gap", "mean"), min_AOD=("average_odds_gap", "min"),
          max_AOD=("average_odds_gap", "max"), mean_groups=("n_groups", "mean")).reset_index()
    sens_summary.to_csv(OUT / "support_sensitivity_summary.csv", index=False)

    base_gap = gap[(gap.method == "baseline") & (gap.gap_scope == "primary_substantive")]
    base_gap_index = base_gap.set_index(["seed", "model", "audit_axis"])
    base_perf = perf[perf.method == "baseline"].set_index(["seed", "model"])
    target_rows, cross_rows = [], []
    for method in ["reweighing", "equalized_odds_postprocessing"]:
        for target in TARGET_AXES:
            target_gap = gap[(gap.method == method) & (gap.target_axis == target) &
                             (gap.audit_axis == target) & (gap.gap_scope == "primary_substantive")].merge(
                base_gap_index[["average_odds_gap"]], left_on=["seed", "model", "audit_axis"],
                right_index=True, suffixes=("_mitigated", "_baseline"))
            dg = target_gap.average_odds_gap_mitigated - target_gap.average_odds_gap_baseline
            p = perf[(perf.method == method) & (perf.target_axis == target)].merge(
                base_perf[["balanced_accuracy", "roc_auc"]], left_on=["seed", "model"],
                right_index=True, suffixes=("", "_baseline"))
            dba = p.balanced_accuracy - p.balanced_accuracy_baseline
            target_rows.append({"method": method, "target_axis": target,
                                "baseline_target_AOD": target_gap.average_odds_gap_baseline.mean(),
                                "mitigated_target_AOD": target_gap.average_odds_gap_mitigated.mean(),
                                "delta_target_AOD": dg.mean(), "delta_target_AOD_min": dg.min(),
                                "delta_target_AOD_max": dg.max(),
                                "baseline_balanced_accuracy": p.balanced_accuracy_baseline.mean(),
                                "mitigated_balanced_accuracy": p.balanced_accuracy.mean(),
                                "delta_balanced_accuracy": dba.mean(),
                                "delta_ROC_AUC": (p.roc_auc - p.roc_auc_baseline).mean()})
            for audit in AUDIT_AXES:
                x = gap[(gap.method == method) & (gap.target_axis == target) &
                        (gap.audit_axis == audit) & (gap.gap_scope == "primary_substantive")].merge(
                    base_gap_index[["average_odds_gap"]], left_on=["seed", "model", "audit_axis"],
                    right_index=True, suffixes=("_mitigated", "_baseline"))
                dx = x.average_odds_gap_mitigated - x.average_odds_gap_baseline
                cross_rows.append({"method": method, "target_axis": target, "audit_axis": audit,
                                   "baseline_AOD": x.average_odds_gap_baseline.mean(),
                                   "mitigated_AOD": x.average_odds_gap_mitigated.mean(),
                                   "delta_AOD": dx.mean(), "delta_AOD_min": dx.min(),
                                   "delta_AOD_max": dx.max()})
    pd.DataFrame(target_rows).to_csv(OUT / "targeted_mitigation_summary.csv", index=False)
    pd.DataFrame(cross_rows).to_csv(OUT / "cross_axis_mitigation_summary.csv", index=False)


if __name__ == "__main__":
    main()

"""Summarize NHANES OOF predictions into unweighted paper metrics and diagnostics."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from common.paths import PROJECT_ROOT, run_results_dir

ROOT = PROJECT_ROOT
OUT = run_results_dir("nhanes")
KEYS = ["task", "seed", "model", "method", "target_axis"]
AUDIT_AXES = ["sex", "race_ethnicity", "age_group", "sex_x_race_ethnicity",
              "race_ethnicity_x_age_group", "sex_x_age_group",
              "sex_x_race_ethnicity_x_age_group"]
SUPPORT_CUTOFFS = [10, 20, 30, 50]
# The article and archived primary summaries use 30; 50 is a sensitivity cutoff.
PRIMARY_SUPPORT_CUTOFF = 30


def calc(y, p, score, w):
    y, p, score, w = map(np.asarray, (y, p, score, w))
    pos, neg = y == 1, y == 0
    pos_w, neg_w = w[pos].sum(), w[neg].sum()
    tp, fp = np.sum(w[pos] * p[pos]), np.sum(w[neg] * p[neg])
    tpr, fpr = tp / pos_w, fp / neg_w
    return {
        "accuracy": (tp + neg_w - fp) / w.sum(),
        "balanced_accuracy": (tpr + 1 - fpr) / 2,
        "tpr": tpr, "fpr": fpr, "specificity": 1 - fpr,
        "selection_rate": np.sum(w * p) / w.sum(),
        "ppv": tp / (tp + fp) if tp + fp else np.nan,
        "roc_auc": roc_auc_score(y, score, sample_weight=w),
        "average_precision": average_precision_score(y, score, sample_weight=w),
        "brier": np.average((y - score) ** 2, weights=w),
        "n": len(y), "positive_n": int(pos.sum()), "negative_n": int(neg.sum()),
    }


def summarize_condition(frame):
    y = frame.y.to_numpy(int); p = frame.p_positive.to_numpy(float); score = frame.score.to_numpy(float)
    w = frame.WTMECPRP.to_numpy(float)
    result = {k: frame.iloc[0][k] for k in KEYS}
    for suffix, wt in (("w", w), ("unw", np.ones(len(y)))):
        for metric, value in calc(y, p, score, wt).items():
            result[f"{metric}_{suffix}"] = value
    result["eo_fallback_n"] = int(frame.condition_status.astype(str).str.startswith("not_estimable").sum())
    return result


def group_rows(frame):
    y = frame.y.to_numpy(int); p = frame.p_positive.to_numpy(float); w = frame.WTMECPRP.to_numpy(float)
    out = []
    fields = {k: frame.iloc[0][k] for k in KEYS}
    for axis in AUDIT_AXES:
        groups = frame[axis].astype(str).to_numpy()
        for group in sorted(np.unique(groups)):
            ix = groups == group; pos = ix & (y == 1); neg = ix & (y == 0)
            gp, gn = int(pos.sum()), int(neg.sum())
            row = {**fields, "audit_axis": axis, "group": group, "n": int(ix.sum()),
                   "positive_n": gp, "negative_n": gn,
                   "expected_positive_n": float(p[ix].sum()),
                   "positive_weight": float(w[pos].sum()), "negative_weight": float(w[neg].sum()),
                   "substantive_group": not any(x in group.lower() for x in ["unknown", "missing"])}
            for suffix, wt in (("w", w), ("unw", np.ones(len(y)))):
                wp, wn = wt[pos].sum(), wt[neg].sum()
                tp = np.sum(wt[pos] * p[pos]) if gp else np.nan
                fp = np.sum(wt[neg] * p[neg]) if gn else np.nan
                pp = tp + fp if np.isfinite(tp) and np.isfinite(fp) else np.nan
                row[f"tpr_{suffix}"] = tp / wp if wp else np.nan
                row[f"fpr_{suffix}"] = fp / wn if wn else np.nan
                row[f"selection_rate_{suffix}"] = np.sum(wt[ix] * p[ix]) / wt[ix].sum() if wt[ix].sum() else np.nan
                row[f"ppv_{suffix}"] = tp / pp if pp and pp > 0 else np.nan
            out.append(row)
    return out


def gap_rows(groups, cutoff):
    keys = KEYS + ["audit_axis"]
    result = []
    for key, frame in groups.groupby(keys, dropna=False, sort=False):
        base = dict(zip(keys, key))
        eligible = frame[(frame.positive_n >= cutoff) & (frame.negative_n >= cutoff)]
        for scope, selected in (("primary_substantive", eligible[eligible.substantive_group]),
                                ("including_unknown_missing", eligible)):
            row = {**base, "gap_scope": scope, "n_groups": int(selected.group.nunique()),
                   "support_threshold_each_class": cutoff}
            for suffix in ("w", "unw"):
                for metric in ("tpr", "fpr", "selection_rate", "ppv"):
                    vals = selected[f"{metric}_{suffix}"].dropna().to_numpy()
                    row[f"{metric}_gap_{suffix}"] = float(vals.max() - vals.min()) if len(vals) >= 2 else np.nan
                t, f = row[f"tpr_gap_{suffix}"], row[f"fpr_gap_{suffix}"]
                row[f"average_odds_gap_{suffix}"] = (t + f) / 2 if np.isfinite(t) and np.isfinite(f) else np.nan
            result.append(row)
    return result


def mean_range(frame, keys, values):
    agg = frame.groupby(keys, dropna=False)[values].agg(["mean", "min", "max"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    return agg.reset_index()


def main(output_dir: Path | None = None, predictions_filename: str = "oof_predictions_internal.csv.gz"):
    OUT = Path(output_dir) if output_dir is not None else run_results_dir("nhanes")
    path = OUT / predictions_filename
    if not path.exists():
        raise FileNotFoundError(path)
    pred = pd.read_csv(path)
    performances, all_groups = [], []
    for i, (_, frame) in enumerate(pred.groupby(KEYS, sort=False), start=1):
        performances.append(summarize_condition(frame))
        all_groups.extend(group_rows(frame))
        if i % 20 == 0:
            print(f"Summarized {i} conditions")
    performance = pd.DataFrame(performances)
    groups = pd.DataFrame(all_groups)
    performance.to_csv(OUT / "performance_by_seed.csv", index=False)
    groups.to_csv(OUT / "group_metrics_by_seed.csv", index=False)
    fair = pd.DataFrame(gap_rows(groups, PRIMARY_SUPPORT_CUTOFF))
    fair.to_csv(OUT / "fairness_gaps_by_seed.csv", index=False)

    perf_cols = [c for c in performance.columns if c.endswith("_w") or c.endswith("_unw")]
    mean_range(performance, ["task", "model", "method", "target_axis"], perf_cols).to_csv(
        OUT / "performance_summary.csv", index=False)
    fair_cols = [c for c in fair.columns if c.endswith("_w") or c.endswith("_unw")]
    mean_range(fair, ["task", "model", "method", "target_axis", "audit_axis", "gap_scope"],
               fair_cols).to_csv(OUT / "fairness_gap_summary.csv", index=False)

    sensitivity = pd.concat([pd.DataFrame(gap_rows(groups, cut)) for cut in SUPPORT_CUTOFFS], ignore_index=True)
    sensitivity.to_csv(OUT / "support_sensitivity_by_seed.csv", index=False)
    sens_primary = sensitivity[sensitivity.gap_scope.eq("primary_substantive")]
    mean_range(sens_primary, ["support_threshold_each_class", "task", "model", "method",
               "target_axis", "audit_axis"], ["average_odds_gap_unw", "tpr_gap_unw", "fpr_gap_unw",
               "n_groups"]).to_csv(OUT / "support_sensitivity_summary.csv", index=False)

    baseline = fair[(fair.method == "baseline") & (fair.gap_scope == "primary_substantive")]
    base_gap = baseline.set_index(["task", "seed", "model", "audit_axis"])
    base_perf = performance[performance.method == "baseline"].set_index(["task", "seed", "model"])
    targeted, cross_axis = [], []
    for method in ("reweighing", "equalized_odds_postprocessing"):
        for target in ["sex", "race_ethnicity", "age_group", "sex_x_race_ethnicity",
                       "sex_x_race_ethnicity_x_age_group"]:
            for task in performance.task.unique():
                for model in performance.model.unique():
                    tg = fair[(fair.task == task) & (fair.method == method) &
                              (fair.target_axis == target) & (fair.audit_axis == target) &
                              (fair.gap_scope == "primary_substantive")].set_index("seed")
                    tp = performance[(performance.task == task) & (performance.method == method) &
                                     (performance.target_axis == target) & (performance.model == model)].set_index("seed")
                    # Target gap is summarized across all models below; performance is model-specific.
                    if len(tg) and len(tp):
                        pass
                    for seed in sorted(performance.seed.unique()):
                        gkey = (task, seed, model, target)
                        bkey = (task, seed, model)
                        try:
                            g = fair[(fair.task == task) & (fair.seed == seed) & (fair.model == model) &
                                     (fair.method == method) & (fair.target_axis == target) &
                                     (fair.audit_axis == target) & (fair.gap_scope == "primary_substantive")].iloc[0]
                            bg = base_gap.loc[gkey, "average_odds_gap_unw"]
                            p = performance[(performance.task == task) & (performance.seed == seed) &
                                            (performance.model == model) & (performance.method == method) &
                                            (performance.target_axis == target)].iloc[0]
                            bp = base_perf.loc[bkey]
                            targeted.append({"task": task, "seed": seed, "model": model,
                                "method": method, "target_axis": target,
                                "baseline_target_AOD": bg, "mitigated_target_AOD": g.average_odds_gap_unw,
                                "delta_target_AOD": g.average_odds_gap_unw - bg,
                                "baseline_balanced_accuracy": bp.balanced_accuracy_unw,
                                "mitigated_balanced_accuracy": p.balanced_accuracy_unw,
                                "delta_balanced_accuracy": p.balanced_accuracy_unw - bp.balanced_accuracy_unw,
                                "delta_auc": p.roc_auc_unw - bp.roc_auc_unw})
                        except (KeyError, IndexError):
                            continue
                    for audit in AUDIT_AXES:
                        for seed in sorted(performance.seed.unique()):
                            try:
                                g = fair[(fair.task == task) & (fair.seed == seed) & (fair.model == model) &
                                         (fair.method == method) & (fair.target_axis == target) &
                                         (fair.audit_axis == audit) & (fair.gap_scope == "primary_substantive")].iloc[0]
                                bg = base_gap.loc[(task, seed, model, audit), "average_odds_gap_unw"]
                                cross_axis.append({"task": task, "seed": seed, "model": model,
                                    "method": method, "target_axis": target, "audit_axis": audit,
                                    "baseline_AOD": bg, "mitigated_AOD": g.average_odds_gap_unw,
                                    "delta_AOD": g.average_odds_gap_unw - bg})
                            except (KeyError, IndexError):
                                continue
    pd.DataFrame(targeted).to_csv(OUT / "targeted_mitigation_by_seed.csv", index=False)
    if targeted:
        mean_range(pd.DataFrame(targeted), ["task", "model", "method", "target_axis"],
                   ["baseline_target_AOD", "mitigated_target_AOD", "delta_target_AOD",
                    "baseline_balanced_accuracy", "mitigated_balanced_accuracy",
                    "delta_balanced_accuracy", "delta_auc"]).to_csv(
                        OUT / "targeted_mitigation_summary.csv", index=False)
    pd.DataFrame(cross_axis).to_csv(OUT / "cross_axis_mitigation_by_seed.csv", index=False)
    if cross_axis:
        mean_range(pd.DataFrame(cross_axis), ["task", "model", "method", "target_axis", "audit_axis"],
                   ["baseline_AOD", "mitigated_AOD", "delta_AOD"]).to_csv(
                       OUT / "cross_axis_mitigation_summary.csv", index=False)
    print(f"Wrote {len(performance)} condition summaries and {len(groups)} group rows in {OUT}")


if __name__ == "__main__":
    main()

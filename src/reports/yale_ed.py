"""Aggregate the three Yale ED visit-level split runs."""
from pathlib import Path
import pandas as pd

from common.paths import PROJECT_ROOT, run_results_dir


ROOT = PROJECT_ROOT
OUT = run_results_dir("yale_ed")


def main():
    parts = [OUT / "core", OUT / "repeat_seeds"]
    overall = pd.concat([pd.read_csv(p / "overall_by_split.csv") for p in parts], ignore_index=True)
    gaps = pd.concat([pd.read_csv(p / "fairness_gaps_by_split.csv") for p in parts], ignore_index=True)
    overall.to_csv(OUT / "overall_all_seeds.csv", index=False)
    gaps.to_csv(OUT / "gaps_all_seeds.csv", index=False)
    overall.to_csv(OUT / "overall_by_split.csv", index=False)
    gaps.to_csv(OUT / "fairness_gaps_by_split.csv", index=False)
    pd.concat([pd.read_csv(p / "group_metrics_by_split.csv") for p in parts], ignore_index=True).to_csv(
        OUT / "group_metrics_by_split.csv", index=False
    )

    baseline = overall[overall.method == "baseline"]
    performance = baseline.groupby("model").agg(
        seed_n=("seed", "nunique"), auc_mean=("roc_auc_score", "mean"),
        auc_min=("roc_auc_score", "min"), auc_max=("roc_auc_score", "max"),
        ap_mean=("average_precision_score", "mean"),
        ba_mean=("balanced_accuracy_expected", "mean"),
        ba_min=("balanced_accuracy_expected", "min"),
        ba_max=("balanced_accuracy_expected", "max"),
    ).reset_index()
    performance.to_csv(OUT / "baseline_performance_summary.csv", index=False)

    base_gaps = gaps[gaps.method == "baseline"].groupby(["model", "audit_axis"]).agg(
        tpr_gap_mean=("tpr_gap", "mean"), tpr_gap_min=("tpr_gap", "min"),
        tpr_gap_max=("tpr_gap", "max"), fpr_gap_mean=("fpr_gap", "mean"),
        fpr_gap_min=("fpr_gap", "min"), fpr_gap_max=("fpr_gap", "max"),
        adequate_groups_min=("groups_in_gap_n", "min"),
        low_support_groups_max=("groups_low_support_n", "max"),
    ).reset_index()
    base_gaps.to_csv(OUT / "baseline_gap_summary.csv", index=False)

    records = []
    for row in overall[overall.method.isin(["reweighing", "threshold_optimizer"])].itertuples():
        own = gaps[(gaps.seed == row.seed) & (gaps.model == row.model)
                   & (gaps.method == row.method) & (gaps.target_axis == row.target_axis)
                   & (gaps.audit_axis == row.target_axis)]
        ref = gaps[(gaps.seed == row.seed) & (gaps.model == row.model)
                   & (gaps.method == "baseline") & (gaps.target_axis == "none")
                   & (gaps.audit_axis == row.target_axis)]
        base_model = baseline[(baseline.seed == row.seed) & (baseline.model == row.model)]
        if len(own) != 1 or len(ref) != 1 or len(base_model) != 1:
            continue
        own, ref, base_model = own.iloc[0], ref.iloc[0], base_model.iloc[0]
        records.append({
            "seed": row.seed, "model": row.model, "method": row.method,
            "target_axis": row.target_axis, "tpr_gap": own.tpr_gap,
            "tpr_gap_baseline": ref.tpr_gap, "delta_tpr_gap": own.tpr_gap-ref.tpr_gap,
            "fpr_gap": own.fpr_gap, "fpr_gap_baseline": ref.fpr_gap,
            "delta_fpr_gap": own.fpr_gap-ref.fpr_gap,
            "balanced_accuracy": row.balanced_accuracy_expected,
            "balanced_accuracy_baseline": base_model.balanced_accuracy_expected,
            "delta_balanced_accuracy": row.balanced_accuracy_expected-base_model.balanced_accuracy_expected,
        })
    impacts = pd.DataFrame(records)
    impacts.to_csv(OUT / "targeted_mitigation_impacts_all_seeds.csv", index=False)
    impact_summary = impacts.groupby(["model", "method", "target_axis"]).agg(
        seed_n=("seed", "nunique"), tpr_gap_mean=("tpr_gap", "mean"),
        delta_tpr_gap_mean=("delta_tpr_gap", "mean"),
        delta_tpr_gap_min=("delta_tpr_gap", "min"),
        delta_tpr_gap_max=("delta_tpr_gap", "max"),
        fpr_gap_mean=("fpr_gap", "mean"), delta_fpr_gap_mean=("delta_fpr_gap", "mean"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        delta_balanced_accuracy_mean=("delta_balanced_accuracy", "mean"),
    ).reset_index()
    impact_summary.to_csv(OUT / "targeted_mitigation_impact_summary.csv", index=False)

    # Keep this index derived from the actual outputs of the current run.
    lines = [
        "# Yale ED experiment results", "",
        "This directory contains visit-level results for the three study seeds. "
        "The public data do not provide patient identifiers.", "",
        "The audit includes age and race × gender × age, and both are mitigation targets. "
        "The primary intersectional comparison uses race × gender × age.", "",
        "## Baseline performance", "",
        "| Classifier | ROC-AUC mean (range) | Balanced accuracy mean (range) |",
        "|---|---:|---:|",
    ]
    for r in performance.itertuples():
        lines.append(
            f"| {r.model} | {r.auc_mean:.3f} ({r.auc_min:.3f}–{r.auc_max:.3f}) "
            f"| {r.ba_mean:.3f} ({r.ba_min:.3f}–{r.ba_max:.3f}) |"
        )
    lines += [
        "", "## Files", "",
        "- `core/` and `repeat_seeds/`: performance, group metrics, gaps, support and logs by evaluation.",
        "- `overall_all_seeds.csv` and `gaps_all_seeds.csv`: concatenated results.",
        "- `baseline_performance_summary.csv` and `baseline_gap_summary.csv`: baseline summaries.",
        "- `targeted_mitigation_impacts_all_seeds.csv` and `targeted_mitigation_impact_summary.csv`: mitigation comparisons.",
        "- `paired_effects_summary.csv`, `paired_effects_by_split.csv`, and `paired_group_effects_by_split.csv`: cross-axis and group-level comparisons.",
        "", "The repeated evaluations are descriptive and are not independent samples.", "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Aggregated {overall.seed.nunique()} seeds; wrote results summaries to {OUT}")
    from reports.yale_ed_diagnostics import main as diagnostics
    import sys
    previous = sys.argv
    try:
        sys.argv = ["reports.yale_ed_diagnostics", "--run-dir", str(OUT)]
        diagnostics()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()

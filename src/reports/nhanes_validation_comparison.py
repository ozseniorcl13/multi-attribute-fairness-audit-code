"""Compare the two NHANES validation designs from their per-seed outputs."""

from pathlib import Path

import pandas as pd

from common.paths import RUN_RESULTS_ROOT, run_results_dir

TASKS = {
    "hba1c_high": "HbA1c",
    "measured_bp_high": "Blood pressure",
    "obesity_bmi30": "BMI",
}
DESIGNS = {
    "nhanes_single_split": "repeated_single_split_60_20_20",
    "nhanes": "five_fold_cv_combined_out_of_fold",
}
AXIS = "sex_x_race_ethnicity_x_age_group"


def build_comparison() -> pd.DataFrame:
    rows = []
    for folder, design in DESIGNS.items():
        root = run_results_dir(folder)
        performance = pd.read_csv(root / "performance_by_seed.csv")
        gaps = pd.read_csv(root / "fairness_gaps_by_seed.csv")
        performance = performance[
            performance.method.eq("baseline") & performance.target_axis.eq("none")
        ]
        gaps = gaps[
            gaps.method.eq("baseline") & gaps.target_axis.eq("none")
            & gaps.audit_axis.eq(AXIS) & gaps.gap_scope.eq("primary_substantive")
            & gaps.support_threshold_each_class.eq(30)
        ]
        for task, label in TASKS.items():
            perf = performance[performance.task.eq(task)]
            group = gaps[gaps.task.eq(task)]
            if len(perf) != 9 or len(group) != 9:
                raise ValueError(f"Expected 3 models × 3 seeds for {folder}/{task}")
            if perf.duplicated(["seed", "model"]).any() or group.duplicated(["seed", "model"]).any():
                raise ValueError(f"Duplicate evaluation for {folder}/{task}")
            rows.append({
                "task": label,
                "design": design,
                "mean_roc_auc": perf.roc_auc_unw.mean(),
                "mean_balanced_accuracy": perf.balanced_accuracy_unw.mean(),
                "mean_supported_three_way_groups_min30": group.n_groups.mean(),
                "mean_three_way_aod_min30": group.average_odds_gap_unw.mean(),
            })
    comparison = pd.DataFrame(rows)
    return comparison.sort_values(["task", "design"], kind="stable").reset_index(drop=True)


def main() -> None:
    output = RUN_RESULTS_ROOT / "nhanes_validation_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    comparison = build_comparison()
    comparison.to_csv(output, index=False)
    supplement = RUN_RESULTS_ROOT.parent / "supplement" / "nhanes_split_comparison.csv"
    supplement.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(supplement, index=False)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()

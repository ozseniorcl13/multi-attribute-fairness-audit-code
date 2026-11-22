# Yale ED experiment results

This directory contains visit-level results for the three study seeds. The public data do not provide patient identifiers.

The audit includes age and race × gender × age, and both are mitigation targets. The primary intersectional comparison uses race × gender × age.

## Baseline performance

| Classifier | ROC-AUC mean (range) | Balanced accuracy mean (range) |
|---|---:|---:|
| logistic_regression | 0.856 (0.856–0.857) | 0.769 (0.769–0.770) |
| random_forest | 0.857 (0.857–0.857) | 0.771 (0.770–0.772) |
| xgboost | 0.855 (0.854–0.855) | 0.768 (0.768–0.769) |

## Files

- `core/` and `repeat_seeds/`: performance, group metrics, gaps, support and logs by evaluation.
- `overall_all_seeds.csv` and `gaps_all_seeds.csv`: concatenated results.
- `baseline_performance_summary.csv` and `baseline_gap_summary.csv`: baseline summaries.
- `targeted_mitigation_impacts_all_seeds.csv` and `targeted_mitigation_impact_summary.csv`: mitigation comparisons.
- `paired_effects_summary.csv`, `paired_effects_by_split.csv`, and `paired_group_effects_by_split.csv`: cross-axis and group-level comparisons.

The repeated evaluations are descriptive and are not independent samples.

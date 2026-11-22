"""Check the recovered sensitivity results against the published comparison."""

import unittest

import numpy as np
import pandas as pd

from common.paths import PROJECT_ROOT
from experiments.nhanes_single_split import stratified_three_way
from reports.nhanes_validation_comparison import build_comparison


class ValidationComparisonTest(unittest.TestCase):
    def test_splits_are_reproducible_disjoint_and_stratified(self):
        y = np.tile(np.array([0, 0, 0, 1]), 250)
        train, validation, test = stratified_three_way(y, 20261027)
        self.assertEqual([len(train), len(validation), len(test)], [600, 200, 200])
        self.assertEqual(len(set(train) & set(validation)), 0)
        self.assertEqual(len(set(train) & set(test)), 0)
        self.assertEqual(len(set(validation) & set(test)), 0)
        self.assertTrue(all(y[part].mean() == 0.25 for part in (train, validation, test)))
        repeat = stratified_three_way(y, 20261027)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip((train, validation, test), repeat)))

    def test_comparison_matches_manuscript_at_reported_precision(self):
        actual = build_comparison().set_index(["task", "design"]).sort_index()
        published = pd.read_csv(PROJECT_ROOT / "results/supplement/nhanes_split_comparison.csv")
        published = published.set_index(["task", "design"]).sort_index()
        pd.testing.assert_index_equal(actual.index, published.index)
        for column, precision in (
            ("mean_roc_auc", 3),
            ("mean_balanced_accuracy", 3),
            ("mean_supported_three_way_groups_min30", 1),
            ("mean_three_way_aod_min30", 3),
        ):
            actual_values = pd.to_numeric(actual[column], errors="coerce").round(precision)
            published_values = pd.to_numeric(published[column], errors="coerce").round(precision)
            np.testing.assert_allclose(actual_values, published_values, rtol=0, atol=0, equal_nan=True)

    def test_hba1c_audit_and_mitigation_estimability_are_distinct(self):
        root = PROJECT_ROOT / "results/runs/nhanes_single_split"
        conditions = pd.read_csv(root / "fold_conditions.csv")
        axis = "sex_x_race_ethnicity_x_age_group"
        fit = conditions[
            conditions.task.eq("hba1c_high")
            & conditions.method.eq("equalized_odds_postprocessing")
            & conditions.target_axis.eq(axis)
        ]
        self.assertEqual(len(fit), 9)
        self.assertTrue(fit.status.eq("estimated").all())
        gaps = pd.read_csv(root / "fairness_gaps_by_seed.csv")
        audit = gaps[
            gaps.task.eq("hba1c_high") & gaps.method.eq("baseline")
            & gaps.target_axis.eq("none") & gaps.audit_axis.eq(axis)
            & gaps.gap_scope.eq("primary_substantive")
        ]
        self.assertEqual(len(audit), 9)
        self.assertTrue(audit.average_odds_gap_unw.isna().all())


if __name__ == "__main__":
    unittest.main()

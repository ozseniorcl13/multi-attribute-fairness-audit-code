# NHANES single-split sensitivity run

These outputs were regenerated with the recovered 60/20/20 implementation,
not copied from the original run. The original code changes were recovered from
the execution history; the original per-record predictions were not found.

The run uses seeds 20261027, 20261028 and 20261029, three classifiers and the
same NHANES tasks and mitigation methods as the primary five-fold experiment.
It contains 477,576 held-out prediction rows. The resulting task-level means
match `results/supplement/nhanes_split_comparison.csv` at the precision reported
in the manuscript. The full-precision comparison is available in
`results/runs/nhanes_validation_comparison.csv`.

For HbA1c, the three-way test audit with a minimum of 30 cases per class has
fewer than two eligible groups in every evaluation, so its AOD is not
computable. This does not mean that the mitigation fit failed: in this
regenerated run, `fold_conditions.csv` records the three-way equalized-odds
fit as `estimated` in all nine evaluations, after grouping low-support cells.
These are different support checks, and this observation should be checked
against the manuscript's description of mitigation estimability.

Recreate this run and rebuild the comparison from the repository root:

```bash
uv run health-fairness run nhanes-single-split
uv run python -m reports.nhanes_validation_comparison
uv run python -m unittest discover -s tests -v
```

The `run` command writes to this directory and replaces the regenerated outputs.
The primary five-fold NHANES outputs are stored separately in `runs/nhanes/`.

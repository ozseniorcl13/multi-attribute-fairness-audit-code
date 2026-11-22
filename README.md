# Multi-Attribute Fairness Audit

Code and results for *Beyond Single-Attribute Audits: How Group Definitions,
Mitigation Targets and Data Coverage Shape Fairness Reports in Machine Learning
for Health*.

The study evaluates fairness auditing and mitigation across four health datasets,
six prediction tasks, and three classifiers: logistic regression, random forest,
and XGBoost. The experimental conditions include baseline training, reweighing,
and equalized-odds post-processing.

## Project structure

```text
.
├── data/           # Source datasets and derived NHANES cohort
├── notebooks/      # Analysis tables and figures
├── results/        # Experiment runs and supplementary tables
├── scripts/        # Dataset download, checks, and supplementary analyses
├── src/
│   ├── common/     # Shared paths, model definitions, and seeds
│   ├── datasets/   # NHANES cohort preparation
│   ├── experiments/# Dataset-specific experiments
│   └── reports/    # Result summaries and notebook inputs
├── pyproject.toml
└── uv.lock
```

## Environment preparation

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). On Linux, open a terminal;
on Windows, use Ubuntu/WSL. The commands below are run from the repository root.

```bash
uv sync --locked --extra analysis
```

## Datasets

| Dataset | Location in this repository | Original dataset |
|---|---|---|
| Yale ED | `data/raw/yale_ed/5v_cleandf.RData` | [Yale EMMLC](https://github.com/yaleemmlc/admissionprediction/tree/master/Results) |
| BRFSS 2023 | `data/raw/brfss_2023/LLCP2023.XPT` (download required) | [CDC BRFSS 2023](https://www.cdc.gov/brfss/annual_data/annual_2023.html) |
| Diabetes 130-US Hospitals | `data/raw/diabetes_130/diabetic_data.csv` | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008) |
| NHANES 2017–March 2020 | `data/raw/nhanes/` | [CDC NHANES](https://wwwn.cdc.gov/nchs/nhanes/search/DataPage.aspx?Cycle=2017-2020) |

To download BRFSS and verify the input files, run:

```bash
uv run python scripts/download_brfss_2023.py
uv run python scripts/verify_raw_data.py --all
```

## Results in the short manuscript

The experiment results and analysis inputs are available in `results/`. The
notebook reads the stored inputs, so reviewing the results does not require
rerunning model training or downloading BRFSS.

| Result in the manuscript | Where to find it |
|---|---|
| Baseline performance table | `results/runs/paper_inputs/baseline_performance_by_classifier.csv` |
| Mitigation comparison table | `results/runs/paper_inputs/mitigation_by_classifier.csv` and `mitigation_paired.csv` |
| Per-run performance | `results/runs/paper_inputs/performance_by_run.csv` |
| Figure 2: audit scope | `figures/fig2_audit_scope.png`; data: `results/runs/paper_inputs/audit_by_condition.csv` |
| Table 6: intersectional AOD by EO target, all tasks | `results/runs/paper_inputs/audit_by_condition.csv` (method `EO`, axis `Intersection`); BRFSS detail in `figures/fig3_target.png` |
| Selected group examples | `results/runs/paper_inputs/selected_group_examples.csv` |
| NHANES HbA1c audit on fixed participants (sex versus three-way groups, one evaluation) | `results/runs/paper_inputs/nhanes_matched_population_groups.csv` |
| NHANES validation-design comparison | `results/runs/nhanes_validation_comparison.csv`; reported values: `results/supplement/nhanes_split_comparison.csv` |
| Figure 3: support and coverage | `figures/fig6_support.png`; data: `results/runs/paper_inputs/audit_by_condition.csv` |
| Figure 4: positive cases per NHANES HbA1c intersectional group | `figures/fig7_hba1c_support.png`; data: `data/processed/nhanes/survey_data.csv` |
| Changes across supported groups | `results/supplement/group_rates_baseline_vs_eo.csv` and `group_change_summary.csv` |

The notebook presents the four result tables in the order used by the short
manuscript: predictive performance, mitigation, selected group examples, and
the NHANES validation-design comparison. The validation comparison uses the
per-seed outputs from both NHANES designs. The single-split implementation was
recovered from the original execution history and is included in the complete
reproduction command.

To inspect the tables and regenerate the analysis figures:

```bash
uv run --extra analysis jupyter lab notebooks/01_final_results_and_figures.ipynb
```

Run all cells in the notebook. Generated figures and their supporting data are
saved in `figures/`. The notebook generates the result images listed above.
The study-design figure in the manuscript is maintained separately from this
repository.

Individual experiment outputs, including performance, group metrics, fairness
gaps, support, and checkpoints, are in `results/runs/<dataset>/`. NHANES
out-of-fold predictions are in `results/runs/nhanes/`.

## Survey weights and Fairlearn

For BRFSS and NHANES, survey weights are used when fitting the baseline and
reweighed classifiers. Reweighing combines the sampling weights with its
group--outcome factors. Global validation thresholds and Fairlearn's
`ThresholdOptimizer` are fitted without sample weights. Paper-facing
performance and fairness metrics use unweighted held-out records, as they do
for the other datasets. Dataset definitions, splits, features, group
definitions, and support rules are unchanged.

## Support rules for fitting mitigation

These rules apply only when fitting reweighing and equalized-odds
post-processing. They are separate from the test-set rule that decides which
groups enter the audits (at least 100 positive and 100 negative test cases; 30
of each for NHANES). Groups merged during fitting share the same weights or
decision rule, but each original group is still audited separately. Counts are
unweighted records in each outcome class.

**Reweighing.** In Yale ED, BRFSS and Diabetes 130-US, a training group keeps
its own group-outcome factor when it has at least 100 positive and 100
negative cases; in NHANES, the minimum is 30 of each. Smaller groups are merged
before the factors are calculated. In BRFSS and NHANES, the factors are then
multiplied by the sampling weights.

**Equalized-odds post-processing.**

| Dataset | A group receives its own rule when it has | Otherwise |
|---|---|---|
| Yale ED | at least 100 positive and 100 negative training cases and at least one of each in validation | small training groups are merged; groups without both validation classes or unseen in validation use the global baseline threshold |
| BRFSS 2023 | at least 100 positive and 100 negative training cases and at least one positive and one negative validation case | it is merged into a low-support group; if that group still lacks one of the classes, its test records use the global validation threshold, which is also used for all test records when fewer than two groups remain |
| Diabetes 130-US | at least 100 positive and 100 negative training cases and at least 20 of each in validation | it is merged; the merged group receives a rule only if it has at least 20 of each class in validation, otherwise its test records keep the baseline decisions; when fewer than two groups remain, EO is not estimable for that fit |
| NHANES | at least 30 positive and 30 negative training cases and at least 10 of each in validation | it is merged; if the merged group has fewer than 10 of each class in validation, its test records use the baseline threshold, which is also used for the whole test fold when fewer than two groups remain |

The thresholds are defined at the top of each script in `src/experiments/`, and
the merging logic is in `src/common/fairness.py`.

## Run the experiments

To run an individual dataset:

```bash
uv run health-fairness run yale-ed
uv run health-fairness run brfss-2023
uv run health-fairness run diabetes-130
uv run health-fairness run nhanes
```

For Yale ED, the standard audit includes age and race × gender × age; both are
also mitigation targets. The results include cross-axis comparisons and paired
group-level rates. Merged groups share a rule; validation counts repeated on
their constituent training-group rows refer to the same pooled group.

To run the four main dataset pipelines, the NHANES single-split sensitivity
analysis, and assemble the paper and supplementary inputs:

```bash
uv run health-fairness run all
```

The NHANES commands prepare their cohort automatically. BRFSS must be
downloaded before running its experiment or `run all`.

Existing BRFSS and Diabetes 130 checkpoints are reused by default, so `run all`
on this checkout does not retrain every model. To rerun the main experiments from
raw data, use a separate copy with an empty `results/runs/`. Experiment and
summary commands write to that directory and can overwrite stored results.

To run only the NHANES single-split sensitivity experiment:

```bash
uv run health-fairness run nhanes-single-split
```

The single-split command writes to `results/runs/nhanes_single_split/` and does
not replace the five-fold outputs. Run `uv run health-fairness summarize all`
after both designs finish to rebuild the validation comparison, the seven
paper-facing CSV inputs, and the group-change supplement.

To rebuild the summaries and notebook inputs from completed runs:

```bash
uv run health-fairness summarize all
```

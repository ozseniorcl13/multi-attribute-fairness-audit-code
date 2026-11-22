# Versioned experiment results

This directory contains the aggregate outputs used to support the manuscript.
The canonical run set covers Yale ED, BRFSS 2023, Diabetes 130-US Hospitals,
and NHANES 2017--March 2020. `runs/paper_inputs/` contains the seven compact
CSV files assembled from the experiment outputs. The notebook consumes the
six analysis inputs needed for its tables and figures.

The run outputs were consolidated from the experiment archive used to
prepare the manuscript. They include per-seed or per-split performance,
group-level rates, fairness gaps, mitigation summaries, support sensitivity,
split/fold metadata, and checkpoint summaries. The run manifests and split
metadata should be retained with the corresponding dataset directory.

The NHANES `oof_predictions_internal.csv.gz` file and the run logs are also
versioned. The OOF file contains the row-level predictions used to rebuild the
unweighted NHANES summaries and matched-population comparison. Logs provide the
execution trace for the BRFSS, Diabetes 130-US Hospitals, and NHANES runs.
The derived NHANES cohort in `data/processed/nhanes/` is versioned alongside
the source inputs because it is required to rebuild the summaries from the
stored predictions. No BRFSS raw data is stored here: its official download
remains reproducible through the documented download script.

`supplement/nhanes_split_comparison.csv` preserves the reported comparison of
single-split and cross-validation results. The recovered single-split
implementation was rerun from the NHANES data; regenerated per-run outputs and
predictions are in `runs/nhanes_single_split/`. The comparison computed from
both designs is stored in `runs/nhanes_validation_comparison.csv`.

`supplement/` contains the paired group-level EO analysis, its summary, and the
NHANES validation-design comparison. To inspect the results and reproduce the
figures, set up the environment and open the notebook:

```bash
cd /path/to/multi-attribute-fairness-audit-code
uv sync --extra analysis
uv run --extra analysis jupyter lab notebooks/01_final_results_and_figures.ipynb
```

"""Rebuild group-level EO comparisons from the final experiment outputs.

The script pairs baseline and intersection-targeted equalized-odds rates for
the same supported test groups, using unweighted held-out records throughout.
It writes the paired rates and their descriptive summary under
``results/supplement/``. It does not read or reuse exploratory optimizer runs.

Run from the repository root:
    uv run python scripts/summarize_eo_group_changes.py
"""
from __future__ import annotations

import csv
import statistics as stats
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "runs"
OUT = ROOT / "results" / "supplement"

SPECS = {
    "yale": {
        "files": ["yale_ed/core/group_metrics_by_split.csv",
                  "yale_ed/repeat_seeds/group_metrics_by_split.csv"],
        "dataset": "yale", "axis": "race_x_gender_x_age_band", "eo_method": "threshold_optimizer",
        "cutoff": 100, "rate_suffix": "_expected", "use_substantive": False,
    },
    "brfss": {
        "files": ["brfss_2023/group_metrics_by_split.csv"],
        "dataset": "brfss", "axis": "race_x_sex_x_age",
        "eo_method": "equalized_odds_postprocessing", "cutoff": 100,
        "rate_suffix": "_unw", "use_substantive": True,
    },
    "diabetes130": {
        "files": ["diabetes_130/group_metrics_by_split.csv"],
        "dataset": "diabetes130", "axis": "race_x_gender_x_age",
        "eo_method": "equalized_odds_postprocessing", "cutoff": 100,
        "rate_suffix": "", "use_substantive": True,
    },
    "nhanes_hba1c": {
        "files": ["nhanes/group_metrics_by_seed.csv"], "task": "hba1c_high",
        "dataset": "nhanes_hba1c", "axis": "sex_x_race_ethnicity_x_age_group",
        "eo_method": "equalized_odds_postprocessing", "cutoff": 30,
        "rate_suffix": "_unw", "use_substantive": True,
    },
    "nhanes_bp": {
        "files": ["nhanes/group_metrics_by_seed.csv"], "task": "measured_bp_high",
        "dataset": "nhanes_bp", "axis": "sex_x_race_ethnicity_x_age_group",
        "eo_method": "equalized_odds_postprocessing", "cutoff": 30,
        "rate_suffix": "_unw", "use_substantive": True,
    },
    "nhanes_obesity": {
        "files": ["nhanes/group_metrics_by_seed.csv"], "task": "obesity_bmi30",
        "dataset": "nhanes_obesity", "axis": "sex_x_race_ethnicity_x_age_group",
        "eo_method": "equalized_odds_postprocessing", "cutoff": 30,
        "rate_suffix": "_unw", "use_substantive": True,
    },
}


def _load(spec: dict) -> pd.DataFrame:
    frames = [pd.read_csv(RUNS / relative) for relative in spec["files"]]
    data = pd.concat(frames, ignore_index=True)
    if "task" in spec:
        data = data[data.task.eq(spec["task"])]
    data = data[data.audit_axis.eq(spec["axis"])]
    if spec["use_substantive"] and "substantive_group" in data:
        data = data[data.substantive_group.fillna(False).astype(bool)]
    data = data[(data.positive_n >= spec["cutoff"]) & (data.negative_n >= spec["cutoff"])]
    if "support_status" in data:
        data = data[data.support_status.eq("adequate")]
    if "adequate_support" in data:
        data = data[data.adequate_support.fillna(False).astype(bool)]
    return data


def _paired_dataset(name: str, spec: dict) -> pd.DataFrame:
    data = _load(spec)
    base = data[(data.method == "baseline") & (data.target_axis == "none")].copy()
    eo = data[(data.method == spec["eo_method"]) & (data.target_axis == spec["axis"])].copy()
    suffix = spec["rate_suffix"]
    columns = [f"tpr{suffix}", f"fpr{suffix}"]
    for col in columns:
        if col not in data:
            raise KeyError(f"Missing unweighted rate {col!r} in {name} output")
    keys = ["seed", "model", "group"]
    if base.duplicated(keys).any() or eo.duplicated(keys).any():
        raise ValueError(f"Duplicate group rows in baseline/EO outputs for {name}")
    left = base[keys + ["positive_n", "negative_n", *columns]].rename(columns={
        "positive_n": "n_positive", "negative_n": "n_negative",
        columns[0]: "tpr_baseline", columns[1]: "fpr_baseline",
    })
    right = eo[keys + columns].rename(columns={columns[0]: "tpr_eo", columns[1]: "fpr_eo"})
    paired = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if len(paired) != len(left):
        raise ValueError(f"Unpaired baseline groups for {name}: {len(left)} baseline vs {len(paired)} pairs")
    paired.insert(0, "dataset", name)
    return paired


def _summarize(group_rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    for task, frame in group_rows.groupby("dataset", sort=False):
        values = list(zip(frame.tpr_baseline, frame.tpr_eo, frame.fpr_baseline, frame.fpr_eo))
        n = len(values)
        if n == 0:
            raise ValueError(f"No paired supported groups found for {task}")
        ranked = sorted(values, key=lambda value: value[0])
        k = max(1, n // 3)
        output.append({
            "task": task,
            "group_evaluation_pairs": n,
            "share_tpr_down": sum(te < tb for tb, te, _, _ in values) / n,
            "share_fpr_up": sum(fe > fb for _, _, fb, fe in values) / n,
            "share_worse_both": sum(te < tb and fe > fb for tb, te, fb, fe in values) / n,
            "share_better_both": sum(te > tb and fe < fb for tb, te, fb, fe in values) / n,
            "mean_dtpr_top_third_baseline_tpr": stats.mean(te - tb for tb, te, _, _ in ranked[-k:]),
            "mean_dtpr_bottom_third_baseline_tpr": stats.mean(te - tb for tb, te, _, _ in ranked[:k]),
        })
    return pd.DataFrame(output)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    groups = pd.concat([_paired_dataset(name, spec) for name, spec in SPECS.items()], ignore_index=True)
    expected_pairs = 1071
    if len(groups) != expected_pairs:
        raise ValueError(f"Expected {expected_pairs} supported pairs, found {len(groups)}")
    groups = groups.sort_values(["dataset", "seed", "model", "group"]).reset_index(drop=True)
    groups.to_csv(OUT / "group_rates_baseline_vs_eo.csv", index=False, float_format="%.6f")
    summary = _summarize(groups)
    summary.to_csv(OUT / "group_change_summary.csv", index=False, float_format="%.6f")
    print(f"Wrote {len(groups)} matched group-evaluation pairs across {groups.dataset.nunique()} tasks")
    print(f"Wrote paired group results and {len(summary)} task summaries to {OUT}")


if __name__ == "__main__":
    main()

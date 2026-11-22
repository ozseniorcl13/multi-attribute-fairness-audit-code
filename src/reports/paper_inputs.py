"""Assemble the compact tables used by the final analysis notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import DEFAULT_SEEDS
from common.paths import PROJECT_ROOT, RUN_RESULTS_ROOT


PAPER_INPUT_NAMES = (
    "audit_by_condition.csv",
    "baseline_performance_by_classifier.csv",
    "mitigation_by_classifier.csv",
    "mitigation_paired.csv",
    "performance_by_run.csv",
    "nhanes_matched_population_groups.csv",
    "selected_group_examples.csv",
)

SPECS = {
    "yale": {
        "run_name": "yale_ed",
        "main_axis": "race_x_gender_x_age_band",
        "main_target": "race_x_gender_x_age_band",
        "primary_support": 100,
        "support_cutoffs": [100],
        "audit_axes": {"gender": "Sex/gender", "race": "Race/ethnicity",
                       "age_band": "Age", "race_x_gender_x_age_band": "Intersection"},
        "target_labels": {"gender": "Sex/gender", "race": "Race/ethnicity",
                          "age_band": "Age", "race_x_gender": "race_x_gender",
                          "race_x_gender_x_age_band": "Intersection"},
        "methods": {"baseline": "Baseline", "reweighing": "Reweighing",
                    "threshold_optimizer": "EO"},
        "rate_suffix": "_expected",
        "performance_columns": ("roc_auc_score", "balanced_accuracy_expected"),
        "task": "",
    },
    "brfss": {
        "run_name": "brfss_2023",
        "main_axis": "race_x_sex_x_age",
        "main_target": "race_x_sex_x_age",
        "primary_support": 100,
        "support_cutoffs": [100],
        "audit_axes": {"sex": "Sex/gender", "race_ethnicity": "Race/ethnicity",
                       "age_band": "Age", "race_x_sex_x_age": "Intersection"},
        "target_labels": {"sex": "Sex/gender", "race_ethnicity": "Race/ethnicity",
                          "age_band": "Age", "race_x_sex_x_age": "Intersection"},
        "methods": {"baseline": "Baseline", "reweighing": "Reweighing",
                    "equalized_odds_postprocessing": "EO"},
        "rate_suffix": "_unw",
        "performance_columns": ("roc_auc_unw", "balanced_accuracy_unw"),
        "task": "",
    },
    "diabetes130": {
        "run_name": "diabetes_130",
        "main_axis": "race_x_gender_x_age",
        "main_target": "race_x_gender_x_age",
        "primary_support": 100,
        "support_cutoffs": [20, 100, 200],
        "audit_axes": {"gender": "Sex/gender", "race": "Race/ethnicity",
                       "age": "Age", "race_x_gender_x_age": "Intersection"},
        "target_labels": {"gender": "Sex/gender", "race": "Race/ethnicity",
                          "age": "Age", "race_x_gender": "race_x_gender",
                          "race_x_gender_x_age": "Intersection"},
        "methods": {"baseline": "Baseline", "reweighing": "Reweighing",
                    "equalized_odds_postprocessing": "EO"},
        "rate_suffix": "",
        "performance_columns": ("roc_auc", "balanced_accuracy"),
        "task": "",
    },
    "nhanes": {
        "run_name": "nhanes",
        "main_axis": "sex_x_race_ethnicity_x_age_group",
        "main_target": "sex_x_race_ethnicity_x_age_group",
        "primary_support": 30,
        "support_cutoffs": [10, 20, 30, 50],
        "audit_axes": {"sex": "Sex/gender", "race_ethnicity": "Race/ethnicity",
                       "age_group": "Age", "sex_x_race_ethnicity_x_age_group": "Intersection"},
        "target_labels": {"sex": "Sex/gender", "race_ethnicity": "Race/ethnicity",
                          "age_group": "Age", "sex_x_race_ethnicity_x_age_group": "Intersection"},
        "methods": {"baseline": "Baseline", "reweighing": "Reweighing",
                    "equalized_odds_postprocessing": "EO"},
        "rate_suffix": "_unw",
        "performance_columns": ("roc_auc_unw", "balanced_accuracy_unw"),
        "task": "",
    },
}

SEED_COLUMN = "seed"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required result file is missing: {path}")
    return pd.read_csv(path)


def load_dataset_outputs(source_root: Path, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = SPECS[name]
    folder = source_root / spec["run_name"]

    if name == "yale":
        overall = _read_csv(folder / "overall_all_seeds.csv")
        groups = pd.concat(
            [_read_csv(folder / "core/group_metrics_by_split.csv"),
             _read_csv(folder / "repeat_seeds/group_metrics_by_split.csv")],
            ignore_index=True,
        )
        return overall, groups

    perf_name = "performance_by_seed.csv" if name == "nhanes" else "performance_by_split.csv"
    group_name = "group_metrics_by_seed.csv" if name == "nhanes" else "group_metrics_by_split.csv"
    return _read_csv(folder / perf_name), _read_csv(folder / group_name)


def _task_name(name: str, source_task: str | None) -> str:
    if name != "nhanes":
        return name
    return {
        "hba1c_high": "nhanes_hba1c",
        "measured_bp_high": "nhanes_bp",
        "obesity_bmi30": "nhanes_obesity",
    }[str(source_task)]


def _task_slice(frame: pd.DataFrame, name: str, task: str | None) -> pd.DataFrame:
    if name == "nhanes":
        return frame[frame.task.eq(task)].copy()
    return frame.copy()


def _rate_column(frame: pd.DataFrame, metric: str, spec: dict) -> str:
    suffix = spec["rate_suffix"]
    if metric == "tpr":
        options = [f"tpr{suffix}", "tpr"]
    elif metric == "fpr":
        options = [f"fpr{suffix}", "fpr"]
    elif metric == "selection_rate":
        options = [f"selection_rate{suffix}", "selection_rate"]
    else:
        raise ValueError(metric)
    for column in options:
        if column in frame:
            return column
    raise KeyError(f"No {metric} column found for this dataset: {options}")


def _is_substantive(frame: pd.DataFrame) -> pd.Series:
    if "substantive_group" in frame:
        return frame.substantive_group.fillna(False).astype(bool)
    label = frame.group.astype(str).str.lower()
    return ~label.str.contains("unknown|missing|refused|not stated|not recorded", regex=True)


def _supported_rows(frame: pd.DataFrame, cutoff: int, substantive_only: bool = True) -> pd.DataFrame:
    selected = frame[(frame.positive_n >= cutoff) & (frame.negative_n >= cutoff)].copy()
    if "support_status" in selected:
        selected = selected[selected.support_status.eq("adequate")]
    return selected[_is_substantive(selected)] if substantive_only else selected


def build_audit_table(outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    rows: list[dict] = []
    for name, (performance, groups) in outputs.items():
        spec = SPECS[name]
        task_values = sorted(groups.task.dropna().unique()) if name == "nhanes" else [None]
        audit_keys = set(spec["audit_axes"])
        for source_task in task_values:
            task_groups = _task_slice(groups, name, source_task)
            dataset = _task_name(name, source_task)
            for cutoff in spec["support_cutoffs"]:
                # The manuscript Yale fairness tables treat refusal/not-stated as a
                # reported race/ethnicity category; retain it for exact paper
                # reproduction. Other datasets mark non-substantive groups.
                supported = _supported_rows(
                    task_groups, cutoff, substantive_only=name != "yale"
                )
                supported = supported[supported.audit_axis.isin(audit_keys)]
                keys = ["seed", "model", "method", "target_axis", "audit_axis"]
                for key, cell in supported.groupby(keys, dropna=False, sort=False):
                    seed, model, method, target, axis = key
                    tpr = cell[_rate_column(cell, "tpr", spec)].dropna().to_numpy(float)
                    fpr = cell[_rate_column(cell, "fpr", spec)].dropna().to_numpy(float)
                    if len(tpr) < 2 or len(fpr) < 2:
                        continue
                    target_label = spec["target_labels"].get(str(target), str(target))
                    rows.append({
                        "dataset": dataset,
                        "seed": int(seed),
                        "model": model,
                        "method": spec["methods"].get(str(method), str(method)),
                        "target": target_label,
                        "axis": spec["audit_axes"][str(axis)],
                        "minimum": int(cutoff),
                        "groups": int(cell.group.nunique()),
                        "tpr_gap": float(tpr.max() - tpr.min()),
                        "fpr_gap": float(fpr.max() - fpr.min()),
                        "aod": float(((tpr.max() - tpr.min()) + (fpr.max() - fpr.min())) / 2),
                        "primary": bool(cutoff == spec["primary_support"]),
                    })
    result = pd.DataFrame(rows)
    return result.sort_values(["dataset", "seed", "model", "method", "target", "axis", "minimum"]).reset_index(drop=True)


def build_performance_table(outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for name, (performance, _) in outputs.items():
        spec = SPECS[name]
        tasks = sorted(performance.task.dropna().unique()) if name == "nhanes" else [None]
        auc_col, ba_col = spec["performance_columns"]
        for source_task in tasks:
            data = _task_slice(performance, name, source_task)
            data = data[data.method.eq("baseline")]
            if name != "yale":
                data = data[data.target_axis.eq("none")]
            grouped = data.groupby("model", dropna=False)[[auc_col, ba_col]].mean()
            dataset = _task_name(name, source_task)
            for model, row in grouped.iterrows():
                rows.append({"dataset": dataset, "model": model,
                             "auc": float(row[auc_col]), "ba": float(row[ba_col])})
    return pd.DataFrame(rows).sort_values(["dataset", "model"]).reset_index(drop=True)


def build_performance_by_run(outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    """Collect unweighted held-out performance for the three paper conditions."""
    rows = []
    for name, (performance, _) in outputs.items():
        spec = SPECS[name]
        tasks = sorted(performance.task.dropna().unique()) if name == "nhanes" else [None]
        auc_col, ba_col = spec["performance_columns"]
        for source_task in tasks:
            frame = _task_slice(performance, name, source_task)
            for method_key, method_label in spec["methods"].items():
                target_axis = "none" if method_key == "baseline" else spec["main_target"]
                selected = frame[(frame.method == method_key) &
                                 (frame.target_axis == target_axis)]
                for _, record in selected.iterrows():
                    n_col = "n_unw" if "n_unw" in record.index else "n"
                    rows.append({
                        "dataset": _task_name(name, source_task),
                        "seed": int(record[SEED_COLUMN]),
                        "model": record["model"],
                        "method": method_label,
                        "roc_auc": float(record[auc_col]),
                        "balanced_accuracy": float(record[ba_col]),
                        "test_records": int(record[n_col]) if pd.notna(record[n_col]) else None,
                    })
    result = pd.DataFrame(rows)
    return result.sort_values(["dataset", "seed", "model", "method"]).reset_index(drop=True)


def _performance_lookup(outputs, name: str, source_task: str | None) -> pd.DataFrame:
    spec = SPECS[name]
    performance, _ = outputs[name]
    data = _task_slice(performance, name, source_task)
    auc_col, ba_col = spec["performance_columns"]
    if name == "yale":
        # The primary baseline is stored under target_axis=none.
        pass
    data = data.rename(columns={auc_col: "auc", ba_col: "ba"})
    return data


def build_mitigation_table(audit: pd.DataFrame, outputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired_rows = []
    for name, (performance, _) in outputs.items():
        spec = SPECS[name]
        tasks = sorted(performance.task.dropna().unique()) if name == "nhanes" else [None]
        for source_task in tasks:
            dataset = _task_name(name, source_task)
            task_audit = audit[audit.dataset.eq(dataset) & audit.primary & audit.axis.eq("Intersection")]
            perf = _performance_lookup(outputs, name, source_task)
            baseline_perf = perf[(perf.method == "baseline") & (perf.target_axis == "none")]
            baseline_gap = task_audit[(task_audit.method == "Baseline") & (task_audit.target == "none")]
            for method in ("Reweighing", "EO"):
                method_rows = task_audit[(task_audit.method == method) &
                                          (task_audit.target == "Intersection")]
                for _, gap in method_rows.iterrows():
                    model, seed = gap.model, gap.seed
                    base_gap = baseline_gap[(baseline_gap.model == model) & (baseline_gap.seed == seed)]
                    if base_gap.empty:
                        continue
                    source_method = {v: k for k, v in spec["methods"].items()}[method]
                    mitigated_perf = perf[(perf.seed == seed) & (perf.model == model) &
                                          (perf.method == source_method) &
                                          (perf.target_axis == spec["main_target"])]
                    base_perf = baseline_perf[(baseline_perf.seed == seed) &
                                              (baseline_perf.model == model)]
                    if mitigated_perf.empty or base_perf.empty:
                        continue
                    base_gap_row = base_gap.iloc[0]
                    mp, bp = mitigated_perf.iloc[0], base_perf.iloc[0]
                    paired_rows.append({
                        **{c: gap[c] for c in ["dataset", "seed", "model", "method", "target", "axis", "minimum", "groups", "tpr_gap", "fpr_gap", "aod", "primary"]},
                        "baseline_aod": float(base_gap_row.aod),
                        "ba": float(mp.ba), "baseline_ba": float(bp.ba),
                        "delta_aod": float(gap.aod - base_gap_row.aod),
                        "delta_ba": float(mp.ba - bp.ba),
                    })
    paired = pd.DataFrame(paired_rows).sort_values(["dataset", "seed", "model", "method"]).reset_index(drop=True)
    summary = paired.groupby(["dataset", "model", "method"], as_index=False).agg(
        baseline_aod=("baseline_aod", "mean"), aod=("aod", "mean"), delta_ba=("delta_ba", "mean"))
    return paired, summary


def build_selected_groups(outputs) -> pd.DataFrame:
    rows = []
    for name, (performance, groups) in outputs.items():
        spec = SPECS[name]
        tasks = sorted(groups.task.dropna().unique()) if name == "nhanes" else [None]
        for source_task in tasks:
            data = _task_slice(groups, name, source_task)
            data = data[(data.seed == DEFAULT_SEEDS[0]) &
                        (data.model == "logistic_regression") &
                        (data.audit_axis == spec["main_axis"])]
            base = data[(data.method == "baseline") & (data.target_axis == "none")]
            eo_method = next(method for method, label in spec["methods"].items() if label == "EO")
            eo = data[(data.method == eo_method) & (data.target_axis == spec["main_target"])]
            cutoff = spec["primary_support"]
            base, eo = (_supported_rows(base, cutoff, substantive_only=False),
                        _supported_rows(eo, cutoff, substantive_only=False))
            if name == "yale":
                rate_cols = {m: f"{m}_expected" for m in ["tpr", "fpr", "selection_rate"]}
            else:
                rate_cols = {m: _rate_column(base, m, spec) for m in ["tpr", "fpr", "selection_rate"]}
            if name != "yale":
                eo_rate_cols = {m: _rate_column(eo, m, spec) for m in rate_cols}
            else:
                eo_rate_cols = rate_cols
            keys = ["group"]
            merged = base.merge(eo, on=keys, suffixes=("_baseline", "_eo"), how="inner")
            dataset = _task_name(name, source_task)
            for _, row in merged.iterrows():
                rows.append({
                    "dataset": dataset, "seed": int(DEFAULT_SEEDS[0]), "model": "logistic_regression",
                    "group": row.group,
                    "n": int(row["n_baseline"] if "n_baseline" in row else row["n"]),
                    "n_positive": int(row["positive_n_baseline"] if "positive_n_baseline" in row else row["positive_n"]),
                    "n_negative": int(row["negative_n_baseline"] if "negative_n_baseline" in row else row["negative_n"]),
                    "tpr_baseline": float(row[f"{rate_cols['tpr']}_baseline"]),
                    "fpr_baseline": float(row[f"{rate_cols['fpr']}_baseline"]),
                    "selection_rate_baseline": float(row[f"{rate_cols['selection_rate']}_baseline"]),
                    "tpr_eo": float(row[f"{eo_rate_cols['tpr']}_eo"]),
                    "fpr_eo": float(row[f"{eo_rate_cols['fpr']}_eo"]),
                    "selection_rate_eo": float(row[f"{eo_rate_cols['selection_rate']}_eo"]),
                })
    return pd.DataFrame(rows).sort_values(["dataset", "group"]).reset_index(drop=True)


def _nhanes_matched_population(source_root: Path, output_dir: Path) -> None:
    predictions_path = source_root / "nhanes" / "oof_predictions_internal.csv.gz"
    if not predictions_path.is_file():
        raise FileNotFoundError("NHANES OOF predictions are missing")

    target = "sex_x_race_ethnicity_x_age_group"
    # Matched-population comparison: retain the supported three-way groups,
    # then recompute sex-only and three-way rates on precisely those records.
    usecols = ["task", "seed", "model", "y", "sex", target, "method", "target_axis", "p_positive"]
    full = pd.read_csv(predictions_path, usecols=usecols)
    condition = (full.task.eq("hba1c_high") & full.seed.eq(DEFAULT_SEEDS[0]) &
                 full.model.eq("logistic_regression") & full.method.eq("baseline") &
                 full.target_axis.eq("none"))
    base = full[condition].copy()
    intersections = base.groupby(target).agg(
        positive_n=("y", "sum"), negative_n=("y", lambda y: int((y == 0).sum())))
    eligible = intersections[(intersections.positive_n >= 30) & (intersections.negative_n >= 30)].index
    base = base[base[target].isin(eligible)]
    matched_rows = []
    for axis in ("sex", target):
        for group, part in base.groupby(axis, sort=True):
            y = part.y.to_numpy(int); p = part.p_positive.to_numpy(float)
            pos, neg = y == 1, y == 0
            matched_rows.append({"axis": axis, "group": group, "n": len(part),
                "positive_n": int(pos.sum()), "negative_n": int(neg.sum()),
                "tpr": float(np.mean(p[pos])),
                "fpr": float(np.mean(p[neg]))})
    pd.DataFrame(matched_rows).to_csv(output_dir / "nhanes_matched_population_groups.csv", index=False)


def assemble_paper_inputs(output_dir: Path | None = None) -> Path:
    source_root = RUN_RESULTS_ROOT
    if output_dir is None:
        output_dir = RUN_RESULTS_ROOT / "paper_inputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {name: load_dataset_outputs(source_root, name) for name in SPECS}

    for name, (performance, groups) in outputs.items():
        for label, frame in (("performance", performance), ("group", groups)):
            observed = set(pd.to_numeric(frame[SEED_COLUMN], errors="coerce").dropna().astype(int).unique())
            missing = set(DEFAULT_SEEDS) - observed
            if missing:
                raise ValueError(f"{label} outputs for {name} are missing paper seeds: {sorted(missing)}")

    audit = build_audit_table(outputs)
    baseline = build_performance_table(outputs)
    performance_by_run = build_performance_by_run(outputs)
    paired, mitigation = build_mitigation_table(audit, outputs)
    selected = build_selected_groups(outputs)
    audit.to_csv(output_dir / "audit_by_condition.csv", index=False)
    baseline.to_csv(output_dir / "baseline_performance_by_classifier.csv", index=False)
    mitigation.to_csv(output_dir / "mitigation_by_classifier.csv", index=False)
    paired.to_csv(output_dir / "mitigation_paired.csv", index=False)
    performance_by_run.to_csv(output_dir / "performance_by_run.csv", index=False)
    selected.to_csv(output_dir / "selected_group_examples.csv", index=False)
    _nhanes_matched_population(source_root, output_dir)

    expected = {"baseline_performance_by_classifier.csv": 18,
                "mitigation_by_classifier.csv": 36,
                "mitigation_paired.csv": 108,
                "performance_by_run.csv": 162}
    for filename, count in expected.items():
        actual = len(pd.read_csv(output_dir / filename))
        if actual != count:
            raise RuntimeError(f"{filename} has {actual} rows; the manuscript expects {count}")
    if output_dir == RUN_RESULTS_ROOT / "paper_inputs":
        from reports.seed_dispersion import main as summarize_seed_dispersion
        summarize_seed_dispersion()
    print(f"Built final analysis inputs from experiment runs at {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble final analysis tables and figure inputs from completed runs.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    assemble_paper_inputs(args.output_dir)


if __name__ == "__main__":
    main()

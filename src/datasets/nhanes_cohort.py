"""Build the NHANES pre-pandemic MEC analysis file and three adult markers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common.paths import raw_data_dir, PROCESSED_DATA_ROOT


RAW_FILES = {
    "demographics": "P_DEMO.xpt",
    "hba1c": "P_GHB.xpt",
    "body": "P_BMX.xpt",
    "blood_pressure": "P_BPXO.xpt",
}


def read_xpt(path: Path) -> pd.DataFrame:
    return pd.read_sas(path, format="xport", encoding="utf-8")


def weighted_mean(values: pd.Series, weights: pd.Series) -> float | None:
    ok = values.notna() & weights.notna() & weights.gt(0)
    if not ok.any():
        return None
    return float((values.loc[ok].astype(float) * weights.loc[ok]).sum() / weights.loc[ok].sum())


def build_survey_file(raw_dir: Path) -> pd.DataFrame:
    demo = read_xpt(raw_dir / RAW_FILES["demographics"])
    ghb = read_xpt(raw_dir / RAW_FILES["hba1c"])
    body = read_xpt(raw_dir / RAW_FILES["body"])
    bp = read_xpt(raw_dir / RAW_FILES["blood_pressure"])

    demo_cols = [
        "SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "DMDEDUC2",
        "INDFMPIR", "WTMECPRP", "SDMVSTRA", "SDMVPSU", "RIDSTATR",
    ]
    frame = demo[demo_cols].merge(
        ghb[["SEQN", "LBXGH"]], on="SEQN", how="left", validate="one_to_one"
    )
    frame = frame.merge(
        body[["SEQN", "BMXBMI"]], on="SEQN", how="left", validate="one_to_one"
    )
    bp_cols = ["BPXOSY1", "BPXOSY2", "BPXOSY3", "BPXODI1", "BPXODI2", "BPXODI3"]
    frame = frame.merge(
        bp[["SEQN", *bp_cols]], on="SEQN", how="left", validate="one_to_one"
    )

    # The MEC weight defines the eligible examination sample. Keep every record
    # with a positive MEC weight so design-based domain analyses retain the PSU/stratum structure.
    frame = frame.loc[frame["WTMECPRP"].gt(0) & frame["RIDSTATR"].eq(2)].copy()
    systolic = [c for c in bp_cols if c.startswith("BPXOSY")]
    diastolic = [c for c in bp_cols if c.startswith("BPXODI")]
    frame["mean_systolic"] = frame[systolic].mean(axis=1)
    frame["mean_diastolic"] = frame[diastolic].mean(axis=1)

    frame["adult_domain"] = frame["RIDAGEYR"].ge(20).astype("int8")
    frame["hba1c_high"] = frame["LBXGH"].ge(6.5).where(frame["LBXGH"].notna()).astype("Int64")
    bp_observed = frame["mean_systolic"].notna() & frame["mean_diastolic"].notna()
    frame["measured_bp_high"] = (
        (frame["mean_systolic"].ge(130) | frame["mean_diastolic"].ge(80))
        .where(bp_observed)
        .astype("Int64")
    )
    frame["obesity_bmi30"] = frame["BMXBMI"].ge(30).where(frame["BMXBMI"].notna()).astype("Int64")
    for target in ["hba1c_high", "measured_bp_high", "obesity_bmi30"]:
        frame[f"eligible_{target}"] = (frame["adult_domain"].eq(1) & frame[target].notna()).astype("int8")
    frame["sex"] = frame["RIAGENDR"].map({1: "Male", 2: "Female"}).fillna("Unknown")
    frame["race_ethnicity"] = frame["RIDRETH3"].map({
        1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
        4: "Non-Hispanic Black", 6: "Non-Hispanic Asian", 7: "Other/Multi-racial",
    }).fillna("Unknown")
    frame["age_group"] = pd.cut(
        frame["RIDAGEYR"], bins=[19, 39, 59, float("inf")],
        labels=["20-39", "40-59", "60+"], right=True,
    ).astype("string").fillna("Under 20")
    frame["education"] = frame["DMDEDUC2"].replace({7: pd.NA, 9: pd.NA}).astype("string")
    frame["sex_x_race_ethnicity"] = frame["sex"].astype(str) + " | " + frame["race_ethnicity"].astype(str)
    frame["race_ethnicity_x_age_group"] = frame["race_ethnicity"].astype(str) + " | " + frame["age_group"].astype(str)
    frame["sex_x_race_ethnicity_x_age_group"] = (
        frame["sex"].astype(str) + " | " + frame["race_ethnicity"].astype(str) + " | " + frame["age_group"].astype(str)
    )
    return frame.reset_index(drop=True)


def main() -> None:
    base = PROCESSED_DATA_ROOT / "nhanes"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=raw_data_dir("nhanes"))
    parser.add_argument("--output", type=Path, default=base / "survey_data.csv")
    args = parser.parse_args()
    survey_data = build_survey_file(args.raw_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    survey_data.to_csv(args.output, index=False)

    adults = survey_data["adult_domain"].eq(1)
    task_vars = ["hba1c_high", "measured_bp_high", "obesity_bmi30"]
    task_summary = {}
    for target in task_vars:
        task_rows = adults & survey_data[target].notna()
        y, w = survey_data.loc[task_rows, target], survey_data.loc[task_rows, "WTMECPRP"]
        task_summary[target] = {
            "n_adults_with_observed_target": int(task_rows.sum()),
            "positive_n_unweighted": int(y.sum()),
            "prevalence_unweighted": float(y.mean()),
            "prevalence_weighted_point_estimate": weighted_mean(y, w),
            "missing_predictors_percent": {
                c: float(survey_data.loc[task_rows, c].isna().mean() * 100)
                for c in ["RIDAGEYR", "BMXBMI", "mean_systolic", "mean_diastolic", "LBXGH", "INDFMPIR", "education"]
            },
        }
    summary = {
        "dataset": "NHANES 2017-March 2020 pre-pandemic",
        "mec_records_with_positive_weight": int(len(survey_data)),
        "adult_domain_n": int(adults.sum()),
        "n_strata": int(survey_data["SDMVSTRA"].nunique()),
        "n_psu_nested_in_strata": int(survey_data[["SDMVSTRA", "SDMVPSU"]].drop_duplicates().shape[0]),
        "tasks": task_summary,
        "notes": [
            "Keep the complete positive-MEC-weight examination sample for design-based domain estimation.",
            "All three targets are measured markers from a single examination, not confirmed clinical diagnoses.",
            "Target-specific eligible adults are selected as survey domains, not by discarding other records before survey design setup.",
        ],
    }
    args.output.with_name("cohort_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Survey analysis file written: {args.output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

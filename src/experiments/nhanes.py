"""Survey-weighted NHANES model fitting with standard Fairlearn post-processing."""
from __future__ import annotations

import gc
import json
import logging
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from common.paths import PROJECT_ROOT, run_results_dir, PROCESSED_DATA_ROOT
from common.config import DEFAULT_SEEDS
from common.models import make_models as _shared_make_models
from common.fairness import (
    LOW_VALIDATION_GROUP, UNCALIBRATED_GROUP, apply_equalized_odds,
    calibration_groups, fit_equalized_odds, map_prediction_groups, youden_threshold,
)

ROOT = PROJECT_ROOT
DATA = PROCESSED_DATA_ROOT / "nhanes" / "survey_data.csv"
OUT = run_results_dir("nhanes")
SEEDS = DEFAULT_SEEDS
TASKS = {
    "hba1c_high": ["BMXBMI", "mean_systolic", "mean_diastolic", "INDFMPIR", "education"],
    "measured_bp_high": ["BMXBMI", "LBXGH", "INDFMPIR", "education"],
    "obesity_bmi30": ["mean_systolic", "mean_diastolic", "LBXGH", "INDFMPIR", "education"],
}
NUMERIC = ["BMXBMI", "mean_systolic", "mean_diastolic", "LBXGH", "INDFMPIR"]
TARGET_AXES = ["sex", "race_ethnicity", "age_group", "sex_x_race_ethnicity",
               "sex_x_race_ethnicity_x_age_group"]
AUDIT_AXES = ["sex", "race_ethnicity", "age_group", "sex_x_race_ethnicity",
              "race_ethnicity_x_age_group", "sex_x_age_group",
              "sex_x_race_ethnicity_x_age_group"]
TRAIN_MIN_CLASS = 30
CAL_MIN_CLASS = 10
LOW = "__LOW_SUPPORT_POOLED__"


def setup_logging() -> logging.Logger:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(OUT / "logs/harmonized.log", mode="a"),
                  logging.StreamHandler()], force=True,
    )
    return logging.getLogger("nhanes_harmonized")


def make_transformer(features: list[str]) -> ColumnTransformer:
    numeric = [c for c in features if c in NUMERIC]
    categorical = [c for c in features if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ]), categorical))
    return ColumnTransformer(transformers, sparse_threshold=0.3)


def make_models(seed: int) -> dict:
    return _shared_make_models(seed)


def train_group_map(a: np.ndarray, y: np.ndarray):
    tab = pd.DataFrame({"a": np.asarray(a, dtype=str), "y": y}).groupby("a").y.agg(
        n="size", positive_n="sum")
    tab["negative_n"] = tab.n - tab.positive_n
    supported = set(tab.index[(tab.positive_n >= TRAIN_MIN_CLASS) &
                              (tab.negative_n >= TRAIN_MIN_CLASS)])
    def collapse(values):
        values = np.asarray(values, dtype=str)
        return np.where(np.isin(values, list(supported)), values, LOW)
    return collapse(a), collapse, tab, supported


def reweighing_weights(a: np.ndarray, y: np.ndarray, survey_weight: np.ndarray):
    frame = pd.DataFrame({"a": np.asarray(a, dtype=str), "y": y,
                          "w": np.asarray(survey_weight, dtype=float)})
    total = frame.w.sum()
    pa = frame.groupby("a").w.sum() / total
    py = frame.groupby("y").w.sum() / total
    pay = frame.groupby(["a", "y"]).w.sum() / total
    factors = np.array([pa[g] * py[k] / pay[g, k] for g, k in zip(frame.a, frame.y)])
    combined = frame.w.to_numpy() * factors
    return combined / combined.mean(), factors


def main():
    log = setup_logging()
    if not DATA.exists():
        raise FileNotFoundError(f"Processed NHANES data missing: {DATA}")
    d = pd.read_csv(DATA)
    d["PSU_NESTED_ID"] = d.SDMVSTRA.astype(str) + "_" + d.SDMVPSU.astype(str)
    required_axes = set(AUDIT_AXES)
    if "sex_x_age_group" not in d:
        d["sex_x_age_group"] = d.sex.astype(str) + " | " + d.age_group.astype(str)
    if not required_axes.issubset(d.columns):
        raise ValueError(f"Missing audit axes: {sorted(required_axes - set(d.columns))}")

    prediction_parts, fit_support_rows, run_rows = [], [], []
    for task, features in TASKS.items():
        eligible = d[d.adult_domain.eq(1) & d[task].notna()].copy().reset_index(drop=True)
        y_all = eligible[task].astype(int).to_numpy()
        if y_all.min() == y_all.max():
            raise ValueError(f"Task {task} has one outcome class")
        log.info("Task %s: n=%d positive=%d weighted prevalence=%.4f", task, len(eligible),
                 int(y_all.sum()), np.average(y_all, weights=eligible.WTMECPRP))
        for seed in SEEDS:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for fold, (dev_ix, test_ix) in enumerate(cv.split(eligible[features], y_all), start=1):
                fit_ix, val_ix = train_test_split(dev_ix, test_size=0.25,
                    random_state=seed + fold * 101, stratify=y_all[dev_ix])
                fit, val, test = (eligible.iloc[idx].copy() for idx in (fit_ix, val_ix, test_ix))
                X_fit, X_val, X_test = fit[features], val[features], test[features]
                y_fit, y_val, y_test = (x[task].astype(int).to_numpy() for x in (fit, val, test))
                w_fit_raw = fit.WTMECPRP.to_numpy(float)
                w_fit = w_fit_raw / w_fit_raw.mean()
                w_val, w_test = val.WTMECPRP.to_numpy(float), test.WTMECPRP.to_numpy(float)
                prep = make_transformer(features)
                X_fit_t = prep.fit_transform(X_fit)
                X_val_t, X_test_t = prep.transform(X_val), prep.transform(X_test)
                for model_name, model in make_models(seed + fold).items():
                    log.info("task=%s seed=%d fold=%d model=%s", task, seed, fold, model_name)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        model.fit(X_fit_t, y_fit, sample_weight=w_fit)
                        for warning in caught:
                            log.warning("%s: %s", warning.category.__name__, warning.message)
                    score_val = model.predict_proba(X_val_t)[:, 1]
                    score_test = model.predict_proba(X_test_t)[:, 1]
                    base_cut = youden_threshold(y_val, score_val)
                    common = {
                        "SEQN": test.SEQN.to_numpy(), "SDMVSTRA": test.SDMVSTRA.to_numpy(),
                        "PSU_NESTED_ID": test.PSU_NESTED_ID.to_numpy(),
                        "WTMECPRP": w_test, "task": task, "seed": seed, "fold": fold,
                        "model": model_name, "y": y_test,
                    }
                    group_cols = {axis: test[axis].astype(str).to_numpy() for axis in AUDIT_AXES}

                    def save_condition(method, target_axis, probs, scores, status="estimated", threshold=np.nan):
                        frame = pd.DataFrame({**common, **group_cols,
                            "method": method, "target_axis": target_axis,
                            "score": np.asarray(scores, dtype=float),
                            "p_positive": np.asarray(probs, dtype=float),
                            "condition_status": status,
                        })
                        prediction_parts.append(frame)
                        run_rows.append({"task": task, "seed": seed, "fold": fold,
                            "model": model_name, "method": method, "target_axis": target_axis,
                            "n_train": len(fit), "n_validation": len(val), "n_test": len(test),
                            "threshold": threshold,
                            "status": status})

                    save_condition("baseline", "none", (score_test >= base_cut).astype(float),
                                   score_test, threshold=base_cut)
                    for axis in TARGET_AXES:
                        a_fit, collapse, support, supported = train_group_map(
                            fit[axis].astype(str).to_numpy(), y_fit)
                        a_val = collapse(val[axis].astype(str).to_numpy())
                        a_test = collapse(test[axis].astype(str).to_numpy())
                        for group, row in support.iterrows():
                            fit_support_rows.append({"task": task, "seed": seed, "fold": fold,
                                "model": model_name, "target_axis": axis, "group": group,
                                "n_train": int(row.n), "positive_train": int(row.positive_n),
                                "negative_train": int(row.negative_n),
                                "used_separately": bool(group in supported)})
                        rw_weights, _ = reweighing_weights(a_fit, y_fit, w_fit_raw)
                        rw_model = make_models(seed + fold)[model_name]
                        rw_model.fit(X_fit_t, y_fit, sample_weight=rw_weights)
                        rw_val = rw_model.predict_proba(X_val_t)[:, 1]
                        rw_test = rw_model.predict_proba(X_test_t)[:, 1]
                        rw_cut = youden_threshold(y_val, rw_val)
                        save_condition("reweighing", axis, (rw_test >= rw_cut).astype(float),
                                       rw_test, threshold=rw_cut)

                        a_val_cal, valid_cal = calibration_groups(a_val, y_val, CAL_MIN_CLASS)
                        status = "estimated"
                        try:
                            keep = np.isin(a_val_cal, list(valid_cal))
                            if len(valid_cal) < 2:
                                raise ValueError("Fewer than two validation groups have both classes")
                            policy = fit_equalized_odds(score_val[keep], y_val[keep], a_val_cal[keep])
                            a_test_cal = map_prediction_groups(a_test, valid_cal)
                            eo_test = apply_equalized_odds(
                                policy, score_test, a_test_cal, valid_cal, base_cut
                            )
                        except Exception as exc:
                            status = "not_estimable_baseline_fallback: " + str(exc)[:160]
                            eo_test = (score_test >= base_cut).astype(float)
                        save_condition("equalized_odds_postprocessing", axis, eo_test, score_test,
                                       status, threshold=base_cut)
                    del model
                    gc.collect()
                log.info("Completed task=%s seed=%d fold=%d", task, seed, fold)
    if not prediction_parts:
        raise RuntimeError("No predictions generated")
    predictions = pd.concat(prediction_parts, ignore_index=True)
    pred_path = OUT / "oof_predictions_internal.csv.gz"
    predictions.to_csv(pred_path, index=False, compression="gzip")
    pd.DataFrame(run_rows).to_csv(OUT / "fold_conditions.csv", index=False)
    pd.DataFrame(fit_support_rows).to_csv(OUT / "training_group_support.csv", index=False)
    manifest = {
        "dataset": "NHANES 2017-March 2020 pre-pandemic",
        "n_predictions": int(len(predictions)), "tasks": list(TASKS),
        "models": ["logistic_regression", "random_forest", "xgboost"],
        "seeds": SEEDS, "outer_folds_per_seed": 5,
        "conditions": ["baseline", "reweighing", "equalized_odds_postprocessing"],
        "sensitive_predictors_excluded": ["sex", "race_ethnicity", "age_group"],
        "sampling_weights": "WTMECPRP used during classifier fitting; thresholds and Fairlearn fitting are unweighted",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                     "scipy": scipy.__version__, "xgboost": xgboost.__version__},
        "row_level_predictions": "internal only; do not distribute",
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Saved %s rows to %s", len(predictions), pred_path)


if __name__ == "__main__":
    main()

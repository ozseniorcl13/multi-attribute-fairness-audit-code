"""Recovered NHANES sensitivity experiment with repeated 60/20/20 splits.

The split and modeling loop were recovered from the original execution; model,
preprocessing, weighting and mitigation helpers are shared with the main run.
"""
from __future__ import annotations

import gc
import json
import logging
import platform
import warnings

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from sklearn.model_selection import train_test_split

from common.paths import PROCESSED_DATA_ROOT, run_results_dir
from common.fairness import (
    apply_equalized_odds, calibration_groups, fit_equalized_odds,
    map_prediction_groups, youden_threshold,
)
from experiments.nhanes import (
    AUDIT_AXES, CAL_MIN_CLASS, SEEDS, TARGET_AXES, TASKS,
    make_models, make_transformer, reweighing_weights, train_group_map,
)

DATA = PROCESSED_DATA_ROOT / "nhanes" / "survey_data.csv"
OUT = run_results_dir("nhanes_single_split")


def setup_logging() -> logging.Logger:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(OUT / "logs/harmonized.log", mode="a"),
                  logging.StreamHandler()], force=True,
    )
    return logging.getLogger("nhanes_harmonized")


def stratified_three_way(y: np.ndarray, seed: int):
    """Return one reproducible 60/20/20 train/validation/test partition."""
    y = np.asarray(y, dtype=int)
    all_ix = np.arange(len(y))
    dev_ix, test_ix = train_test_split(
        all_ix, test_size=0.20, random_state=seed, stratify=y
    )
    train_ix, val_ix = train_test_split(
        dev_ix, test_size=0.25, random_state=seed + 1000, stratify=y[dev_ix]
    )
    return train_ix, val_ix, test_ix


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
            fit_ix, val_ix, test_ix = stratified_three_way(y_all, seed)
            fold = 1
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
    pred_path = OUT / "predictions_internal.csv.gz"
    predictions.to_csv(pred_path, index=False, compression="gzip")
    pd.DataFrame(run_rows).to_csv(OUT / "fold_conditions.csv", index=False)
    pd.DataFrame(fit_support_rows).to_csv(OUT / "training_group_support.csv", index=False)
    manifest = {
        "dataset": "NHANES 2017-March 2020 pre-pandemic",
        "n_predictions": int(len(predictions)), "tasks": list(TASKS),
        "models": ["logistic_regression", "random_forest", "xgboost"],
        "seeds": SEEDS,
        "split_protocol": "three repeated 60/20/20 train/validation/test partitions",
        "conditions": ["baseline", "reweighing", "equalized_odds_postprocessing"],
        "sensitive_predictors_excluded": ["sex", "race_ethnicity", "age_group"],
        "sampling_weights": "WTMECPRP used during classifier fitting; thresholds and Fairlearn fitting are unweighted",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                     "scipy": scipy.__version__, "xgboost": xgboost.__version__},
        "row_level_predictions": "held-out predictions for the single-split sensitivity analysis",
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Saved %s rows to %s", len(predictions), pred_path)


if __name__ == "__main__":
    main()

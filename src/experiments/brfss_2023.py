"""BRFSS 2023 diabetes fairness experiment; writes aggregate outputs only."""
from __future__ import annotations

import gc
import argparse
import json
import logging
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from common.paths import PROJECT_ROOT, run_results_dir, raw_data_dir
from common.config import DEFAULT_SEEDS
from common.models import make_models as _shared_make_models
from common.fairness import (
    apply_equalized_odds, calibration_groups, fit_equalized_odds,
    map_prediction_groups, youden_threshold,
)

ROOT = PROJECT_ROOT
RAW = raw_data_dir("brfss_2023") / "LLCP2023.XPT"
OUT = run_results_dir("brfss_2023")
LOG = OUT / "logs/experiments.log"
SEEDS = DEFAULT_SEEDS
MIN_CELL_SUPPORT = 100
CAL_MIN_CLASS = 1
TARGET_AXES = ["sex", "race_ethnicity", "age_band", "race_x_sex", "race_x_sex_x_age"]
AUDIT_AXES = [
    "race_ethnicity", "sex", "age_band", "race_x_sex", "race_x_age",
    "sex_x_age", "race_x_sex_x_age",
]
CAT = ["GENHLTH", "_RFHYPE6", "_RFCHOL3", "_TOTINDA", "_SMOKER3", "CHECKUP1", "MEDCOST1"]
NUM = ["_BMI5", "PHYSHLTH", "MENTHLTH"]
FEATURES = CAT + NUM
SENSITIVE_RAW = ["_RACEGR3", "SEXVAR", "_AGE80"]
REQUIRED = FEATURES + SENSITIVE_RAW + ["DIABETE4", "_LLCPWT", "_STATE", "_STSTR", "_PSU"]


def logger():
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG, mode="a"), logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger("brfss_experiments")


def read_compact_data(log):
    if not RAW.exists():
        raise FileNotFoundError(f"Expected raw file at {RAW}")
    pieces = []
    for i, chunk in enumerate(pd.read_sas(RAW, format="xport", iterator=True, chunksize=50000), start=1):
        missing = sorted(set(REQUIRED) - set(chunk.columns))
        if missing:
            raise ValueError(f"Required fields absent from XPT: {missing}")
        pieces.append(chunk[REQUIRED].copy())
        if i % 3 == 0:
            log.info("Read %s XPT chunks", i)
    d = pd.concat(pieces, ignore_index=True)
    del pieces
    d = d[d["DIABETE4"].isin([1, 3])].copy().reset_index(drop=True)
    d["y"] = (d["DIABETE4"] == 1).astype("int8")
    d = d[d["_LLCPWT"].notna() & (d["_LLCPWT"] > 0)].copy().reset_index(drop=True)

    race_labels = {
        1.0: "White non-Hispanic", 2.0: "Black non-Hispanic",
        3.0: "Other race non-Hispanic", 4.0: "Multiracial non-Hispanic",
        5.0: "Hispanic", 9.0: "Unknown/refused",
    }
    d["race_ethnicity"] = d["_RACEGR3"].map(race_labels).fillna("Missing race")
    d["sex"] = d["SEXVAR"].map({1.0: "Male", 2.0: "Female"}).fillna("Missing sex")
    age = pd.to_numeric(d["_AGE80"], errors="coerce")
    d["age_band"] = pd.cut(
        age, bins=[-np.inf, 39, 64, np.inf], labels=["18-39", "40-64", "65+"], right=True
    ).astype("object").fillna("Missing age")
    d["race_x_sex"] = d["race_ethnicity"].astype(str) + " × " + d["sex"].astype(str)
    d["race_x_age"] = d["race_ethnicity"].astype(str) + " × " + d["age_band"].astype(str)
    d["sex_x_age"] = d["sex"].astype(str) + " × " + d["age_band"].astype(str)
    d["race_x_sex_x_age"] = (
        d["race_ethnicity"].astype(str) + " × " + d["sex"].astype(str) + " × " + d["age_band"].astype(str)
    )

    for col, invalid in {
        "GENHLTH": [7, 9], "_RFHYPE6": [9], "_RFCHOL3": [9], "_TOTINDA": [9],
        "_SMOKER3": [9], "CHECKUP1": [7, 9], "MEDCOST1": [7, 9],
    }.items():
        d[col] = d[col].mask(d[col].isin(invalid), np.nan).astype("object")
    for col in ["PHYSHLTH", "MENTHLTH"]:
        d[col] = d[col].replace({88: 0, 77: np.nan, 99: np.nan}).astype(float)
    d["_BMI5"] = pd.to_numeric(d["_BMI5"], errors="coerce") / 100.0
    d["_BMI5"] = d["_BMI5"].where(d["_BMI5"].between(10, 100), np.nan)
    # Convert categorical survey codes to strings after invalid-value recoding.
    for col in CAT:
        d[col] = d[col].map(lambda v: str(int(v)) if pd.notna(v) else np.nan)
    return d.reset_index(drop=True)


def split_indices(y, state, seed):
    strata = pd.Series(state.astype(str) + "|y=" + np.asarray(y).astype(str))
    all_idx = np.arange(len(y))
    dev, test = train_test_split(all_idx, test_size=0.20, random_state=seed, stratify=strata)
    train, val = train_test_split(
        dev, test_size=0.25, random_state=seed + 1000, stratify=strata.iloc[dev]
    )
    return train, val, test


def make_transformer():
    return ColumnTransformer(
        [
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]), NUM),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50)),
            ]), CAT),
        ],
        sparse_threshold=0.3,
    )


def make_models(seed):
    return _shared_make_models(seed)


def group_fit_map(a_fit, y_fit):
    frame = pd.DataFrame({"a": np.asarray(a_fit, dtype=str), "y": np.asarray(y_fit, dtype=int)})
    tab = frame.groupby("a", observed=True).y.agg(n="size", positive_n="sum")
    tab["negative_n"] = tab["n"] - tab["positive_n"]
    supported = set(tab.index[(tab.positive_n >= MIN_CELL_SUPPORT) & (tab.negative_n >= MIN_CELL_SUPPORT)])
    if len(supported) < 2:
        raise ValueError("Fewer than two target groups meet the training support rule")
    def collapse(values):
        vals = np.asarray(values, dtype=str)
        return np.where(np.isin(vals, list(supported)), vals, "__LOW_SUPPORT_POOLED__")
    return collapse(a_fit), collapse, tab


def reweighing_weights(a, y, survey_weight):
    a = np.asarray(a, dtype=str)
    y = np.asarray(y, dtype=int)
    sw = np.asarray(survey_weight, dtype=float)
    total = sw.sum()
    tmp = pd.DataFrame({"a": a, "y": y, "w": sw})
    pa = tmp.groupby("a", observed=True).w.sum() / total
    py = tmp.groupby("y", observed=True).w.sum() / total
    pay = tmp.groupby(["a", "y"], observed=True).w.sum() / total
    factors = np.array([pa.loc[g] * py.loc[k] / pay.loc[(g, k)] for g, k in zip(a, y)])
    combined = sw * factors
    return combined / combined.mean(), factors


def calculate_metrics(y, pred_prob, scores, weights):
    y = np.asarray(y, dtype=int)
    p = np.asarray(pred_prob, dtype=float)
    w = np.asarray(weights, dtype=float)
    pos, neg = y == 1, y == 0
    pos_w, neg_w = w[pos].sum(), w[neg].sum()
    tp = float(np.sum(w[pos] * p[pos]))
    fp = float(np.sum(w[neg] * p[neg]))
    fn, tn = pos_w - tp, neg_w - fp
    pred_pos = tp + fp
    tpr, fpr = tp / pos_w, fp / neg_w
    return {
        "accuracy": (tp + tn) / w.sum(),
        "balanced_accuracy": (tpr + 1 - fpr) / 2,
        "tpr": tpr,
        "fpr": fpr,
        "specificity": 1 - fpr,
        "ppv": tp / pred_pos if pred_pos > 0 else np.nan,
        "selection_rate": pred_pos / w.sum(),
        "roc_auc": roc_auc_score(y, scores, sample_weight=w),
        "average_precision": average_precision_score(y, scores, sample_weight=w),
        "positive_weight": pos_w,
        "negative_weight": neg_w,
        "n": len(y),
        "positive_n": int(pos.sum()),
        "negative_n": int(neg.sum()),
    }


def metric_record(y, pred, score, survey_w, base_fields):
    row = dict(base_fields)
    for suffix, weight in [("w", survey_w), ("unw", np.ones(len(y), dtype=float))]:
        metrics = calculate_metrics(y, pred, score, weight)
        for key, value in metrics.items():
            row[f"{key}_{suffix}"] = value
    return row


def is_substantive(group):
    g = str(group)
    return "Unknown/refused" not in g and "Missing" not in g and "LOW_SUPPORT" not in g


def audit_condition(y, pred, survey_w, sens, base_fields, group_rows, gap_rows):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=float)
    sw = np.asarray(survey_w, dtype=float)
    for axis in AUDIT_AXES:
        a = np.asarray(sens[axis], dtype=str)
        rates_by_group = []
        for g in sorted(np.unique(a)):
            ix = a == g
            pos, neg = ix & (y == 1), ix & (y == 0)
            npos, nneg = int(pos.sum()), int(neg.sum())
            adequate = npos >= MIN_CELL_SUPPORT and nneg >= MIN_CELL_SUPPORT
            row = {**base_fields, "audit_axis": axis, "group": g, "n": int(ix.sum()),
                   "positive_n": npos, "negative_n": nneg,
                   "positive_weight": float(sw[pos].sum()), "negative_weight": float(sw[neg].sum()),
                   "adequate_support": adequate, "substantive_group": is_substantive(g)}
            for suffix, w in [("w", sw), ("unw", np.ones(len(y), dtype=float))]:
                wp, wn = w[pos].sum(), w[neg].sum()
                tp = float(np.sum(w[pos] * pred[pos])) if npos else np.nan
                fp = float(np.sum(w[neg] * pred[neg])) if nneg else np.nan
                pp = tp + fp if np.isfinite(tp) and np.isfinite(fp) else np.nan
                row[f"tpr_{suffix}"] = tp / wp if wp > 0 else np.nan
                row[f"fpr_{suffix}"] = fp / wn if wn > 0 else np.nan
                row[f"selection_rate_{suffix}"] = float(np.sum(w[ix] * pred[ix]) / w[ix].sum()) if w[ix].sum() else np.nan
                row[f"ppv_{suffix}"] = tp / pp if pp and pp > 0 else np.nan
            group_rows.append(row)
            if adequate:
                rates_by_group.append(row)
        for sensitivity, include_unknown in [("primary_substantive", False), ("including_unknown_refused", True)]:
            selected_groups = [r for r in rates_by_group if include_unknown or r["substantive_group"]]
            entry = {**base_fields, "audit_axis": axis, "gap_scope": sensitivity,
                     "n_groups": len(selected_groups), "support_threshold_each_class": MIN_CELL_SUPPORT}
            for suffix in ["w", "unw"]:
                for rate in ["tpr", "fpr", "selection_rate", "ppv"]:
                    vals = [r[f"{rate}_{suffix}"] for r in selected_groups if np.isfinite(r[f"{rate}_{suffix}"])]
                    entry[f"{rate}_gap_{suffix}"] = float(max(vals) - min(vals)) if len(vals) >= 2 else np.nan
                entry[f"average_odds_gap_{suffix}"] = (
                    (entry[f"tpr_gap_{suffix}"] + entry[f"fpr_gap_{suffix}"]) / 2
                    if np.isfinite(entry[f"tpr_gap_{suffix}"]) and np.isfinite(entry[f"fpr_gap_{suffix}"])
                    else np.nan
                )
            gap_rows.append(entry)


def gaps_from_group_table(groups):
    """Rebuild one complete weighted/unweighted gap row per audit scope."""
    keys = ["seed", "model", "method", "target_axis", "audit_axis"]
    rows = []
    for key, frame in groups.groupby(keys, dropna=False, sort=False):
        base = dict(zip(keys, key))
        supported = frame[frame.adequate_support.astype(bool)]
        for scope, selected in [
            ("primary_substantive", supported[supported.substantive_group.astype(bool)]),
            ("including_unknown_refused", supported),
        ]:
            entry = {**base, "gap_scope": scope, "n_groups": int(selected.group.nunique()),
                     "support_threshold_each_class": MIN_CELL_SUPPORT}
            for suffix in ["w", "unw"]:
                for metric in ["tpr", "fpr", "selection_rate", "ppv"]:
                    vals = selected[f"{metric}_{suffix}"].dropna().to_numpy()
                    entry[f"{metric}_gap_{suffix}"] = float(vals.max() - vals.min()) if len(vals) >= 2 else np.nan
                entry[f"average_odds_gap_{suffix}"] = (
                    (entry[f"tpr_gap_{suffix}"] + entry[f"fpr_gap_{suffix}"]) / 2
                    if np.isfinite(entry[f"tpr_gap_{suffix}"]) and np.isfinite(entry[f"fpr_gap_{suffix}"])
                    else np.nan
                )
            rows.append(entry)
    return pd.DataFrame(rows)


def evaluate_one(y, pred, score, sw, sens, seed, model, method, target_axis,
                 performance_rows, group_rows, gap_rows):
    fields = {"seed": seed, "model": model, "method": method, "target_axis": target_axis}
    performance_rows.append(metric_record(y, pred, score, sw, fields))
    audit_condition(y, pred, sw, sens, fields, group_rows, gap_rows)


def save_model_checkpoint(seed, model, buckets):
    checkpoint = OUT / "checkpoints" / f"seed_{seed}_{model}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    for key, rows in buckets.items():
        pd.DataFrame(rows).to_csv(checkpoint / f"{key}.csv", index=False)
    (checkpoint / "DONE").write_text("complete\n", encoding="utf-8")


def load_checkpoint_table(name):
    files = sorted((OUT / "checkpoints").glob(f"seed_*_*/{name}.csv"))
    frames = [pd.read_csv(path) for path in files if path.stat().st_size > 1]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true", help="ignore existing model checkpoints and recompute")
    parser.add_argument("--seed", type=int, choices=SEEDS, help="run one split seed")
    parser.add_argument("--model", choices=["logistic_regression", "random_forest", "xgboost"],
                        help="run one classifier")
    args = parser.parse_args()
    seeds_to_run = [args.seed] if args.seed is not None else SEEDS
    models_to_run = ([args.model] if args.model else ["logistic_regression", "random_forest", "xgboost"])
    log = logger()
    start = time.time()
    checkpoint_root = OUT / "checkpoints"
    if args.restart and checkpoint_root.exists():
        import shutil
        shutil.rmtree(checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    log.info("Loading compact BRFSS 2023 task data")
    d = read_compact_data(log)
    log.info("Eligible records=%d; outcome prevalence unweighted=%.4f weighted=%.4f",
             len(d), d.y.mean(), np.average(d.y, weights=d._LLCPWT))
    for seed in seeds_to_run:
        log.info("===== split seed %d =====", seed)
        tr_i, va_i, te_i = split_indices(d.y.to_numpy(), d._STATE.to_numpy(), seed)
        completed_models = {
            model_name for model_name in models_to_run
            if (checkpoint_root / f"seed_{seed}_{model_name}" / "DONE").exists()
        }
        if len(completed_models) == len(models_to_run):
            log.info("seed=%d requested model checkpoints already present; skipping", seed)
            continue
        train, val, test = d.iloc[tr_i].copy(), d.iloc[va_i].copy(), d.iloc[te_i].copy()
        Xtr, Xva, Xte = train[FEATURES], val[FEATURES], test[FEATURES]
        ytr, yva, yte = train.y.to_numpy(), val.y.to_numpy(), test.y.to_numpy()
        wtr_raw, wva, wte = train._LLCPWT.to_numpy(), val._LLCPWT.to_numpy(), test._LLCPWT.to_numpy()
        wtr = wtr_raw / wtr_raw.mean()
        sens_tr = train
        sens_va = val
        sens_te = test
        # The transform is learned once from training records and reused unchanged.
        transform = make_transformer()
        Xtr_t = transform.fit_transform(Xtr)
        Xva_t = transform.transform(Xva)
        Xte_t = transform.transform(Xte)
        log.info("seed=%d transformed shape=%s sparse=%s", seed, Xtr_t.shape, hasattr(Xtr_t, "tocsr"))
        for model_name, model in make_models(seed).items():
            if model_name not in models_to_run:
                continue
            if model_name in completed_models:
                log.info("seed=%d model=%s checkpoint exists; skipping", seed, model_name)
                continue
            buckets = {"performance": [], "groups": [], "gaps": [], "support": [], "weights": []}
            log.info("seed=%d model=%s baseline fit", seed, model_name)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(Xtr_t, ytr, sample_weight=wtr)
                for warn in caught:
                    if issubclass(warn.category, ConvergenceWarning):
                        log.warning("%s convergence warning: %s", model_name, warn.message)
            score_va = model.predict_proba(Xva_t)[:, 1]
            score_te = model.predict_proba(Xte_t)[:, 1]
            base_threshold = youden_threshold(yva, score_va)
            baseline_pred = (score_te >= base_threshold).astype(float)
            evaluate_one(yte, baseline_pred, score_te, wte, sens_te, seed, model_name,
                         "baseline", "none", buckets["performance"], buckets["groups"], buckets["gaps"])

            for axis in TARGET_AXES:
                log.info("seed=%d model=%s reweighing axis=%s", seed, model_name, axis)
                a_tr, collapse, train_support = group_fit_map(train[axis].to_numpy(), ytr)
                a_va, a_te = collapse(val[axis].to_numpy()), collapse(test[axis].to_numpy())
                rw, factors = reweighing_weights(a_tr, ytr, wtr_raw)
                buckets["weights"].append({
                    "seed": seed, "model": model_name, "target_axis": axis,
                    "combined_weight_min": float(rw.min()), "combined_weight_median": float(np.median(rw)),
                    "combined_weight_max": float(rw.max()), "reweigh_factor_min": float(factors.min()),
                    "reweigh_factor_median": float(np.median(factors)), "reweigh_factor_max": float(factors.max()),
                    "pooled_training_groups": int((a_tr == "__LOW_SUPPORT_POOLED__").sum()),
                    "supported_training_groups": len(set(a_tr) - {"__LOW_SUPPORT_POOLED__"}),
                })
                rw_model = make_models(seed)[model_name]
                rw_model.fit(Xtr_t, ytr, sample_weight=rw)
                score_rw_va = rw_model.predict_proba(Xva_t)[:, 1]
                score_rw_te = rw_model.predict_proba(Xte_t)[:, 1]
                rw_threshold = youden_threshold(yva, score_rw_va)
                rw_pred = (score_rw_te >= rw_threshold).astype(float)
                evaluate_one(yte, rw_pred, score_rw_te, wte, sens_te, seed, model_name,
                             "reweighing", axis, buckets["performance"], buckets["groups"], buckets["gaps"])

                log.info("seed=%d model=%s Fairlearn equalized odds axis=%s", seed, model_name, axis)
                a_va_cal, valid_groups = calibration_groups(a_va, yva, CAL_MIN_CLASS)
                keep = np.isin(a_va_cal, list(valid_groups))
                eo_pred = (score_te >= base_threshold).astype(float)
                if len(valid_groups) >= 2:
                    policy = fit_equalized_odds(score_va[keep], yva[keep], a_va_cal[keep])
                    a_te_cal = map_prediction_groups(a_te, valid_groups)
                    eo_pred = apply_equalized_odds(
                        policy, score_te, a_te_cal, valid_groups, base_threshold
                    )
                else:
                    policy = None
                    log.warning("seed=%d model=%s target=%s: fewer than two validation groups; using baseline rule",
                                seed, model_name, axis)
                evaluate_one(yte, eo_pred, score_te, wte, sens_te, seed, model_name,
                             "equalized_odds_postprocessing", axis, buckets["performance"], buckets["groups"], buckets["gaps"])
                # Store training support as aggregate audit provenance.
                for group, row in train_support.iterrows():
                    buckets["support"].append({"seed": seed, "model": model_name, "target_axis": axis,
                                         "group": group, "train_n": int(row.n),
                                         "train_positive_n": int(row.positive_n),
                                         "train_negative_n": int(row.negative_n),
                                         "used_as_group": group in set(a_tr)})
                del rw_model, score_rw_va, score_rw_te, policy
                gc.collect()
            save_model_checkpoint(seed, model_name, buckets)
            log.info("seed=%d model=%s checkpoint saved (%d conditions)", seed, model_name,
                     len(buckets["performance"]))
            del score_va, score_te, model
            gc.collect()
        log.info("seed=%d completed; elapsed %.1f minutes", seed, (time.time() - start) / 60)
        del train, val, test, Xtr_t, Xva_t, Xte_t, transform
        gc.collect()

    performance = load_checkpoint_table("performance")
    groups = load_checkpoint_table("groups")
    # Recompute gaps from the single source of truth (held-out group rates),
    # keeping weighted and unweighted metrics on the same row.
    gaps = gaps_from_group_table(groups)
    support = load_checkpoint_table("support")
    weight_diagnostics = load_checkpoint_table("weights")
    performance.to_csv(OUT / "performance_by_split.csv", index=False)
    groups.to_csv(OUT / "group_metrics_by_split.csv", index=False)
    gaps.to_csv(OUT / "fairness_gaps_by_split.csv", index=False)
    support.to_csv(OUT / "mitigation_training_support.csv", index=False)
    weight_diagnostics.to_csv(OUT / "combined_weight_diagnostics.csv", index=False)
    # Summaries are descriptive across three overlapping random partitions.
    numeric_perf = [c for c in performance.columns if c.endswith("_w") or c.endswith("_unw")]
    perf_summary = performance.groupby(["model", "method", "target_axis"], dropna=False)[numeric_perf].agg(["mean", "min", "max"])
    perf_summary.columns = [f"{a}_{b}" for a, b in perf_summary.columns]
    perf_summary.reset_index().to_csv(OUT / "performance_summary.csv", index=False)
    gap_numeric = [c for c in gaps.columns if c.endswith("_w") or c.endswith("_unw")]
    gap_summary = gaps.groupby(["model", "method", "target_axis", "audit_axis", "gap_scope"], dropna=False)[gap_numeric].agg(["mean", "min", "max"])
    gap_summary.columns = [f"{a}_{b}" for a, b in gap_summary.columns]
    gap_summary.reset_index().to_csv(OUT / "fairness_gap_summary.csv", index=False)
    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "seeds": SEEDS, "target_axes": TARGET_AXES, "audit_axes": AUDIT_AXES,
        "support_per_class": MIN_CELL_SUPPORT, "eo_validation_min_per_class": CAL_MIN_CLASS,
        "eo_implementation": "Fairlearn ThresholdOptimizer; unweighted validation fit",
        "n_eligible_rows": int(len(d)), "n_features": len(FEATURES),
    }
    try:
        import xgboost
        environment["xgboost_version"] = xgboost.__version__
    except Exception:
        pass
    (OUT / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    split_sizes = []
    for seed in SEEDS:
        ix_train, ix_val, ix_test = split_indices(d.y.to_numpy(), d._STATE.to_numpy(), seed)
        split_sizes.append({"seed": seed, "n_train": len(ix_train), "n_validation": len(ix_val), "n_test": len(ix_test)})
    (OUT / "split_sizes.json").write_text(json.dumps(split_sizes, indent=2), encoding="utf-8")
    log.info("All runs complete in %.1f minutes; outputs are aggregate-only", (time.time() - start) / 60)


if __name__ == "__main__":
    main()

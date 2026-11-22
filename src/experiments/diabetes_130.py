"""Patient-disjoint Diabetes 130 run harmonized to Yale/BRFSS conditions."""
from __future__ import annotations

import argparse
import gc
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
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from common.paths import PROJECT_ROOT, run_results_dir, raw_data_dir
from common.config import DEFAULT_SEEDS, MODEL_NAMES
from common.models import make_model as _shared_make_model

ROOT = PROJECT_ROOT
DATA = raw_data_dir("diabetes_130") / "diabetic_data.csv"
OUT = run_results_dir("diabetes_130")
LOG = OUT / "logs/harmonized.log"
SEEDS = DEFAULT_SEEDS
MODELS = MODEL_NAMES
MIN_TRAIN_SUPPORT = 100
MIN_EO_VALIDATION_SUPPORT = 20
MIN_TEST_SUPPORT = 100
MIN_SENSITIVITY_SUPPORT = 20
TARGET_AXES = ["gender", "race", "age", "race_x_gender", "race_x_gender_x_age"]
AUDIT_AXES = ["race", "gender", "age", "race_x_gender", "race_x_age", "gender_x_age", "race_x_gender_x_age"]
SENSITIVE = ["race_audit", "gender_audit", "age_audit"]
DROP_FEATURES = {"encounter_id", "patient_nbr", "readmitted", "weight", "race", "gender", "age"}
EO_LOW = "__LOW_SUPPORT_POOLED__"


def setup_logger():
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(LOG, mode="a"), logging.StreamHandler()], force=True)
    return logging.getLogger("diabetes130_harmonized")


def load_data():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.DtypeWarning)
        d = pd.read_csv(DATA, na_values=["?"], low_memory=False)
    d = d[d.readmitted.isin(["<30", ">30", "NO"])].copy().reset_index(drop=True)
    d["y"] = (d.readmitted == "<30").astype("int8")
    d["race_audit"] = d.race.fillna("Missing/Unknown").astype(str)
    d["gender_audit"] = d.gender.fillna("Missing/Unknown").replace({"Unknown/Invalid": "Missing/Unknown"}).astype(str)
    d["age_audit"] = d.age.fillna("Missing/Unknown").astype(str)
    d["race_x_gender"] = d.race_audit + " × " + d.gender_audit
    d["race_x_age"] = d.race_audit + " × " + d.age_audit
    d["gender_x_age"] = d.gender_audit + " × " + d.age_audit
    d["race_x_gender_x_age"] = d.race_audit + " × " + d.gender_audit + " × " + d.age_audit
    xcols = [c for c in d.columns if c not in DROP_FEATURES and c not in SENSITIVE and
             c not in {"y", "race_x_gender", "race_x_age", "gender_x_age", "race_x_gender_x_age"}]
    X = d[xcols].copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype("object")
    return d, X, xcols


def grouped_split(y, patient, seed):
    y = np.asarray(y, dtype=int)
    patient = np.asarray(patient)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_val, test = next(outer.split(np.zeros(len(y)), y, patient))
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 1000)
    train_local, val_local = next(inner.split(np.zeros(len(train_val)), y[train_val], patient[train_val]))
    train, val = train_val[train_local], train_val[val_local]
    return train, val, test


def make_transformer(X_train):
    categorical = [c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])]
    numeric = [c for c in X_train.columns if c not in categorical]
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                         ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50))])
    return ColumnTransformer([("numeric", num_pipe, numeric), ("categorical", cat_pipe, categorical)],
                             sparse_threshold=0.3)


def make_model(name, seed):
    return _shared_make_model(name, seed)


def support_map(a_train, y_train, cutoff):
    frame = pd.DataFrame({"a": np.asarray(a_train, dtype=str), "y": np.asarray(y_train, dtype=int)})
    counts = frame.groupby("a", observed=True).y.agg(n="size", positive_n="sum")
    counts["negative_n"] = counts.n - counts.positive_n
    active = set(counts.index[(counts.positive_n >= cutoff) & (counts.negative_n >= cutoff)])
    def collapse(values):
        values = np.asarray(values, dtype=str)
        return np.where(np.isin(values, list(active)), values, EO_LOW)
    return active, collapse, counts


def make_axis_values(d, axis):
    mapping = {
        "race": "race_audit", "gender": "gender_audit", "age": "age_audit",
        "race_x_gender": "race_x_gender", "race_x_age": "race_x_age",
        "gender_x_age": "gender_x_age", "race_x_gender_x_age": "race_x_gender_x_age",
    }
    return d[mapping[axis]].astype(str).to_numpy()


def reweigh_weights(a, y):
    a, y = np.asarray(a, dtype=str), np.asarray(y, dtype=int)
    df = pd.DataFrame({"a": a, "y": y})
    n = len(df)
    pa = df.a.value_counts() / n
    py = df.y.value_counts() / n
    pay = df.groupby(["a", "y"], observed=True).size() / n
    w = np.asarray([pa.loc[g] * py.loc[k] / pay.loc[(g, k)] for g, k in zip(a, y)], dtype=float)
    return w / w.mean(), w


def youden_threshold(y, scores):
    fpr, tpr, thresholds = roc_curve(y, scores)
    return float(thresholds[np.argmax(tpr - fpr)])


def expected_metrics(y, positive_probability, score):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(positive_probability, dtype=float), 0, 1)
    score = np.asarray(score, dtype=float)
    pos, neg = y == 1, y == 0
    tp, fp = float(p[pos].sum()), float(p[neg].sum())
    fn, tn = float((1-p[pos]).sum()), float((1-p[neg]).sum())
    pred_pos = tp + fp
    tpr, fpr = tp / pos.sum() if pos.any() else np.nan, fp / neg.sum() if neg.any() else np.nan
    mcc_den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return {
        "n": len(y), "positive_n": int(pos.sum()), "negative_n": int(neg.sum()),
        "accuracy": (tp+tn)/len(y), "balanced_accuracy": (tpr + 1-fpr)/2,
        "tpr": tpr, "fpr": fpr, "specificity": 1-fpr,
        "ppv": tp/pred_pos if pred_pos else np.nan, "selection_rate": pred_pos/len(y),
        "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,
        "mcc": (tp*tn-fp*fn)/mcc_den if mcc_den else np.nan,
        "roc_auc": roc_auc_score(y, score), "average_precision": average_precision_score(y, score),
    }


def is_substantive(g):
    return "Missing" not in str(g) and "Unknown" not in str(g) and EO_LOW not in str(g)


def record_condition(y, pred_p, score, d_test, base, performance, group_rows):
    metrics = expected_metrics(y, pred_p, score)
    performance.append({**base, **metrics})
    for axis in AUDIT_AXES:
        a = make_axis_values(d_test, axis)
        for g in sorted(np.unique(a)):
            idx = a == g
            yp, pp = y[idx], pred_p[idx]
            pos, neg = yp == 1, yp == 0
            pos_n, neg_n = int(pos.sum()), int(neg.sum())
            tp = float(pp[pos].sum()) if pos_n else np.nan
            fp = float(pp[neg].sum()) if neg_n else np.nan
            pos_pred = tp + fp if np.isfinite(tp) and np.isfinite(fp) else np.nan
            group_rows.append({
                **base, "audit_axis": axis, "group": g, "n": int(idx.sum()),
                "positive_n": pos_n, "negative_n": neg_n,
                "adequate_support": pos_n >= MIN_TEST_SUPPORT and neg_n >= MIN_TEST_SUPPORT,
                "substantive_group": is_substantive(g),
                "tpr": tp/pos_n if pos_n else np.nan, "fpr": fp/neg_n if neg_n else np.nan,
                "selection_rate": float(pp.sum()/idx.sum()),
                "ppv": tp/pos_pred if pos_pred and pos_pred > 0 else np.nan,
            })


def save_checkpoint(seed, model, tables):
    folder = OUT / "checkpoints" / f"seed_{seed}_{model}"
    folder.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(folder / f"{name}.csv", index=False)
    (folder / "DONE").write_text("complete\n", encoding="utf-8")


def checkpoint_table(name):
    paths = sorted((OUT / "checkpoints").glob(f"seed_*_*/{name}.csv"))
    frames = [pd.read_csv(p) for p in paths if p.stat().st_size > 1]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    seeds = [args.seed] if args.seed else SEEDS
    models = [args.model] if args.model else MODELS
    log = setup_logger()
    start = time.time()
    if args.restart and (OUT / "checkpoints").exists():
        import shutil
        shutil.rmtree(OUT / "checkpoints")
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    log.info("Loading UCI Diabetes 130, binary early-readmission task")
    d, X, feature_cols = load_data()
    y = d.y.to_numpy(dtype=int)
    patient = d.patient_nbr.to_numpy()
    log.info("eligible encounters=%d, patients=%d, prevalence=%.4f, features=%d",
             len(d), pd.Series(patient).nunique(), y.mean(), len(feature_cols))

    all_splits = []
    for seed in seeds:
        tr, va, te = grouped_split(y, patient, seed)
        completed = {m for m in models if (OUT / "checkpoints" / f"seed_{seed}_{m}" / "DONE").exists()}
        if len(completed) == len(models):
            log.info("seed=%d requested model checkpoints exist; skipping", seed)
            continue
        p_tr, p_va, p_te = patient[tr], patient[va], patient[te]
        if (set(p_tr) & set(p_va)) or (set(p_tr) & set(p_te)) or (set(p_va) & set(p_te)):
            raise RuntimeError("Patient overlap across train, validation, and test")
        all_splits.append({"seed": seed, "n_train": len(tr), "n_validation": len(va), "n_test": len(te),
                           "train_patients": len(np.unique(p_tr)), "validation_patients": len(np.unique(p_va)),
                           "test_patients": len(np.unique(p_te)), "train_test_patient_overlap": 0,
                           "train_validation_patient_overlap": 0, "validation_test_patient_overlap": 0,
                           "positive_train": int(y[tr].sum()), "positive_validation": int(y[va].sum()),
                           "positive_test": int(y[te].sum())})
        transform = make_transformer(X.iloc[tr])
        Xtr, Xva, Xte = transform.fit_transform(X.iloc[tr]), transform.transform(X.iloc[va]), transform.transform(X.iloc[te])
        ytr, yva, yte = y[tr], y[va], y[te]
        dtr, dva, dte = d.iloc[tr], d.iloc[va], d.iloc[te]
        log.info("seed=%d split sizes train/val/test=%d/%d/%d transformed=%s", seed, len(tr), len(va), len(te), Xtr.shape)

        for model_name in models:
            if model_name in completed:
                log.info("seed=%d model=%s checkpoint exists; skipping", seed, model_name)
                continue
            tables = {"performance": [], "groups": [], "support": [], "fold": []}
            model = make_model(model_name, seed)
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(Xtr, ytr)
            for w in ws:
                if issubclass(w.category, ConvergenceWarning):
                    log.warning("seed=%d %s convergence warning: %s", seed, model_name, w.message)
            score_va, score_te = model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1]
            baseline_cut = youden_threshold(yva, score_va)
            baseline_pred = (score_te >= baseline_cut).astype(float)
            record_condition(yte, baseline_pred, score_te, dte,
                             {"seed": seed, "model": model_name, "method": "baseline", "target_axis": "none"},
                             tables["performance"], tables["groups"])

            for axis in TARGET_AXES:
                a_train = make_axis_values(dtr, axis)
                active, collapse, train_counts = support_map(a_train, ytr, MIN_TRAIN_SUPPORT)
                a_tr, a_va, a_te = collapse(a_train), collapse(make_axis_values(dva, axis)), collapse(make_axis_values(dte, axis))
                rw, raw_factors = reweigh_weights(a_tr, ytr)
                tables["support"].append({"seed": seed, "model": model_name, "target_axis": axis,
                                           "supported_train_groups": len(active),
                                           "pooled_train_encounters": int((a_tr == EO_LOW).sum()),
                                           "reweigh_factor_max": float(raw_factors.max()),
                                           "reweigh_factor_min": float(raw_factors.min()),
                                           "combined_weight_max": float(rw.max()),
                                           "combined_weight_min": float(rw.min())})
                for group, row in train_counts.iterrows():
                    tables["support"].append({"seed": seed, "model": model_name, "target_axis": axis,
                                               "support_group": group, "train_n": int(row.n),
                                               "train_positive_n": int(row.positive_n),
                                               "train_negative_n": int(row.negative_n),
                                               "used_as_mitigation_group": group in active})
                rw_model = make_model(model_name, seed)
                rw_model.fit(Xtr, ytr, sample_weight=rw)
                rw_va, rw_te = rw_model.predict_proba(Xva)[:, 1], rw_model.predict_proba(Xte)[:, 1]
                rw_cut = youden_threshold(yva, rw_va)
                record_condition(yte, (rw_te >= rw_cut).astype(float), rw_te, dte,
                                 {"seed": seed, "model": model_name, "method": "reweighing", "target_axis": axis},
                                 tables["performance"], tables["groups"])

                # EO is calibrated on validation data. Pool groups with weak
                # train support or fewer than 20 examples of either class in calibration.
                val_active = set(active)
                raw_va = make_axis_values(dva, axis)
                val_counts = pd.DataFrame({"a": raw_va, "y": yva}).groupby("a", observed=True).y.agg(n="size", positive_n="sum")
                val_counts["negative_n"] = val_counts.n - val_counts.positive_n
                eo_active = {g for g in val_active if g in val_counts.index and
                             val_counts.loc[g, "positive_n"] >= MIN_EO_VALIDATION_SUPPORT and
                             val_counts.loc[g, "negative_n"] >= MIN_EO_VALIDATION_SUPPORT}
                def eo_collapse(values):
                    values = np.asarray(values, dtype=str)
                    return np.where(np.isin(values, list(eo_active)), values, EO_LOW)
                eo_va, eo_te = eo_collapse(raw_va), eo_collapse(make_axis_values(dte, axis))
                # Fairlearn cannot calibrate a group with a missing outcome
                # class. Usually the pooled low-support bucket is estimable;
                # when it is not, calibrate only adequately supported groups
                # and use the validation-selected global baseline decision
                # for low-support test records.
                pooled_counts = pd.Series(yva[eo_va == EO_LOW]).value_counts()
                pool_is_estimable = (pooled_counts.get(0, 0) >= MIN_EO_VALIDATION_SUPPORT and
                                     pooled_counts.get(1, 0) >= MIN_EO_VALIDATION_SUPPORT)
                fit_mask = np.isin(eo_va, list(eo_active)) | ((eo_va == EO_LOW) & pool_is_estimable)
                policy_groups = eo_va[fit_mask]
                test_policy_mask = np.isin(eo_te, np.unique(policy_groups))
                if len(set(policy_groups)) < 2 or len(np.unique(yva[fit_mask])) < 2:
                    tables["fold"].append({"seed": seed, "model": model_name, "target_axis": axis,
                                           "status": "not_estimable", "eo_groups": len(set(policy_groups)),
                                           "pooled_low_support_estimable": bool(pool_is_estimable)})
                    continue
                post = ThresholdOptimizer(estimator=model, constraints="equalized_odds",
                                          objective="balanced_accuracy_score", prefit=True,
                                          predict_method="predict_proba")
                # Fairlearn 0.15 validates dense input here even though all
                # underlying estimators accept the sparse one-hot matrix.
                Xva_dense = Xva.toarray() if hasattr(Xva, "toarray") else Xva
                Xte_dense = Xte.toarray() if hasattr(Xte, "toarray") else Xte
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    post.fit(Xva_dense[fit_mask], yva[fit_mask], sensitive_features=policy_groups)
                    eo_pred = baseline_pred.astype(float)
                    eo_pred[test_policy_mask] = post.interpolated_thresholder_._pmf_predict(
                        Xte_dense[test_policy_mask], sensitive_features=eo_te[test_policy_mask])[:, 1]
                record_condition(yte, eo_pred, score_te, dte,
                                 {"seed": seed, "model": model_name, "method": "equalized_odds_postprocessing",
                                  "target_axis": axis}, tables["performance"], tables["groups"])
                tables["fold"].append({"seed": seed, "model": model_name, "target_axis": axis,
                                       "status": "estimated", "eo_groups": len(set(eo_va)),
                                       "eo_training_supported_groups": len(active),
                                       "eo_groups_with_validation_support": len(eo_active),
                                       "eo_validation_low_support_n": int((eo_va == EO_LOW).sum()),
                                       "pooled_low_support_estimable": bool(pool_is_estimable),
                                       "test_global_baseline_fallback_n": int((~test_policy_mask).sum())})
                del Xva_dense, Xte_dense
                del rw_model, rw_va, rw_te, post
                gc.collect()
                log.info("seed=%d model=%s target=%s completed", seed, model_name, axis)
            save_checkpoint(seed, model_name, tables)
            log.info("seed=%d model=%s checkpoint saved: %d conditions", seed, model_name, len(tables["performance"]))
            del score_va, score_te, model
            gc.collect()
        del Xtr, Xva, Xte, transform
        gc.collect()

    # Rebuild combined aggregate files from all complete per-model checkpoints.
    performance = checkpoint_table("performance")
    groups = checkpoint_table("groups")
    support = checkpoint_table("support")
    fold = checkpoint_table("fold")
    performance.to_csv(OUT / "performance_by_split.csv", index=False)
    groups.to_csv(OUT / "group_metrics_by_split.csv", index=False)
    support.to_csv(OUT / "mitigation_support_by_split.csv", index=False)
    fold.to_csv(OUT / "eo_calibration_support.csv", index=False)
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                   "fairlearn": "0.15.0", "scipy": scipy.__version__,
                   "seeds_configured": SEEDS, "models_configured": MODELS, "seeds_run_this_call": seeds,
                   "models_run_this_call": models, "target_axes": TARGET_AXES, "audit_axes": AUDIT_AXES,
                   "n_eligible_encounters": len(d), "n_features": len(feature_cols)}
    import xgboost
    environment["xgboost_version"] = xgboost.__version__
    (OUT / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    # Persist exact split checks for every configured seed even when resuming.
    manifest_path = OUT / "split_manifest.json"
    if manifest_path.exists():
        splits = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        splits = []
        for seed in SEEDS:
            tr, va, te = grouped_split(y, patient, seed)
            splits.append({"seed": seed, "train_n": len(tr), "validation_n": len(va), "test_n": len(te),
                           "train_patients": int(pd.Series(patient[tr]).nunique()),
                           "validation_patients": int(pd.Series(patient[va]).nunique()),
                           "test_patients": int(pd.Series(patient[te]).nunique()),
                           "patient_overlap_n": int(len(set(patient[tr]) & set(patient[va]))+
                                                      len(set(patient[tr]) & set(patient[te]))+
                                                      len(set(patient[va]) & set(patient[te])))})
        manifest_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    log.info("Harmonized run aggregation complete in %.1f minutes; conditions=%d", (time.time()-start)/60, len(performance))


if __name__ == "__main__":
    main()

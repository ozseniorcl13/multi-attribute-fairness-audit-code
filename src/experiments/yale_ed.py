"""Run the Yale ED multi-attribute/multimethod fairness experiment.

Outputs contain fold-level aggregate results only, not encounter-level
predictions. The public release lacks a patient identifier, so partitions are
visit-level and conclusions are descriptive.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import EqualizedOdds
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.exceptions import ConvergenceWarning


from common.paths import PROJECT_ROOT, run_results_dir, raw_data_dir
from common.config import DEFAULT_SEEDS
from common.models import make_models as _shared_make_models

ROOT = PROJECT_ROOT
RAW = raw_data_dir("yale_ed") / "5v_cleandf.RData"
OUT = run_results_dir("yale_ed")
MIN_CELL_SUPPORT = 100
EO_BOUND = 0.02
# Mitigation targets include individual attributes and intersectional groups.
TARGET_AXES = ["age_band", "race_x_gender_x_age_band", "gender", "race", "ethnicity", "race_x_gender"]
AUDIT_AXES = [
    "gender", "race", "ethnicity", "age_band", "race_x_gender",
    "race_x_ethnicity", "gender_x_ethnicity", "race_x_age_band",
    "gender_x_age_band", "ethnicity_x_age_band", "race_x_gender_x_age_band",
]


def sensitive_frame(d: pd.DataFrame) -> pd.DataFrame:
    out = d[["gender", "race", "ethnicity"]].astype("object").copy()
    age = pd.to_numeric(d["age"], errors="coerce")
    out["age_band"] = pd.cut(
        age, bins=[-np.inf, 39, 64, np.inf], labels=["18-39", "40-64", "65+"], right=True
    ).astype("object")
    for col in out:
        out[col] = out[col].where(out[col].notna(), "Missing").astype(str)
    out["race_x_gender"] = out["race"] + " × " + out["gender"]
    out["race_x_ethnicity"] = out["race"] + " × " + out["ethnicity"]
    out["gender_x_ethnicity"] = out["gender"] + " × " + out["ethnicity"]
    out["race_x_age_band"] = out["race"] + " × " + out["age_band"]
    out["gender_x_age_band"] = out["gender"] + " × " + out["age_band"]
    out["ethnicity_x_age_band"] = out["ethnicity"] + " × " + out["age_band"]
    out["race_x_gender_x_age_band"] = out["race_x_gender"] + " × " + out["age_band"]
    return out


def prepare_data(d: pd.DataFrame):
    y = d["disposition"].map({"Admit": 1, "Discharge": 0})
    keep = y.notna().to_numpy()
    d = d.loc[keep].reset_index(drop=True)
    y = y.loc[keep].astype(int).to_numpy()
    sens = sensitive_frame(d).reset_index(drop=True)

    # Conservative triage-time feature set. Direct sensitive attributes and
    # historical/post-triage fields are kept out of X in the primary analysis.
    cat = [c for c in ["dep_name", "esi", "arrivalmode", "arrivalmonth", "arrivalday", "arrivalhour_bin"] if c in d]
    cat += [c for c in d.columns if c.startswith("cc_")]
    num = [c for c in d.columns if c.startswith("triage_vital_")]
    if not cat or not num:
        raise ValueError("Triage categorical/numeric feature groups were not found")
    X = d[cat + num].copy()
    for c in cat:
        X[c] = X[c].astype("object")
    site = d["dep_name"].astype("object").where(d["dep_name"].notna(), "Missing").astype(str)
    return X, y, sens, site, cat, num


def stratified_three_way(y, site, seed):
    strat = pd.Series(site.astype(str) + "|y=" + y.astype(str))
    all_ix = np.arange(len(y))
    dev_ix, test_ix = train_test_split(
        all_ix, test_size=0.20, random_state=seed, stratify=strat
    )
    dev_strat = strat.iloc[dev_ix]
    train_ix, val_ix = train_test_split(
        dev_ix, test_size=0.25, random_state=seed + 1000, stratify=dev_strat
    )
    return train_ix, val_ix, test_ix


def make_transformer(cat, num):
    return ColumnTransformer(
        [
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]), num),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50)),
            ]), cat),
        ],
        sparse_threshold=0.3,
    )


def make_models(seed):
    return _shared_make_models(seed)


def group_map_for_fit(a_fit, y_fit, a_apply=None):
    frame = pd.DataFrame({"a": np.asarray(a_fit, dtype=str), "y": y_fit})
    tab = frame.groupby("a", observed=True).y.agg(n="size", positive_n="sum")
    tab["negative_n"] = tab["n"] - tab["positive_n"]
    supported = set(tab.index[(tab.positive_n >= MIN_CELL_SUPPORT) & (tab.negative_n >= MIN_CELL_SUPPORT)])
    if len(supported) < 2:
        return None, tab
    def collapse(values):
        values = np.asarray(values, dtype=str)
        return np.where(np.isin(values, list(supported)), values, "__LOW_SUPPORT_POOLED__")
    return collapse(a_fit), (tab, supported, collapse)


def reweighing_weights(a, y):
    a = np.asarray(a, dtype=str)
    y = np.asarray(y, dtype=int)
    table = pd.crosstab(a, y, normalize=False)
    p_a = table.sum(axis=1) / len(y)
    p_y = table.sum(axis=0) / len(y)
    p_ay = table / len(y)
    weights = np.asarray([p_a.loc[g] * p_y.loc[k] / p_ay.loc[g, k] for g, k in zip(a, y)], dtype=float)
    weights /= weights.mean()
    return weights


def youden_cut(y, score):
    fpr, tpr, thresholds = roc_curve(y, score)
    return float(thresholds[np.nanargmax(tpr - fpr)])


def expected_stats(y, p):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    pos, neg = y == 1, y == 0
    tp, fn = float(p[pos].sum()), float((1 - p[pos]).sum())
    fp, tn = float(p[neg].sum()), float((1 - p[neg]).sum())
    tpr = tp / pos.sum() if pos.any() else np.nan
    fpr = fp / neg.sum() if neg.any() else np.nan
    pred_pos = tp + fp
    return {
        "n": int(len(y)), "positive_n": int(pos.sum()), "negative_n": int(neg.sum()),
        "accuracy_expected": (tp + tn) / len(y),
        "balanced_accuracy_expected": (tpr + (tn / neg.sum() if neg.any() else np.nan)) / 2,
        "tpr_expected": tpr, "fpr_expected": fpr,
        "specificity_expected": tn / neg.sum() if neg.any() else np.nan,
        "ppv_expected": tp / pred_pos if pred_pos else np.nan,
        "selection_rate_expected": pred_pos / len(y),
        "f1_expected": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan,
    }


def write_group_metrics(rows, *, y, p, sens, model, method, target_axis, seed):
    for axis in AUDIT_AXES:
        groups = sens[axis].astype(str).to_numpy()
        cells = []
        for group in pd.unique(groups):
            mask = groups == group
            stats = expected_stats(y[mask], p[mask])
            supported = stats["positive_n"] >= MIN_CELL_SUPPORT and stats["negative_n"] >= MIN_CELL_SUPPORT
            cells.append((group, mask, stats, supported))
            rows.append({
                "seed": seed, "model": model, "method": method,
                "target_axis": target_axis, "audit_axis": axis, "group": group,
                "support_status": "adequate" if supported else "low_support",
                **stats,
            })
        good = [stats for _, _, stats, ok in cells if ok]
        gaps = {
            "tpr_gap": max((z["tpr_expected"] for z in good), default=np.nan) - min((z["tpr_expected"] for z in good), default=np.nan) if good else np.nan,
            "fpr_gap": max((z["fpr_expected"] for z in good), default=np.nan) - min((z["fpr_expected"] for z in good), default=np.nan) if good else np.nan,
            "selection_rate_gap": max((z["selection_rate_expected"] for z in good), default=np.nan) - min((z["selection_rate_expected"] for z in good), default=np.nan) if good else np.nan,
            "groups_in_gap_n": len(good), "groups_low_support_n": len(cells) - len(good),
        }
        yield {"seed": seed, "model": model, "method": method, "target_axis": target_axis, "audit_axis": axis, **gaps}


def save_checkpoint(overall, group_rows, gap_rows, support_rows, split_rows, log_rows):
    pd.DataFrame(overall).to_csv(OUT / "overall_by_split.csv", index=False)
    pd.DataFrame(group_rows).to_csv(OUT / "group_metrics_by_split.csv", index=False)
    pd.DataFrame(gap_rows).to_csv(OUT / "fairness_gaps_by_split.csv", index=False)
    pd.DataFrame(support_rows).to_csv(OUT / "mitigation_group_support.csv", index=False)
    pd.DataFrame(split_rows).to_csv(OUT / "split_manifest.csv", index=False)
    pd.DataFrame(log_rows).to_csv(OUT / "run_log.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=str(DEFAULT_SEEDS[0]), help="Comma-separated visit-level split seeds")
    parser.add_argument("--methods", default="baseline,reweighing,threshold_optimizer")
    parser.add_argument("--models", default="logistic_regression,random_forest,xgboost")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = str(run_results_dir("yale_ed"))
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    methods = set(args.methods.split(","))
    model_names = set(args.models.split(","))
    global OUT
    OUT = Path(args.output_dir).resolve()
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading the public de-identified Yale ED release", flush=True)
    rdata = pyreadr.read_r(str(RAW))
    if "df" not in rdata:
        raise ValueError(f"Expected R object 'df', found {list(rdata)}")
    d = rdata["df"]
    X, y, sens, site, cat, num = prepare_data(d)
    del d, rdata
    gc.collect()
    print(f"Prepared {len(X):,} ED visits, {X.shape[1]} candidate columns; patient ID absent", flush=True)

    overall_rows, group_rows, gap_rows, split_rows, support_rows, log_rows = [], [], [], [], [], []
    for seed in seeds:
        train_ix, val_ix, test_ix = stratified_three_way(y, site, seed)
        if "threshold_optimizer" in methods:
            # ThresholdOptimizer requires both outcome classes in each group
            # fitted on validation and cannot predict for unseen test groups.
            for axis in TARGET_AXES:
                a_train, mapping = group_map_for_fit(sens[axis].iloc[train_ix], y[train_ix])
                if mapping is None:
                    raise ValueError(f"seed={seed}, axis={axis}: fewer than two supported training groups")
                collapse = mapping[2]
                a_val = collapse(sens[axis].iloc[val_ix])
                a_test = collapse(sens[axis].iloc[test_ix])
                tab = pd.crosstab(a_val, y[val_ix]).reindex(columns=[0, 1], fill_value=0)
                valid = tab.index[(tab[0] > 0) & (tab[1] > 0)]
                fallback_n = int((~np.isin(a_test, valid)).sum())
                if len(valid) < 2:
                    raise ValueError(f"seed={seed}, axis={axis}: fewer than two valid EO validation groups")
                print(f"seed={seed} target={axis} valid EO groups={len(valid)} fallback test visits={fallback_n}", flush=True)
        transform = make_transformer(cat, num)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            X_train = transform.fit_transform(X.iloc[train_ix])
            X_val = transform.transform(X.iloc[val_ix])
            X_test = transform.transform(X.iloc[test_ix])
        print(f"seed={seed} rows train/val/test={len(train_ix):,}/{len(val_ix):,}/{len(test_ix):,}; features after encoding={X_train.shape[1]}", flush=True)
        split_rows.append({
            "seed": seed, "train_visits": len(train_ix), "validation_visits": len(val_ix),
            "test_visits": len(test_ix), "split_unit": "visit (patient IDs unavailable)",
        })
        models = make_models(seed)
        models = {k: v for k, v in models.items() if k in model_names}
        baseline_scores = {}
        baseline_preds = {}
        for name, model in models.items():
            started = time.time()
            with warnings.catch_warnings(record=True) as fit_warnings:
                warnings.simplefilter("always", category=ConvergenceWarning)
                model.fit(X_train, y[train_ix])
            score_val = model.predict_proba(X_val)[:, 1]
            score_test = model.predict_proba(X_test)[:, 1]
            cut = youden_cut(y[val_ix], score_val)
            pred = (score_test >= cut).astype(float)
            baseline_scores[name] = (score_val, score_test, cut)
            baseline_preds[name] = pred
            if "baseline" in methods:
                stats = expected_stats(y[test_ix], pred)
                overall_rows.append({
                    "seed": seed, "model": name, "method": "baseline", "target_axis": "none",
                    "threshold": cut, "roc_auc_score": float(roc_auc_score(y[test_ix], score_test)),
                    "average_precision_score": float(average_precision_score(y[test_ix], score_test)),
                    "fit_seconds": round(time.time() - started, 3),
                    "convergence_warning_n": sum(issubclass(w.category, ConvergenceWarning) for w in fit_warnings),
                    **stats,
                })
                gap_rows.extend(write_group_metrics(group_rows, y=y[test_ix], p=pred, sens=sens.iloc[test_ix],
                                    model=name, method="baseline", target_axis="none", seed=seed))
            log_rows.append({"seed": seed, "model": name, "method": "baseline", "seconds": round(time.time()-started,3), "status": "completed"})
            save_checkpoint(overall_rows, group_rows, gap_rows, support_rows, split_rows, log_rows)
            print(f"seed={seed} model={name} baseline completed in {time.time()-started:.1f}s", flush=True)
        save_checkpoint(overall_rows, group_rows, gap_rows, support_rows, split_rows, log_rows)

        for axis in TARGET_AXES:
            a_train0 = sens[axis].iloc[train_ix].astype(str).to_numpy()
            a_val0 = sens[axis].iloc[val_ix].astype(str).to_numpy()
            a_test0 = sens[axis].iloc[test_ix].astype(str).to_numpy()
            a_train, mapping = group_map_for_fit(a_train0, y[train_ix])
            if mapping is None:
                support_rows.append({"seed":seed,"target_axis":axis,"status":"not_estimable_fewer_than_two_supported_groups"})
                continue
            train_tab = mapping
            _, supported, collapse = train_tab
            a_val = collapse(a_val0)
            a_test = collapse(a_test0)
            val_tab = pd.crosstab(a_val, y[val_ix]).reindex(columns=[0, 1], fill_value=0)
            for group, row in train_tab[0].iterrows():
                support_rows.append({
                    "seed": seed, "target_axis": axis, "group": group,
                    "train_n": int(row.n), "train_positive_n": int(row.positive_n),
                    "train_negative_n": int(row.negative_n),
                    "mitigation_group": group if group in supported else "__LOW_SUPPORT_POOLED__",
                    "validation_negative_n": int(val_tab.loc[group if group in supported else "__LOW_SUPPORT_POOLED__", 0]) if (group if group in supported else "__LOW_SUPPORT_POOLED__") in val_tab.index else 0,
                    "validation_positive_n": int(val_tab.loc[group if group in supported else "__LOW_SUPPORT_POOLED__", 1]) if (group if group in supported else "__LOW_SUPPORT_POOLED__") in val_tab.index else 0,
                    "support_status": "adequate" if group in supported else "pooled_low_support",
                })

            if "threshold_optimizer" in methods:
                valid_eo_groups = val_tab.index[(val_tab[0] > 0) & (val_tab[1] > 0)]
                val_eo_mask = np.isin(a_val, valid_eo_groups)
                test_eo_mask = np.isin(a_test, valid_eo_groups)
                if len(valid_eo_groups) < 2:
                    raise ValueError(f"seed={seed}, axis={axis}: fewer than two valid EO validation groups")
                for name, model in models.items():
                    started = time.time()
                    post = ThresholdOptimizer(
                        estimator=model, constraints="equalized_odds",
                        objective="balanced_accuracy_score", prefit=True,
                        predict_method="predict_proba", grid_size=300,
                    )
                    post.fit(X_val[val_eo_mask], y[val_ix][val_eo_mask], sensitive_features=a_val[val_eo_mask])
                    # The pinned pandas 2.3/Fairlearn 0.15 combination returns
                    # the exact PMF of the randomized policy. The newer pandas
                    # 3.x behavior raised on an internal float32 -> float64
                    # assignment; see logs/core_retry.log.
                    p = baseline_preds[name].copy()
                    p[test_eo_mask] = post.interpolated_thresholder_._pmf_predict(
                        X_test[test_eo_mask], sensitive_features=a_test[test_eo_mask]
                    )[:, 1]
                    stats = expected_stats(y[test_ix], p)
                    overall_rows.append({
                        "seed":seed,"model":name,"method":"threshold_optimizer","target_axis":axis,
                        "threshold":np.nan,"roc_auc_score":np.nan,"average_precision_score":np.nan,
                        "eo_validation_excluded_n":int((~val_eo_mask).sum()),
                        "eo_test_fallback_n":int((~test_eo_mask).sum()),
                        "fit_seconds":round(time.time()-started,3),"convergence_warning_n":0,**stats,
                    })
                    gap_rows.extend(write_group_metrics(group_rows, y=y[test_ix], p=p, sens=sens.iloc[test_ix],
                                        model=name, method="threshold_optimizer", target_axis=axis, seed=seed))
                    log_rows.append({"seed":seed,"model":name,"method":"threshold_optimizer","target_axis":axis,"seconds":round(time.time()-started,3),"status":"completed"})
                    save_checkpoint(overall_rows, group_rows, gap_rows, support_rows, split_rows, log_rows)
                    print(f"seed={seed} target={axis} model={name} threshold_optimizer completed in {time.time()-started:.1f}s", flush=True)

            if "reweighing" in methods:
                w = reweighing_weights(a_train, y[train_ix])
                for name, model in models.items():
                    started = time.time()
                    weighted_model = clone(model)
                    with warnings.catch_warnings(record=True) as fit_warnings:
                        warnings.simplefilter("always")
                        weighted_model.fit(X_train, y[train_ix], sample_weight=w)
                    score_val = weighted_model.predict_proba(X_val)[:, 1]
                    score_test = weighted_model.predict_proba(X_test)[:, 1]
                    cut = youden_cut(y[val_ix], score_val)
                    p = (score_test >= cut).astype(float)
                    stats = expected_stats(y[test_ix], p)
                    overall_rows.append({
                        "seed":seed,"model":name,"method":"reweighing","target_axis":axis,
                        "threshold":cut,"roc_auc_score":float(roc_auc_score(y[test_ix],score_test)),
                        "average_precision_score":float(average_precision_score(y[test_ix],score_test)),
                        "fit_seconds":round(time.time()-started,3),
                        "convergence_warning_n":sum(issubclass(w.category, ConvergenceWarning) for w in fit_warnings),
                        "weight_min":float(w.min()),"weight_median":float(np.median(w)),"weight_max":float(w.max()),
                        "weight_mean":float(w.mean()),**stats,
                    })
                    gap_rows.extend(write_group_metrics(group_rows, y=y[test_ix], p=p, sens=sens.iloc[test_ix],
                                        model=name, method="reweighing", target_axis=axis, seed=seed))
                    log_rows.append({"seed":seed,"model":name,"method":"reweighing","target_axis":axis,"seconds":round(time.time()-started,3),"status":"completed"})
                    save_checkpoint(overall_rows, group_rows, gap_rows, support_rows, split_rows, log_rows)
                    print(f"seed={seed} target={axis} model={name} reweighing completed in {time.time()-started:.1f}s", flush=True)

            print(f"seed={seed} target={axis} mitigation cells completed", flush=True)

        save_checkpoint(overall_rows, group_rows, gap_rows, support_rows, split_rows, log_rows)

    config = {
        "dataset": "Yale ED public de-identified processed release",
        "seeds": seeds, "methods": sorted(methods), "models": sorted(model_names),
        "target_axes": TARGET_AXES, "audit_axes": AUDIT_AXES,
        "min_group_per_label_for_supported_gap": MIN_CELL_SUPPORT,
        "eo_difference_bound": EO_BOUND,
        "split_unit": "visit; release does not include patient IDs",
        "feature_policy": "triage-only; direct sensitive attributes and historical/post-triage variables excluded",
        "note": "Visit-level split can leak repeat patients across partitions; results are descriptive.",
    }
    (OUT / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print("Experiment run complete", flush=True)


if __name__ == "__main__":
    main()

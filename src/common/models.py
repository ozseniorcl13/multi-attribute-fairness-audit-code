"""Shared classifier definitions used by the final manuscript experiments."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from common.config import MODEL_NAMES


def make_model(name: str, seed: int):
    """Create one of the three classifiers used in the paper."""
    if name == "logistic_regression":
        return LogisticRegression(C=1.0, max_iter=1000, solver="liblinear", random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=250, min_samples_leaf=10, max_features="sqrt",
            n_jobs=4, random_state=seed,
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            tree_method="hist", eval_metric="logloss", n_jobs=4,
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {name}")


def make_models(seed: int) -> dict:
    """Create all paper classifiers with a shared random seed."""
    return {name: make_model(name, seed) for name in MODEL_NAMES}

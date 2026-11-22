"""Shared helpers for unweighted Fairlearn equalized-odds post-processing."""
from __future__ import annotations

import numpy as np
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import BaseEstimator, ClassifierMixin


LOW_VALIDATION_GROUP = "__LOW_VALIDATION_SUPPORT__"
UNCALIBRATED_GROUP = "__UNCALIBRATED__"


class FixedScores(ClassifierMixin, BaseEstimator):
    """Expose frozen model scores to Fairlearn without refitting the classifier."""

    def fit(self, X, y):
        self.classes_ = np.unique(np.asarray(y, dtype=int))
        if len(self.classes_) != 2:
            raise ValueError("Equalized-odds fitting requires both outcome classes")
        return self

    def predict_proba(self, X):
        positive = np.asarray(X, dtype=float).reshape(-1)
        return np.column_stack((1.0 - positive, positive))


def calibration_groups(groups, y, min_per_class: int):
    """Pool validation groups lacking class support; return mapped groups and fit set."""
    groups = np.asarray(groups, dtype=str)
    y = np.asarray(y, dtype=int)
    if len(groups) != len(y):
        raise ValueError("Groups and outcomes must have equal length")
    if min_per_class < 1:
        raise ValueError("min_per_class must be positive")

    labels = np.unique(groups)
    supported = {
        label for label in labels
        if np.sum((groups == label) & (y == 1)) >= min_per_class
        and np.sum((groups == label) & (y == 0)) >= min_per_class
    }
    mapped = np.where(np.isin(groups, list(supported)), groups, LOW_VALIDATION_GROUP)
    pooled = mapped == LOW_VALIDATION_GROUP
    pooled_supported = (
        np.sum(pooled & (y == 1)) >= min_per_class
        and np.sum(pooled & (y == 0)) >= min_per_class
    )
    if not pooled_supported:
        mapped[pooled] = UNCALIBRATED_GROUP
    valid = set(np.unique(mapped)) - {UNCALIBRATED_GROUP}
    return mapped, valid


def fit_equalized_odds(scores, y, groups):
    """Fit Fairlearn's standard ThresholdOptimizer without sample weights.

    The input is the validation score from an already-fitted classifier. The
    Fairlearn optimizer therefore learns only the post-processing rule and
    never refits or changes the underlying model.
    """
    scores = np.asarray(scores, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups, dtype=str)
    if len(scores) != len(y) or len(groups) != len(y):
        raise ValueError("Scores, outcomes, and groups must have equal length")
    if len(np.unique(groups)) < 2:
        raise ValueError("Equalized odds requires at least two validation groups")
    x = scores[:, None]
    estimator = FixedScores().fit(x, y)
    policy = ThresholdOptimizer(
        estimator=estimator,
        constraints="equalized_odds",
        objective="balanced_accuracy_score",
        prefit=True,
        predict_method="predict_proba",
    )
    # Intentionally no sample_weight: this is the ordinary Fairlearn objective.
    policy.fit(x, y, sensitive_features=groups)
    return policy


def apply_equalized_odds(policy, scores, groups, known_groups, fallback_threshold):
    """Return expected positive-decision probabilities, with a global fallback."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    groups = np.asarray(groups, dtype=str)
    if len(scores) != len(groups):
        raise ValueError("Scores and groups must have equal length")
    result = (scores >= float(fallback_threshold)).astype(float)
    apply = np.isin(groups, list(known_groups))
    if apply.any():
        result[apply] = policy._pmf_predict(
            scores[apply, None], sensitive_features=groups[apply]
        )[:, 1]
    return np.clip(result, 0.0, 1.0)


def map_prediction_groups(groups, known_groups):
    """Map test groups to fitted validation groups or an uncalibrated marker."""
    groups = np.asarray(groups, dtype=str)
    known = set(known_groups)
    # Low-support validation groups share a pooled rule when that rule is estimable.
    low_target = LOW_VALIDATION_GROUP if LOW_VALIDATION_GROUP in known else UNCALIBRATED_GROUP
    return np.where(np.isin(groups, list(known - {LOW_VALIDATION_GROUP})), groups, low_target)


def youden_threshold(y, scores):
    """Select the global Youden threshold using unweighted validation records."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(np.asarray(y, dtype=int), np.asarray(scores, dtype=float))
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    return float(thresholds[finite][np.argmax(tpr[finite] - fpr[finite])])

"""Supervised models for sequence classification."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def split_indices(y: np.ndarray, seed: int, test_size: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    return train_idx, test_idx


def _logistic(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def _forest(seed: int, n_estimators: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
        ]
    )


def _roc_auc(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
    positive_label: str | None = None,
) -> float:
    labels = list(classes)
    if len(labels) == 2:
        label = positive_label or str(labels[1])
        index = labels.index(label)
        # sklearn 1.9 scores the greater label, so encode the chosen class as 1.
        y_binary = (np.asarray(y_true) == label).astype(int)
        return float(roc_auc_score(y_binary, proba[:, index]))
    return float(
        roc_auc_score(
            y_true,
            proba,
            multi_class="ovr",
            average="macro",
            labels=labels,
        )
    )


def _cv_accuracy(model: Pipeline, x_train: np.ndarray, y_train: np.ndarray, seed: int, folds: int) -> tuple[float, float]:
    min_class = int(np.min(np.unique(y_train, return_counts=True)[1]))
    usable_folds = min(folds, min_class)
    if usable_folds < 2:
        return float("nan"), float("nan")
    splitter = StratifiedKFold(n_splits=usable_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(model, x_train, y_train, cv=splitter, scoring="accuracy", n_jobs=-1)
    return float(scores.mean()), float(scores.std(ddof=1) if len(scores) > 1 else 0.0)


def evaluate_models(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    cv_folds: int = 5,
    n_estimators: int = 200,
    positive_label: str | None = None,
) -> dict[str, Any]:
    fitted: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    builders = {
        "logistic_regression": lambda: _logistic(seed),
        "random_forest": lambda: _forest(seed, n_estimators),
    }
    for name, builder in builders.items():
        model = builder()
        cv_mean, cv_std = _cv_accuracy(model, x_train, y_train, seed, cv_folds)
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        probabilities = model.predict_proba(x_test)
        classes = model.classes_
        labels = list(classes)
        report = classification_report(
            y_test,
            predicted,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        fitted[name] = model
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, predicted)),
            "macro_f1": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
            "roc_auc": _roc_auc(y_test, probabilities, classes, positive_label),
            "roc_positive_label": positive_label if len(labels) == 2 else None,
            "cv_accuracy_mean": cv_mean,
            "cv_accuracy_std": cv_std,
            "confusion_matrix": confusion_matrix(y_test, predicted, labels=labels).tolist(),
            "classes": labels,
            "classification_report": report,
        }
    majority = max(set(y_train.tolist()), key=y_train.tolist().count)
    baseline = float(np.mean(y_test == majority))
    return {
        "models": fitted,
        "metrics": metrics,
        "baseline_accuracy": baseline,
        "y_train": y_train,
        "y_test": y_test,
        "x_test": x_test,
    }


def cluster_ari(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_clusters: int,
    seed: int,
) -> float:
    scaler = StandardScaler().fit(x_train)
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    model.fit(scaler.transform(x_train))
    predicted = model.predict(scaler.transform(x_test))
    return float(adjusted_rand_score(y_test, predicted))


def coding_coefficients(model: Pipeline, feature_names: list[str], positive_label: str = "coding") -> list[dict[str, float | str]]:
    """Signed logistic-regression coefficients for the coding class, largest first."""
    classifier = model.named_steps["clf"]
    classes = list(model.classes_)
    if positive_label not in classes:
        raise ValueError(f"{positive_label!r} is not a class of this model")
    coefficients = classifier.coef_
    if coefficients.shape[0] == 1:
        signed = coefficients.ravel()
        if classes[1] != positive_label:
            signed = -signed
    else:
        signed = coefficients[classes.index(positive_label)]
    order = np.argsort(np.abs(signed))[::-1]
    return [
        {"feature": feature_names[int(index)], "coefficient": float(signed[int(index)])}
        for index in order
    ]
